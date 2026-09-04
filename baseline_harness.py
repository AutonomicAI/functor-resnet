"""
Baseline-path correction harness: applies each correction via a conventional
per-correction online SGD step (real forward + backward + optimizer.step())
against a live, mutable copy of the SAME base ResNet-18 checkpoint.

Unlike the functor path, gradient-descent corrections are not guaranteed to be
exact, and can have collateral effects on other, unrelated inputs (catastrophic
forgetting). Both are measured here.
"""
import json
import random
import time

import torch
import torch.nn as nn
import torchvision

from data import make_datasets
from corrections import load_correction_vectors, CHECKPOINT_PATH, NUM_CLASSES
from energy import measure_energy

CORRECTION_LR = 0.001
N_CONTROL_SAMPLES = 200  # held-out correctly-classified examples, to measure collateral drift
RESULTS_PATH = "checkpoints/baseline_results.jsonl"
SEED = 5678
ENERGY_BATCH_SIZE = 10  # corrections per RAPL measurement window (see energy.py notes)


def build_control_set(model, val_ds, corrected_indices, n, seed):
    """Pick N correctly-classified validation examples NOT among the corrections."""
    model.eval()
    rng = random.Random(seed)
    candidates = list(range(len(val_ds)))
    rng.shuffle(candidates)
    control = []
    with torch.no_grad():
        for idx in candidates:
            if idx in corrected_indices:
                continue
            x, y = val_ds[idx]
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(x.unsqueeze(0))
            pred = logits.argmax(dim=1).item()
            if pred == y:
                control.append(idx)
            if len(control) >= n:
                break
    model.train()
    return control


def eval_on_indices(model, val_ds, indices):
    model.eval()
    correct = 0
    with torch.no_grad():
        for idx in indices:
            x, y = val_ds[idx]
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(x.unsqueeze(0))
            pred = logits.argmax(dim=1).item()
            if pred == y:
                correct += 1
    model.train()
    return correct / len(indices)


def main():
    _, val_ds, _ = make_datasets()
    corrections = load_correction_vectors(val_ds)
    print(f"loaded {len(corrections)} corrections")
    corrected_indices = {idx for _, _, idx, _ in corrections}

    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    control_indices = build_control_set(model, val_ds, corrected_indices, N_CONTROL_SAMPLES, SEED)
    control_acc_before = eval_on_indices(model, val_ds, control_indices)
    print(f"control set: {len(control_indices)} images, acc_before={control_acc_before:.4f}")

    opt = torch.optim.SGD(model.parameters(), lr=CORRECTION_LR, momentum=0.9, weight_decay=5e-5, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()

    results = []
    energy_batches = []
    t_total_start = time.time()

    for batch_start in range(0, len(corrections), ENERGY_BATCH_SIZE):
        batch = corrections[batch_start:batch_start + ENERGY_BATCH_SIZE]
        with measure_energy() as reading:
            for j, (xv, yv, idx, true_label) in enumerate(batch):
                i = batch_start + j
                x = torch.from_numpy(xv).unsqueeze(0)
                y = torch.tensor([true_label])

                t0 = time.time()
                opt.zero_grad()
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                t1 = time.time()

                with torch.no_grad():
                    fixed = logits.argmax(dim=1).item() == true_label
                results.append({"i": i, "idx": idx, "loss": loss.item(), "fixed_this_step": fixed, "seconds": t1 - t0})

        r = reading.result
        energy_batches.append({
            "batch_start": batch_start,
            "batch_size": len(batch),
            "seconds": r.seconds,
            "total_joules": r.total_joules,
            "watts": r.watts,
            "marginal_joules": r.marginal_joules,
        })
        print(
            f"  batch@{batch_start}: {len(batch)} corrections, "
            f"{r.total_joules:.2f}J total, {r.marginal_joules:.2f}J marginal, {r.watts:.1f}W",
            flush=True,
        )

    t_total = time.time() - t_total_start

    # Did the corrections actually stick? (not guaranteed like the functor path)
    n_actually_fixed = 0
    for xv, yv, idx, true_label in corrections:
        x = torch.from_numpy(xv).unsqueeze(0)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            pred = model(x).argmax(dim=1).item()
        model.train()
        if pred == true_label:
            n_actually_fixed += 1

    control_acc_after = eval_on_indices(model, val_ds, control_indices)

    total_joules = sum(b["total_joules"] for b in energy_batches)
    total_marginal_joules = sum(b["marginal_joules"] for b in energy_batches)
    summary = {
        "n_corrections": len(corrections),
        "n_actually_fixed_after_all_steps": n_actually_fixed,
        "control_acc_before": control_acc_before,
        "control_acc_after": control_acc_after,
        "collateral_drift": control_acc_before - control_acc_after,
        "total_seconds": t_total,
        "mean_seconds_per_correction": t_total / len(corrections),
        "total_joules": total_joules,
        "total_marginal_joules": total_marginal_joules,
        "mean_joules_per_correction": total_joules / len(corrections),
        "mean_marginal_joules_per_correction": total_marginal_joules / len(corrections),
        "correction_lr": CORRECTION_LR,
    }
    print(json.dumps(summary, indent=2))

    with open(RESULTS_PATH, "w") as f:
        f.write(json.dumps(summary) + "\n")
        for b in energy_batches:
            f.write(json.dumps({"type": "energy_batch", **b}) + "\n")
        for r in results:
            f.write(json.dumps({"type": "step", **r}) + "\n")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
