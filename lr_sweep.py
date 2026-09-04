"""
Sweeps the baseline (per-correction online SGD) learning rate to find its best
realistic operating point before freezing a configuration for repeated trials.

Selection criterion (fixed in advance, per protocol discussion):
  1. Highest correction success rate (n_actually_fixed / n_corrections).
  2. Among settings within 1 fix of the best success rate, lowest marginal
     energy per correction.
Collateral drift is reported for every setting regardless, since it matters
for the final writeup even though it isn't the selection criterion.
"""
import json

import torch
import torch.nn as nn
import torchvision

from data import make_datasets
from corrections import load_correction_vectors, CHECKPOINT_PATH, NUM_CLASSES
from baseline_harness import build_control_set, eval_on_indices, N_CONTROL_SAMPLES, SEED
from energy import measure_energy

LR_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
RESULTS_PATH = "checkpoints/lr_sweep_results.jsonl"


def run_one(correction_lr, val_ds, corrections, corrected_indices):
    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    control_indices = build_control_set(model, val_ds, corrected_indices, N_CONTROL_SAMPLES, SEED)
    control_acc_before = eval_on_indices(model, val_ds, control_indices)

    opt = torch.optim.SGD(model.parameters(), lr=correction_lr, momentum=0.9, weight_decay=5e-5, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()

    with measure_energy() as reading:
        for xv, yv, idx, true_label in corrections:
            x = torch.from_numpy(xv).unsqueeze(0)
            y = torch.tensor([true_label])
            opt.zero_grad()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
    r = reading.result

    n_fixed = 0
    for xv, yv, idx, true_label in corrections:
        x = torch.from_numpy(xv).unsqueeze(0)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            pred = model(x).argmax(dim=1).item()
        model.train()
        if pred == true_label:
            n_fixed += 1

    control_acc_after = eval_on_indices(model, val_ds, control_indices)

    return {
        "correction_lr": correction_lr,
        "n_corrections": len(corrections),
        "n_fixed": n_fixed,
        "fix_rate": n_fixed / len(corrections),
        "control_acc_before": control_acc_before,
        "control_acc_after": control_acc_after,
        "collateral_drift": control_acc_before - control_acc_after,
        "total_seconds": r.seconds,
        "total_joules": r.total_joules,
        "marginal_joules": r.marginal_joules,
        "marginal_joules_per_correction": r.marginal_joules / len(corrections),
    }


def main():
    _, val_ds, _ = make_datasets()
    corrections = load_correction_vectors(val_ds)
    corrected_indices = {idx for _, _, idx, _ in corrections}
    print(f"loaded {len(corrections)} corrections; sweeping {len(LR_GRID)} learning rates")

    results = []
    for lr in LR_GRID:
        res = run_one(lr, val_ds, corrections, corrected_indices)
        results.append(res)
        print(json.dumps(res), flush=True)

    with open(RESULTS_PATH, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"wrote {RESULTS_PATH}")

    best_fix_rate = max(r["fix_rate"] for r in results)
    tolerance = 1 / len(corrections)  # "within 1 fix of the best"
    candidates = [r for r in results if r["fix_rate"] >= best_fix_rate - tolerance]
    chosen = min(candidates, key=lambda r: r["marginal_joules_per_correction"])
    print("\n=== SELECTED CONFIG ===")
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
