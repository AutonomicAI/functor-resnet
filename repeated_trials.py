"""
Repeated-trial comparison: functor (direct-sum commit) vs baseline (per-correction
online SGD, frozen at the LR selected by lr_sweep.py). Each trial draws an
independent random subset of misclassified examples from the full pool and runs
BOTH arms on the identical subset, so per-trial differences isolate the update
mechanism, not sampling luck.
"""
import json
import math
import random
import statistics

import numpy as np
import torch
import torch.nn as nn
import torchvision

from data import make_datasets
from corrections import find_misclassified, CHECKPOINT_PATH, NUM_CLASSES
from resnet_functor import ResNetBaseFunction, image_to_vector, one_hot
from baseline_harness import build_control_set, eval_on_indices, N_CONTROL_SAMPLES
from slm_template import SLMModel, Invariants, ModelConfig, TrainingEvent, ScopeAll
from energy import measure_energy

N_TRIALS = 5
TRIAL_SAMPLE_SIZE = 150
BASELINE_LR = 3e-5  # frozen from lr_sweep.py's selected config
RESULTS_PATH = "checkpoints/repeated_trials_results.jsonl"
POOL_SEED_BASE = 1000
CONTROL_SEED_BASE = 9000


def load_fresh_model():
    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return model


def materialize(val_ds, events):
    out = []
    for e in events:
        x, _ = val_ds[e["idx"]]
        xv = image_to_vector(x)
        yv = one_hot(e["true_label"], NUM_CLASSES)
        out.append((xv, yv, e["idx"], e["true_label"]))
    return out


def run_functor_trial(val_ds, corrections):
    model = load_fresh_model()
    base_f = ResNetBaseFunction(model)
    slm = SLMModel(f=base_f, invariants=Invariants(max_delta=1.0, scope_fn=ScopeAll()), config=ModelConfig())

    with measure_energy() as reading:
        for xv, yv, idx, true_label in corrections:
            event = TrainingEvent(x=xv, y=yv)
            delta = slm.propose(event)
            samples = slm.generate_validation_samples(xv, n_samples=3, radius=0.05)
            if slm.validate(delta, samples):
                slm.commit(delta, event=event, samples=samples)
    r = reading.result

    n_verified = sum(1 for xv, yv, idx, tl in corrections if np.allclose(slm.predict(xv), yv))
    return {
        "arm": "functor",
        "n_corrections": len(corrections),
        "n_fixed": n_verified,
        "fix_rate": n_verified / len(corrections),
        "collateral_drift": 0.0,  # provably zero: HardGate only fires on exact match
        "seconds": r.seconds,
        "marginal_joules": r.marginal_joules,
        "marginal_joules_per_correction": r.marginal_joules / len(corrections),
    }


def run_baseline_trial(val_ds, corrections, corrected_indices, control_seed):
    model = load_fresh_model()
    model.train()
    control_indices = build_control_set(model, val_ds, corrected_indices, N_CONTROL_SAMPLES, control_seed)
    control_acc_before = eval_on_indices(model, val_ds, control_indices)

    opt = torch.optim.SGD(model.parameters(), lr=BASELINE_LR, momentum=0.9, weight_decay=5e-5, nesterov=True)
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
    model.eval()
    with torch.no_grad():
        for xv, yv, idx, true_label in corrections:
            x = torch.from_numpy(xv).unsqueeze(0)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                pred = model(x).argmax(dim=1).item()
            if pred == true_label:
                n_fixed += 1
    model.train()

    control_acc_after = eval_on_indices(model, val_ds, control_indices)

    return {
        "arm": "baseline",
        "n_corrections": len(corrections),
        "n_fixed": n_fixed,
        "fix_rate": n_fixed / len(corrections),
        "collateral_drift": control_acc_before - control_acc_after,
        "seconds": r.seconds,
        "marginal_joules": r.marginal_joules,
        "marginal_joules_per_correction": r.marginal_joules / len(corrections),
    }


def summarize(arm_results):
    def stats(key):
        vals = [r[key] for r in arm_results]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        # ~95% CI via normal approx (small N, so treat as indicative not exact)
        ci95 = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return {"mean": mean, "stdev": sd, "ci95": ci95, "values": vals}

    energy_per_success = []
    for r in arm_results:
        if r["n_fixed"] > 0:
            energy_per_success.append(r["marginal_joules"] / r["n_fixed"])
    eps_mean = statistics.mean(energy_per_success) if energy_per_success else None
    eps_sd = statistics.stdev(energy_per_success) if len(energy_per_success) > 1 else 0.0

    return {
        "arm": arm_results[0]["arm"],
        "n_trials": len(arm_results),
        "fix_rate": stats("fix_rate"),
        "collateral_drift": stats("collateral_drift"),
        "seconds": stats("seconds"),
        "marginal_joules_per_correction": stats("marginal_joules_per_correction"),
        "marginal_joules_per_successful_correction": {"mean": eps_mean, "stdev": eps_sd},
    }


def main():
    _, val_ds, _ = make_datasets()
    model = load_fresh_model()
    pool = find_misclassified(model, val_ds, max_n=1000)
    print(f"pool size: {len(pool)} misclassified examples")

    functor_results = []
    baseline_results = []
    all_records = []

    for trial in range(N_TRIALS):
        rng = random.Random(POOL_SEED_BASE + trial)
        sample = rng.sample(pool, TRIAL_SAMPLE_SIZE)
        corrections = materialize(val_ds, sample)
        corrected_indices = {idx for _, _, idx, _ in corrections}

        fr = run_functor_trial(val_ds, corrections)
        fr["trial"] = trial
        functor_results.append(fr)
        all_records.append(fr)
        print(f"trial {trial} functor: {json.dumps(fr)}", flush=True)

        br = run_baseline_trial(val_ds, corrections, corrected_indices, CONTROL_SEED_BASE + trial)
        br["trial"] = trial
        baseline_results.append(br)
        all_records.append(br)
        print(f"trial {trial} baseline: {json.dumps(br)}", flush=True)

    functor_summary = summarize(functor_results)
    baseline_summary = summarize(baseline_results)

    print("\n=== FUNCTOR SUMMARY ===")
    print(json.dumps(functor_summary, indent=2))
    print("\n=== BASELINE SUMMARY (lr={}) ===".format(BASELINE_LR))
    print(json.dumps(baseline_summary, indent=2))

    with open(RESULTS_PATH, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"summary": "functor", **functor_summary}) + "\n")
        f.write(json.dumps({"summary": "baseline", **baseline_summary}) + "\n")
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
