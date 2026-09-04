"""
Functor-path correction harness: applies each correction as a one-shot exact
direct-sum commit (SLMModel.commit) against a FROZEN base ResNet-18. No
gradient computation, no backprop -- just an O(1) exact-match override.
"""
import json
import time

import numpy as np
import torch
import torchvision

from data import make_datasets
from resnet_functor import ResNetBaseFunction, image_to_vector
from corrections import load_correction_vectors, CHECKPOINT_PATH, NUM_CLASSES
from slm_template import SLMModel, Invariants, ModelConfig, TrainingEvent, ScopeAll
from energy import measure_energy

RESULTS_PATH = "checkpoints/functor_results.jsonl"
ENERGY_BATCH_SIZE = 10  # corrections per RAPL measurement window (see energy.py notes)


def main():
    _, val_ds, _ = make_datasets()
    corrections = load_correction_vectors(val_ds)
    print(f"loaded {len(corrections)} corrections")

    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    base_f = ResNetBaseFunction(model)

    slm = SLMModel(
        f=base_f,
        invariants=Invariants(max_delta=1.0, scope_fn=ScopeAll()),
        config=ModelConfig(),
    )

    results = []
    energy_batches = []
    t_total_start = time.time()

    for batch_start in range(0, len(corrections), ENERGY_BATCH_SIZE):
        batch = corrections[batch_start:batch_start + ENERGY_BATCH_SIZE]
        with measure_energy() as reading:
            for j, (xv, yv, idx, true_label) in enumerate(batch):
                i = batch_start + j
                event = TrainingEvent(x=xv, y=yv)

                t0 = time.time()
                delta = slm.propose(event)
                samples = slm.generate_validation_samples(xv, n_samples=3, radius=0.05)
                ok = slm.validate(delta, samples)
                if ok:
                    slm.commit(delta, event=event, samples=samples)
                t1 = time.time()

                results.append({"i": i, "idx": idx, "committed": ok, "seconds": t1 - t0})

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

    # Correctness check: every committed anchor must now predict exactly the target.
    n_verified = 0
    for xv, yv, idx, true_label in corrections:
        pred = slm.predict(xv)
        if np.allclose(pred, yv):
            n_verified += 1
    print(f"verified {n_verified}/{len(corrections)} commits produce exact target on lookup")

    total_joules = sum(b["total_joules"] for b in energy_batches)
    total_marginal_joules = sum(b["marginal_joules"] for b in energy_batches)
    summary = {
        "n_corrections": len(corrections),
        "n_committed": sum(1 for r in results if r["committed"]),
        "n_verified_exact": n_verified,
        "total_seconds": t_total,
        "mean_seconds_per_commit": t_total / len(corrections),
        "total_joules": total_joules,
        "total_marginal_joules": total_marginal_joules,
        "mean_joules_per_commit": total_joules / len(corrections),
        "mean_marginal_joules_per_commit": total_marginal_joules / len(corrections),
        "final_version": slm.version,
        "state_hash": slm.get_state_hash(),
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
