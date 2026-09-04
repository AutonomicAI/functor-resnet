"""
Generates the shared set of "one-shot correction" events used by both the
functor path and the conventional-SGD baseline path.

A correction event = a validation image the frozen base model gets wrong,
paired with its true label. Both experiment arms consume the EXACT SAME
ordered list of corrections, so the energy/time comparison is apples-to-apples.
"""
import json

import numpy as np
import torch
import torchvision

from data import make_datasets
from resnet_functor import image_to_vector, one_hot

MAX_CORRECTIONS = 200
CHECKPOINT_PATH = "checkpoints/resnet18_base.pt"
OUTPUT_PATH = "checkpoints/corrections.jsonl"
NUM_CLASSES = 10


def find_misclassified(model, val_ds, max_n):
    model.eval()
    events = []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            x, y = val_ds[idx]
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(x.unsqueeze(0))
            pred = logits.argmax(dim=1).item()
            if pred != y:
                events.append({"idx": idx, "true_label": y, "pred_label": pred})
                if len(events) >= max_n:
                    break
    return events


def main():
    _, val_ds, _ = make_datasets()

    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded checkpoint: val_acc={ckpt['val_acc']:.4f} epoch={ckpt['epoch']}")

    events = find_misclassified(model, val_ds, MAX_CORRECTIONS)
    print(f"found {len(events)} misclassified examples (cap={MAX_CORRECTIONS})")

    with open(OUTPUT_PATH, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {OUTPUT_PATH}")


def load_correction_vectors(val_ds, path=OUTPUT_PATH):
    """
    Re-materializes correction events as (image_vector, target_vector, idx) tuples,
    using the canonical (deterministic) val transform so hashes match exactly what
    a future lookup of the same physical image would produce.
    """
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            x, _ = val_ds[rec["idx"]]
            xv = image_to_vector(x)
            yv = one_hot(rec["true_label"], NUM_CLASSES)
            out.append((xv, yv, rec["idx"], rec["true_label"]))
    return out


if __name__ == "__main__":
    main()
