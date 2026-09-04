import json
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from data import make_datasets

SEED = 1234
EPOCHS = 60
BATCH_SIZE = 64
LR = 0.025  # linear-scaling convention: 0.1 @ batch256 -> 0.025 @ batch64
WEIGHT_DECAY = 5e-5
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 5.0
NUM_WORKERS = 8
CHECKPOINT_PATH = "checkpoints/resnet18_base.pt"
LOG_PATH = "checkpoints/train_log.jsonl"

torch.manual_seed(SEED)
torch.set_num_threads(16)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            # autocast entered per-step, not wrapping the whole loop -- see note in main()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def main():
    import os
    os.makedirs("checkpoints", exist_ok=True)

    train_ds, val_ds, _ = make_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=NUM_WORKERS)

    print(f"train images: {len(train_ds)}, val images: {len(val_ds)}, classes: {train_ds.classes}")

    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.train()

    # No LR warmup: diagnosed and confirmed (diagnose2.py) that a linear warmup
    # ramp interacting with SGD momentum buildup was causing divergence right as
    # warmup completed. Constant/cosine-decay-from-start at this LR and batch size
    # is stable (verified over 300 steps in isolation) -- warmup is a fix for
    # large-batch instability (thousands+), not needed at batch 64.
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    log_f = open(LOG_PATH, "w")
    run_start = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            opt.zero_grad()
            # Root cause of an earlier divergence bug (diagnosed via diagnose6.py):
            # CPU autocast caches bf16-casted parameter copies for the lifetime of
            # the context. Wrapping the *whole* epoch's loop in one autocast context
            # let that cache go stale across opt.step() calls, corrupting training
            # after epoch 0. Autocast must be entered fresh every step.
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        sched.step()

        val_acc = evaluate(model, val_loader)
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / n_batches
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "lr": sched.get_last_lr()[0],
            "epoch_seconds": epoch_time,
            "elapsed_seconds": time.time() - run_start,
        }
        print(record)
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc, "epoch": epoch}, CHECKPOINT_PATH)

    log_f.close()
    print(f"done. best_val_acc={best_val_acc:.4f}. checkpoint at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
