import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from data import make_datasets

torch.set_num_threads(16)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def run(use_clip: bool, epochs: int = 3):
    torch.manual_seed(1234)
    train_ds, val_ds, _ = make_datasets()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)

    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-5, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    tag = "WITH_CLIP" if use_clip else "NO_CLIP"
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for x, y in train_loader:
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                if use_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                running_loss += loss.item()
                n_batches += 1
        sched.step()
        val_acc = evaluate(model, val_loader)
        print(f"[{tag}] epoch={epoch} train_loss={running_loss/n_batches:.4f} val_acc={val_acc:.4f}")


print("=" * 10, "NO grad clipping", "=" * 10)
run(use_clip=False)
print("=" * 10, "WITH grad clipping (max_norm=5.0)", "=" * 10)
run(use_clip=True)
