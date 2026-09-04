import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from data import make_datasets

torch.set_num_threads(16)
torch.manual_seed(1234)

train_ds, val_ds, _ = make_datasets()
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8)

model = torchvision.models.resnet18(weights=None, num_classes=10)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-5, nesterov=True)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


def train_step(x, y, step):
    opt.zero_grad()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(x)
        loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    if step % 10 == 0:
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        print(f"step={step:03d} loss={loss.item():.4f} batch_acc={acc:.3f}")


def run_eval(tag):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for i, (x, y) in enumerate(val_loader):
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if i >= 5:  # partial pass is enough to test the mode-switch effect
                break
    model.train()
    print(f"--- eval after {tag}: partial val_acc={correct/total:.4f} ---")


print("=== phase 1: train 148 steps (1 epoch equivalent) ===")
step = 0
it = iter(train_loader)
for step in range(148):
    try:
        x, y = next(it)
    except StopIteration:
        it = iter(train_loader)
        x, y = next(it)
    train_step(x, y, step)

print("=== interposing eval() / train() cycle, like evaluate() does ===")
run_eval("epoch0")

print("=== phase 2: train 150 more steps, watching for blowup ===")
it = iter(train_loader)
for step in range(148, 298):
    try:
        x, y = next(it)
    except StopIteration:
        it = iter(train_loader)
        x, y = next(it)
    train_step(x, y, step)
