import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from data import make_datasets

torch.set_num_threads(16)


def run(n_steps: int, lr: float, nesterov: bool, tag: str, log_every: int = 10):
    torch.manual_seed(1234)
    train_ds, _, _ = make_datasets()
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, drop_last=True)

    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5, nesterov=nesterov)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    step = 0
    it = iter(loader)
    while step < n_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        opt.zero_grad()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)
        loss.backward()

        total_norm = sum(p.grad.data.float().norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"[{tag}] step={step:03d} loss={loss.item():.4f} grad_norm={total_norm:.4f} batch_acc={acc:.3f}")
        step += 1


print("=" * 20, "constant LR=0.025, nesterov=True, 300 steps (~2 epochs)", "=" * 20)
run(n_steps=300, lr=0.025, nesterov=True, tag="const_nesterov")

print("=" * 20, "constant LR=0.025, nesterov=False, 300 steps", "=" * 20)
run(n_steps=300, lr=0.025, nesterov=False, tag="const_plain_momentum")
