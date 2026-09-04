import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from data import make_datasets

torch.set_num_threads(16)


def run(use_bf16: bool, n_steps: int = 40, lr: float = 0.025, momentum: float = 0.9, nesterov: bool = True):
    torch.manual_seed(1234)
    train_ds, _, _ = make_datasets()
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, drop_last=True)

    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-5, nesterov=nesterov)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    tag = "bf16" if use_bf16 else "fp32"
    step = 0
    it = iter(loader)
    while step < n_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        opt.zero_grad()
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16) if use_bf16 else torch.enable_grad()
        with ctx:
            logits = model(x)
            loss = loss_fn(logits, y)
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        opt.step()

        finite = torch.isfinite(loss).item()
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        print(f"[{tag}] step={step:03d} loss={loss.item():.4f} finite={finite} grad_norm={total_norm:.4f} batch_acc={acc:.3f}")
        step += 1


print("=" * 20, "FP32 baseline", "=" * 20)
run(use_bf16=False)
print("=" * 20, "BF16 autocast", "=" * 20)
run(use_bf16=True)
