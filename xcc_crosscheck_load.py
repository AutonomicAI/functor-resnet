"""
Standalone synthetic load generator for the XCC/IPMI vs RAPL power cross-check.
Deliberately separate from the frozen benchmark harnesses so this validation
step cannot perturb the already-recorded results. Runs a ResNet-18 forward+
backward loop (same op mix as the real harnesses) for a fixed wall-clock
duration and reports RAPL energy for that exact window.
"""
import sys
import time

import torch
import torch.nn as nn
import torchvision

from energy import measure_energy

DURATION_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

torch.set_num_threads(16)
model = torchvision.models.resnet18(weights=None, num_classes=10)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
x = torch.randn(64, 3, 224, 224)
y = torch.randint(0, 10, (64,))

with measure_energy() as reading:
    t_end = time.time() + DURATION_SECONDS
    n_iters = 0
    while time.time() < t_end:
        opt.zero_grad()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        n_iters += 1

r = reading.result
print(f"n_iters={n_iters} seconds={r.seconds:.2f} rapl_total_joules={r.total_joules:.2f} rapl_watts={r.watts:.2f}")
