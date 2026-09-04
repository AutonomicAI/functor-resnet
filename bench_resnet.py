import time
import torch
import torchvision

torch.set_num_threads(16)

model = torchvision.models.resnet18(weights=None, num_classes=10)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

for img_size in (128, 160, 224):
    for batch_size in (32, 64):
        x = torch.randn(batch_size, 3, img_size, img_size)
        y = torch.randint(0, 10, (batch_size,))

        # warmup
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for _ in range(2):
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

        n_iters = 5
        t0 = time.time()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for _ in range(n_iters):
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
        elapsed = time.time() - t0
        imgs_per_sec = (n_iters * batch_size) / elapsed
        print(f"img_size={img_size} batch={batch_size}: {elapsed/n_iters*1000:.1f} ms/iter, {imgs_per_sec:.1f} img/s (train, fwd+bwd)")

    # single-image forward-only (inference latency, relevant to functor path)
    x1 = torch.randn(1, 3, img_size, img_size)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(3):
            model(x1)
        t0 = time.time()
        for _ in range(20):
            model(x1)
        elapsed = time.time() - t0
    print(f"img_size={img_size} batch=1 INFERENCE ONLY: {elapsed/20*1000:.2f} ms/image")
    model.train()
