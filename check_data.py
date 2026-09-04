import torch
from data import make_datasets

train_ds, val_ds, _ = make_datasets()

print(f"checking {len(train_ds)} training images for anomalies...")
bad = []
for i in range(len(train_ds)):
    try:
        x, y = train_ds[i]
    except Exception as e:
        bad.append((i, train_ds.samples[i][0], f"exception: {e}"))
        continue
    if not torch.isfinite(x).all():
        bad.append((i, train_ds.samples[i][0], "non-finite values"))
        continue
    mn, mx = x.min().item(), x.max().item()
    if mx > 20 or mn < -20:
        bad.append((i, train_ds.samples[i][0], f"extreme range [{mn:.2f}, {mx:.2f}]"))
    if x.shape[0] != 3:
        bad.append((i, train_ds.samples[i][0], f"bad channel count {x.shape}"))
    if i % 2000 == 0:
        print(f"  checked {i}/{len(train_ds)}", flush=True)

print(f"\nfound {len(bad)} anomalous images:")
for i, path, reason in bad:
    print(f"  idx={i} path={path} reason={reason}")

if not bad:
    print("no anomalies found in training set under train_transform sampling")
