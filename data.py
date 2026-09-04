import torchvision
from torchvision import transforms

IMG_SIZE = 224

# Deterministic pipeline: used anywhere the functor layer's exact-hash anchoring
# matters (corrections, future lookups). No random ops.
canonical_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Augmented pipeline: used only for base-model training (not for anything that
# needs to be re-identified later by exact hash).
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.35, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_datasets(root="data/imagenette2"):
    train_ds = torchvision.datasets.ImageFolder(root=f"{root}/train", transform=train_transform)
    val_ds = torchvision.datasets.ImageFolder(root=f"{root}/val", transform=canonical_transform)
    # Also expose the train set under the canonical (deterministic) transform,
    # for anything that needs to reproduce exact anchor tensors from train images.
    train_ds_canonical = torchvision.datasets.ImageFolder(root=f"{root}/train", transform=canonical_transform)
    return train_ds, val_ds, train_ds_canonical
