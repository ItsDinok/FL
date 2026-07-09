"""
CIFAR-10 data in a ViT-friendly format, with optional federated partitioning.

ViT-B/16 expects 224x224 RGB images normalized with ImageNet statistics,
so CIFAR-10's 32x32 images are resized and normalized accordingly.

Standalone use:
    from cifar10_data import load_cifar10
    train_loader, test_loader = load_cifar10(batch_size=32)

Federated use (drop-in for client.py):
    from cifar10_data import load_partition
    train_loader, val_loader = load_partition(cid=0, num_clients=4)

Then in client.py, replace the body of load_data(cid) with:
    return load_partition(cid, num_clients=NUM_CLIENTS)
CIFAR-10 has 10 classes, so NUM_CLASSES = 10 already matches.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATA_DIR = "./data"          # where CIFAR-10 will be downloaded/cached
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------
# Transforms: resize 32x32 -> 224x224 and normalize with ImageNet stats,
# matching what the pretrained ViT-B/16 expects.
# --------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _make_loader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# --------------------------------------------------------------------------
# Centralized loading (whole dataset)
# --------------------------------------------------------------------------
def load_cifar10(batch_size: int = BATCH_SIZE):
    """Full CIFAR-10 train/test loaders in ViT-ready format."""
    train_set = CIFAR10(DATA_DIR, train=True, download=True,
                        transform=train_transform)
    test_set = CIFAR10(DATA_DIR, train=False, download=True,
                       transform=eval_transform)
    return _make_loader(train_set, True, batch_size), \
           _make_loader(test_set, False, batch_size)


# --------------------------------------------------------------------------
# Federated loading (one partition per client)
# --------------------------------------------------------------------------
def load_partition(cid: int, num_clients: int, val_fraction: float = 0.1,
                   batch_size: int = BATCH_SIZE):
    """
    Split CIFAR-10's train set IID across `num_clients` clients and return
    this client's (train_loader, val_loader).

    A fixed seed makes the split identical across processes, so each client
    process picks up the same disjoint shard for its cid.
    """
    if not 0 <= cid < num_clients:
        raise ValueError(f"cid must be in [0, {num_clients - 1}], got {cid}")

    full_train = CIFAR10(DATA_DIR, train=True, download=True,
                         transform=train_transform)

    # Deterministic shuffle, then slice into equal shards
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    shard_size = len(full_train) // num_clients
    shard = indices[cid * shard_size: (cid + 1) * shard_size]
    client_data = Subset(full_train, shard)

    # Carve out a local validation split
    n_val = int(len(client_data) * val_fraction)
    n_train = len(client_data) - n_val
    train_subset, val_subset = random_split(
        client_data, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED + cid),
    )

    return _make_loader(train_subset, True, batch_size), \
           _make_loader(val_subset, False, batch_size)


def load_server_test(batch_size: int = BATCH_SIZE):
    """CIFAR-10 test set, e.g. for centralized evaluation in server.py."""
    test_set = CIFAR10(DATA_DIR, train=False, download=True,
                       transform=eval_transform)
    return _make_loader(test_set, False, batch_size)


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader = load_partition(cid=0, num_clients=4)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {tuple(images.shape)}")   # (32, 3, 224, 224)
    print(f"Labels: {labels[:8].tolist()}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
