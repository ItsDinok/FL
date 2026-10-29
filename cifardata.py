"""
cifardata_dil.py — domain-incremental CIFAR-100 (interference regime,
input-space shift) for the intervention-signal benchmark.

Protocol:
    Label space is FIXED (all 100 classes throughout). The input
    distribution changes between blocks via deterministic corruptions:

        0: clean            (ImageNet normalisation only)
        1: rotate_90        fixed 90-degree rotation
        2: channel_permute  RGB -> BRG
        3: heavy_noise      Gaussian noise, sigma = 0.3

    Domain d is trained during block d, by ALL clients simultaneously
    (correlated drift — the staggered variant suppresses forgetting via
    implicit replay and is unsuitable for signal scoring). Within a block
    the data is split IID across clients.

    This is the only instrumented condition in which the *inputs* shift
    while the label distribution is constant, and the only one in which
    forgetting accumulates over several rounds rather than completing
    within one.

Evaluation:
    load_domain_eval_loaders() returns one loader per domain from the test
    set. Per-domain retained accuracy defines t_forget in
        python horizon.py --logs logs_dil/ --mode task
    (the "task" mode is used unchanged: domain index plays the role of
    task index, and the block schedule is identical.)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

DATA_DIR = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0            # keep 0 in containers (limited /dev/shm)
SEED = 42

NUM_CLASSES = 100
NUM_DOMAINS = 4
DOMAIN_NAMES = ["clean", "rotate_90", "channel_permute", "heavy_noise"]

SAMPLES_PER_CLIENT_PER_BLOCK = 4000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── domain corruptions (applied to [0,1] tensors, before normalisation) ──
class Rotate90:
    """Fixed 90-degree rotation. ViTs are not rotation invariant and the
    positional embeddings make this a genuine representational shift."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))


class ChannelPermute:
    """RGB -> BRG. Destroys pretrained colour statistics, preserves
    spatial structure."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[[2, 0, 1], :, :]


class GaussianNoise:
    def __init__(self, std: float = 0.3):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


def _corruption(domain_id: int):
    return [None, Rotate90(), ChannelPermute(), GaussianNoise(0.3)][domain_id]


def make_transform(domain_id: int, train: bool) -> transforms.Compose:
    """resize -> (flip) -> tensor -> corruption -> ImageNet normalise."""
    steps = [transforms.Resize((224, 224))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    c = _corruption(domain_id)
    if c is not None:
        steps.append(c)
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def _make_loader(dataset, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_domain_partition(cid: int, num_clients: int, domain_id: int,
                          val_fraction: float = 0.1,
                          batch_size: int = BATCH_SIZE):
    """This client's train/val loaders for one domain. The same underlying
    image shard is reused across domains with a different corruption
    applied, so only the input distribution changes."""
    if not 0 <= cid < num_clients:
        raise ValueError(
            f"cid {cid} is outside [0, {num_clients - 1}]. NUM_CLIENTS in "
            f"the client module must match the number of client processes "
            f"the launcher starts; a mismatch silently produces tiny or "
            f"empty partitions.")

    full = CIFAR100(DATA_DIR, train=True, download=True,
                    transform=make_transform(domain_id, train=True))

    # deterministic IID shard, identical across domains for this client.
    # Shards are contiguous slices of one permutation, so they are disjoint
    # by construction and every client gets the same number of samples.
    perm = np.random.default_rng(SEED).permutation(len(full))
    block = len(full) // num_clients
    shard_size = min(SAMPLES_PER_CLIENT_PER_BLOCK, block)
    shard = perm[cid * block: cid * block + shard_size].tolist()
    if len(shard) < batch_size:
        raise ValueError(
            f"client {cid} received only {len(shard)} samples for domain "
            f"{domain_id}; expected {shard_size}. Check num_clients.")
    dataset = Subset(full, shard)

    n_val = int(len(dataset) * val_fraction)
    tr, va = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED + cid + domain_id * 13))
    return _make_loader(tr, True, batch_size), _make_loader(va, False, batch_size)


def load_domain_eval_loaders(samples: int = 1000,
                             batch_size: int = BATCH_SIZE) -> dict:
    """One eval loader PER DOMAIN from the test set (same images, four
    corruptions). Per-domain retained accuracy -> t_forget."""
    rng = np.random.default_rng(SEED + 9999)
    base = CIFAR100(DATA_DIR, train=False, download=True)
    idx = rng.choice(len(base), size=min(samples, len(base)),
                     replace=False).tolist()
    loaders = {}
    for d in range(NUM_DOMAINS):
        test = CIFAR100(DATA_DIR, train=False, download=True,
                        transform=make_transform(d, train=False))
        loaders[d] = _make_loader(Subset(test, idx), False, batch_size)
    return loaders


def ensure_data(data_dir: str = None) -> str:
    """
        Download and verify CIFAR-100 if it is not already present
        Running this before a federated job means clients won't burn through memory reacquiring data
    """
    import shutil

    target = data_dir or DATA_DIR
    os.makedirs(target, exist_ok=True)
    print(f"Data directory: {os.path.abspath(target)}")
    free_gb = shutil.disk_usage(target).free / 1e9
    if free_gb < 1.0:
        print(f"Warning: only {free_gb:.1f} GB free; CIFAR-100 needs 0.2GB")

        for split, train in (("train", True), ("test", False)):
            ds = CIFAR100(target, train=train, download=True)
            print(f"  {split}: {len(ds)} images, {len(ds.classes)} classes")
            print("CIFAR-100 ready.")
            return target


if __name__ == "__main__":
    import argparse    

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--download", action="store_true",
        help="download/verify CIFAR100 and exit")
    _ap.add_argument("--data-dir", default=None,
        help=f"override DATA_DIR (default{DATA_DIR})")
    _args = _ap.parse_args()

    if _args.data_dir:
        DATA_DIR = _args.data_dir
    ensure_data(DATA_DIR)
    if _args.download:
        raise SystemExit(0)

    print(f"Domain-incremental CIFAR-100: {NUM_DOMAINS} domains")
    for d, name in enumerate(DOMAIN_NAMES):
        print(f"  domain {d}: {name}")
    tl, vl = load_domain_partition(0, 3, domain_id=1)
    x, y = next(iter(tl))
    print(f"client 0 domain 1: train={len(tl.dataset)} val={len(vl.dataset)} "
          f"batch={tuple(x.shape)}")
    ev = load_domain_eval_loaders()
    print(f"eval images per domain: {[len(v.dataset) for v in ev.values()]}")
