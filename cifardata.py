"""
CIFAR-10 with HARSH simulated domain shift, for domain-incremental
federated continual learning with a ViT.

Domains (deliberately disjoint so forgetting can actually occur):
    0: clean
    1: rotate_90        (fixed 90-degree rotation)
    2: channel_permute  (RGB -> BRG, breaks pretrained color statistics)
    3: heavy_noise      (gaussian noise, std 0.3)

Same 10 classes in every domain; only the input distribution shifts.
Sanity check: zero-shot (round 1) accuracy on domains 1-3 should be LOW.
If it isn't, the shift isn't harsh enough to study forgetting.

Same API as before — client.py and server.py need no changes.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATA_DIR = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0            # keep 0 in containers (limited /dev/shm)
SEED = 42

NUM_DOMAINS = 4
DOMAIN_NAMES = ["clean", "rotate_90", "channel_permute", "heavy_noise"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Domain corruptions (applied on [0,1] tensors, before normalization)
# --------------------------------------------------------------------------
class Rotate90:
    """Fixed 90-degree rotation. Deterministic and severe: ViTs are not
    rotation invariant, and positional embeddings make this a real shift."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))


class ChannelPermute:
    """RGB -> BRG. Destroys the color statistics the pretrained backbone
    expects while keeping all spatial structure intact."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[[2, 0, 1], :, :]


class GaussianNoise:
    def __init__(self, std: float = 0.3):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


def _domain_corruption(domain_id: int):
    if domain_id == 0:
        return None
    if domain_id == 1:
        return Rotate90()
    if domain_id == 2:
        return ChannelPermute()
    if domain_id == 3:
        return GaussianNoise(std=0.3)
    raise ValueError(f"Unknown domain_id {domain_id}")


def make_transform(domain_id: int, train: bool) -> transforms.Compose:
    """ViT-ready pipeline: resize -> tensor -> domain corruption -> normalize."""
    steps = [transforms.Resize((224, 224))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    corruption = _domain_corruption(domain_id)
    if corruption is not None:
        steps.append(corruption)
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def _make_loader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


# --------------------------------------------------------------------------
# Federated, domain-incremental partitioning (unchanged API)
# --------------------------------------------------------------------------
def _client_indices(n_total: int, cid: int, num_clients: int):
    """Deterministic IID shard of the train set for this client."""
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    shard_size = n_total // num_clients
    return indices[cid * shard_size: (cid + 1) * shard_size]


def load_domain_partition(cid: int, num_clients: int, domain_id: int,
                          val_fraction: float = 0.1,
                          batch_size: int = BATCH_SIZE):
    """This client's train/val loaders for one domain."""
    if not 0 <= cid < num_clients:
        raise ValueError(f"cid must be in [0, {num_clients - 1}], got {cid}")

    train_tf = make_transform(domain_id, train=True)
    full_train = CIFAR10(DATA_DIR, train=True, download=True,
                         transform=train_tf)

    shard = _client_indices(len(full_train), cid, num_clients)
    client_data = Subset(full_train, shard)

    n_val = int(len(client_data) * val_fraction)
    n_train = len(client_data) - n_val
    train_subset, val_subset = random_split(
        client_data, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED + cid),
    )
    return _make_loader(train_subset, True, batch_size), \
           _make_loader(val_subset, False, batch_size)


def load_all_domain_eval_loaders(cid: int, num_clients: int,
                                 samples_per_domain: int = 500,
                                 batch_size: int = BATCH_SIZE):
    """One small eval loader PER domain (from the test set), for measuring
    forgetting on earlier domains as training moves to later ones."""
    loaders = {}
    generator = torch.Generator().manual_seed(SEED + 1000 + cid)
    for d in range(NUM_DOMAINS):
        test_set = CIFAR10(DATA_DIR, train=False, download=True,
                           transform=make_transform(d, train=False))
        idx = torch.randperm(len(test_set), generator=generator)
        idx = idx[:samples_per_domain].tolist()
        loaders[d] = _make_loader(Subset(test_set, idx), False, batch_size)
    return loaders


if __name__ == "__main__":
    train_loader, _ = load_domain_partition(0, 3, domain_id=1)
    images, labels = next(iter(train_loader))
    print(f"Domain 1 ({DOMAIN_NAMES[1]}) batch: {tuple(images.shape)}")
    evals = load_all_domain_eval_loaders(0, 3)
    print(f"Eval loaders for domains: {list(evals.keys())}")
