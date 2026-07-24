"""
CIFAR-100 with Dirichlet class-distribution shift for federated continual
learning research. Drop-in replacement for the CIFAR-10 version —
NUM_CLASSES, CLASS_NAMES, and all public functions have the same signatures.

Why CIFAR-100 over CIFAR-10:
  A pretrained ViT-B/16 effectively already solves CIFAR-10 (97%+ zero-shot),
  leaving almost no headroom to study forgetting. CIFAR-100 is harder: finer
  granularity (100 classes vs 10), smaller per-class training set (500 vs
  5000 samples), and the pretrained model starts meaningfully lower, giving
  the continual learning dynamics real room to operate.

Alpha guidance:
  Dir(alpha * 1_100): with K=100 classes, alpha=0.1 allocates most weight to
  ~1-2 classes per client (too extreme — many classes get 0 samples).
  alpha=0.5 gives ~5-10 effective classes per block, which is heterogeneous
  but not pathological. Default is 0.5; sweep 0.1-2.0 to vary severity.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

DATA_DIR   = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0       # keep 0 in containers (limited /dev/shm)
SEED       = 42

NUM_CLASSES = 100

# CIFAR-100 fine-grained class names in label order (0-99)
CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
]
assert len(CLASS_NAMES) == NUM_CLASSES

SAMPLES_PER_CLIENT_PER_BLOCK = 5000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=14),   # extra augmentation helps on 100 classes
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _make_loader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def get_block_distribution(cid: int, alpha: float, block: int) -> np.ndarray:
    """Deterministic Dirichlet sample for (client, block).
    Each client draws independently — no two clients share a distribution
    within the same block.
    """
    rng = np.random.default_rng(SEED + cid * 1000 + block * 7)
    return rng.dirichlet(alpha * np.ones(NUM_CLASSES))


def _dirichlet_indices(targets: np.ndarray, proportions: np.ndarray,
                       n: int, rng: np.random.Generator) -> list:
    """Sample n indices from targets according to class proportions.
    Allocation uses the largest-remainder method to sum exactly to n."""
    class_indices = {c: np.where(targets == c)[0] for c in range(NUM_CLASSES)}
    alloc = (proportions * n).astype(int)
    residual = n - alloc.sum()
    fracs = proportions * n - alloc
    alloc[np.argsort(fracs)[::-1][:residual]] += 1

    selected = []
    for c in range(NUM_CLASSES):
        want = int(alloc[c])
        if want == 0:
            continue
        pool = class_indices[c]
        take = min(want, len(pool))
        selected.extend(rng.choice(pool, size=take, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def load_dirichlet_partition(cid: int, num_clients: int,
                             alpha: float, block: int,
                             val_fraction: float = 0.1,
                             batch_size: int = BATCH_SIZE):
    """Train/val loaders for client cid in this block, drawn from CIFAR-100
    according to an independently-sampled Dirichlet class distribution."""
    proportions = get_block_distribution(cid, alpha, block)
    rng = np.random.default_rng(SEED + cid * 1000 + block * 7 + 1)

    full_train = CIFAR100(DATA_DIR, train=True, download=True,
                          transform=train_transform)
    targets = np.array(full_train.targets)

    indices = _dirichlet_indices(targets, proportions,
                                 SAMPLES_PER_CLIENT_PER_BLOCK, rng)
    dataset = Subset(full_train, indices)

    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val
    train_sub, val_sub = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED + cid + block * 13))

    return _make_loader(train_sub, True, batch_size), \
           _make_loader(val_sub, False, batch_size)


def load_per_class_eval_loaders(samples_per_class: int = 80,
                                batch_size: int = BATCH_SIZE) -> dict:
    """One DataLoader per CIFAR-100 class from the TEST set.
    CIFAR-100 has exactly 100 test images per class (10,000 total).
    samples_per_class=80 uses 80% of each class's test images.
    """
    test_set = CIFAR100(DATA_DIR, train=False, download=True,
                        transform=eval_transform)
    targets = np.array(test_set.targets)
    loaders = {}
    rng = np.random.default_rng(SEED + 9999)
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        chosen = rng.choice(idx, size=min(samples_per_class, len(idx)),
                            replace=False).tolist()
        loaders[c] = _make_loader(Subset(test_set, chosen), False, batch_size)
    return loaders


if __name__ == "__main__":
    print(f"CIFAR-100 | {NUM_CLASSES} classes | {len(CLASS_NAMES)} names\n")

    print("Block distributions (client 0, alpha=0.5):")
    for b in range(4):
        p = get_block_distribution(0, alpha=0.5, block=b)
        top5 = sorted(range(NUM_CLASSES), key=lambda c: -p[c])[:5]
        print(f"  block {b}: "
              + ", ".join(f"{CLASS_NAMES[c]}({p[c]:.2f})" for c in top5))

    print("\nVerifying partition (client 0, block 0, alpha=0.5):")
    tl, vl = load_dirichlet_partition(0, 3, alpha=0.5, block=0)
    print(f"  train={len(tl.dataset)} val={len(vl.dataset)} "
          f"batches={len(tl)}")
    images, labels = next(iter(tl))
    print(f"  batch shape: {tuple(images.shape)} "
          f"unique classes in batch: {labels.unique().numel()}")

    print(f"\nPer-class eval loaders: {len(load_per_class_eval_loaders())} loaders")
