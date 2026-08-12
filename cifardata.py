"""
cifardata.py — CIFAR-100 with Dirichlet class-distribution shift
(class migration), instrumented-experiment variant.

Each client's class proportions are resampled independently every block
from Dir(alpha * 1_K). Classes migrate in and out of the fleet's effective
training distribution; forgetting is absence-driven. This is the
class-migration condition of the intervention-signal benchmark
(horizon.py --mode class scores it).

"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

DATA_DIR = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

NUM_CLASSES = 100
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
IMAGENET_STD = [0.229, 0.224, 0.225]

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


def _make_loader(dataset, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def get_block_distribution(cid: int, alpha: float, block: int) -> np.ndarray:
    """Deterministic, independent Dirichlet sample per (client, block)."""
    rng = np.random.default_rng(SEED + cid * 1000 + block * 7)
    return rng.dirichlet(alpha * np.ones(NUM_CLASSES))


def _dirichlet_indices(targets, proportions, n, rng):
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
    proportions = get_block_distribution(cid, alpha, block)
    rng = np.random.default_rng(SEED + cid * 1000 + block * 7 + 1)
    full = CIFAR100(DATA_DIR, train=True, download=True,
                    transform=train_transform)
    targets = np.array(full.targets)
    indices = _dirichlet_indices(targets, proportions,
                                 SAMPLES_PER_CLIENT_PER_BLOCK, rng)
    dataset = Subset(full, indices)
    n_val = int(len(dataset) * val_fraction)
    tr, va = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED + cid + block * 13))
    return _make_loader(tr, True, batch_size), _make_loader(va, False, batch_size)


def load_per_class_eval_loaders(samples_per_class: int = 50,
                                batch_size: int = BATCH_SIZE) -> dict:
    """One loader per class from the test set — per-class retained
    accuracy defines the per-class t_forget in horizon.py --mode class."""
    test = CIFAR100(DATA_DIR, train=False, download=True,
                    transform=eval_transform)
    targets = np.array(test.targets)
    rng = np.random.default_rng(SEED + 9999)
    loaders = {}
    for c in range(NUM_CLASSES):
        idx = np.where(targets == c)[0]
        chosen = rng.choice(idx, size=min(samples_per_class, len(idx)),
                            replace=False).tolist()
        loaders[c] = _make_loader(Subset(test, chosen), False, batch_size)
    return loaders


if __name__ == "__main__":
    print("Class-migration CIFAR-100, alpha=0.5 dominant classes:")
    for cid in range(3):
        row = []
        for b in range(4):
            p = get_block_distribution(cid, 0.5, b)
            row.append(f"b{b}:{CLASS_NAMES[int(p.argmax())]}({p.max():.2f})")
        print(f"  client {cid}: " + "  ".join(row))
    tl, vl = load_dirichlet_partition(0, 3, alpha=0.5, block=0)
    print(f"partition sizes: train={len(tl.dataset)} val={len(vl.dataset)}")
