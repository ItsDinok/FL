"""
cifardata_split.py — Split CIFAR-100, the primary (interference-regime)
benchmark for the intervention-signal comparison.

Protocol:
    100 classes are split into NUM_TASKS disjoint tasks (default 4 x 25,
    seeded random assignment). Task t is trained during block t; once the
    fleet moves to task t+1, task t's classes never appear in training
    again. Class-incremental: one shared 100-way head, no task identity at
    test time. Within each task, the task's data is split IID across
    clients (deterministic shard per client).

Evaluation:
    load_task_eval_loaders() returns one loader per task from the test set.
    Per-task retained accuracy defines t_forget in
        python horizon.py --logs logs_split/ --mode task

Named *_split so it coexists with the class-migration condition
(cifardata.py / client.py / server.py) in the same directory.
"""

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
NUM_TASKS = 4
CLASSES_PER_TASK = NUM_CLASSES // NUM_TASKS   # 25

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


def task_classes(task_id: int) -> list:
    """Class IDs belonging to a task (seeded permutation, contiguous cut)."""
    order = np.random.default_rng(SEED).permutation(NUM_CLASSES)
    return order[task_id * CLASSES_PER_TASK:
                 (task_id + 1) * CLASSES_PER_TASK].tolist()


def _make_loader(dataset, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_task_partition(cid: int, num_clients: int, task_id: int,
                        val_fraction: float = 0.1,
                        batch_size: int = BATCH_SIZE):
    """This client's train/val loaders for one task: the task's classes
    filtered from CIFAR-100 train, split IID across clients."""
    classes = task_classes(task_id)
    full = CIFAR100(DATA_DIR, train=True, download=True,
                    transform=train_transform)
    targets = np.array(full.targets)
    task_idx = np.where(np.isin(targets, classes))[0]

    perm = np.random.default_rng(SEED + task_id * 7).permutation(task_idx)
    shard_size = len(perm) // num_clients
    shard = perm[cid * shard_size:(cid + 1) * shard_size].tolist()
    dataset = Subset(full, shard)

    n_val = int(len(dataset) * val_fraction)
    tr, va = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED + cid + task_id * 13))
    return _make_loader(tr, True, batch_size), _make_loader(va, False, batch_size)


def load_task_eval_loaders(samples_per_class: int = 40,
                           batch_size: int = BATCH_SIZE) -> dict:
    """One eval loader PER TASK from the test set (25 classes x
    samples_per_class images). Per-task retained accuracy -> t_forget."""
    test = CIFAR100(DATA_DIR, train=False, download=True,
                    transform=eval_transform)
    targets = np.array(test.targets)
    rng = np.random.default_rng(SEED + 9999)
    loaders = {}
    for t in range(NUM_TASKS):
        idx = []
        for c in task_classes(t):
            pool = np.where(targets == c)[0]
            idx.extend(rng.choice(pool, size=min(samples_per_class, len(pool)),
                                  replace=False).tolist())
        loaders[t] = _make_loader(Subset(test, idx), False, batch_size)
    return loaders


if __name__ == "__main__":
    print(f"Split CIFAR-100: {NUM_TASKS} tasks x {CLASSES_PER_TASK} classes")
    all_c = []
    for t in range(NUM_TASKS):
        cs = task_classes(t)
        all_c += cs
        print(f"  task {t}: {cs[:6]}... ({len(cs)} classes)")
    assert len(all_c) == len(set(all_c)) == NUM_CLASSES
    print("task splits disjoint and complete")
    tl, vl = load_task_partition(0, 3, task_id=0)
    print(f"client 0 task 0: train={len(tl.dataset)} val={len(vl.dataset)}")
    ev = load_task_eval_loaders()
    print(f"eval images per task: {[len(v.dataset) for v in ev.values()]}")
