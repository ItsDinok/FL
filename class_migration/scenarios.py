"""
scenarios.py — continual-learning scenario builders.

A Scenario is the only thing the client talks to. It answers four
questions, and every scenario answers them the same way:

    num_blocks                 how many training blocks
    block_name(b)              human label for logging
    partition(cid, n, b)       this client's train/val loaders for block b
    eval_units()               {unit -> loader} for retained-accuracy eval
    unit_mode                  "task" or "class", i.e. horizon.py --mode

Scenarios compose with any DatasetSpec, so adding a dataset does not
require touching scenario code and vice versa. Three are provided:

    SplitScenario      class-incremental: disjoint class groups
    DomainScenario     domain-incremental: fixed labels, shifting inputs
                       (native domains if the dataset has them, else
                       synthetic corruptions)
    DirichletScenario  class migration: proportions resampled per block

    from datasets import get_dataset
    from scenarios import build_scenario
    sc = build_scenario("split", get_dataset("cifar100"), num_blocks=4)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0          # keep 0 in containers (limited /dev/shm)


# ── synthetic corruptions, for datasets without native domains ──────────
class Identity:
    name = "clean"

    def __call__(self, x):
        return x


class Rotate90:
    name = "rotate_90"

    def __call__(self, x):
        return torch.rot90(x, k=1, dims=(-2, -1))


class ChannelPermute:
    name = "channel_permute"

    def __call__(self, x):
        return x[[2, 0, 1], :, :]


class GaussianNoise:
    name = "heavy_noise"

    def __init__(self, std=0.3):
        self.std = std

    def __call__(self, x):
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


CORRUPTIONS = [Identity(), Rotate90(), ChannelPermute(), GaussianNoise()]


def _loader(dataset, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def _split_train_val(dataset, cid, block, val_fraction=0.1,
                     batch_size=BATCH_SIZE):
    n_val = int(len(dataset) * val_fraction)
    tr, va = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED + cid + block * 13))
    return _loader(tr, True, batch_size), _loader(va, False, batch_size)


class Scenario:
    """Interface. Subclasses set num_blocks, unit_mode and num_classes."""
    unit_mode = "task"

    def __init__(self, spec, num_blocks, samples_per_client=4000,
                 eval_samples=1000):
        self.spec = spec
        self.num_blocks = num_blocks
        self.samples_per_client = samples_per_client
        self.eval_samples = eval_samples
        self.num_classes = spec.num_classes

    def block_name(self, b):
        return f"block{b}"

    def unit_name(self, u):
        return f"task{u}"

    def partition(self, cid, num_clients, block):
        raise NotImplementedError

    def eval_units(self):
        raise NotImplementedError

    def describe(self):
        return (f"{type(self).__name__} on {self.spec.name}: "
                f"{self.num_blocks} blocks, "
                f"{self.num_classes} classes, mode={self.unit_mode}")


# ── class-incremental ───────────────────────────────────────────────────
class SplitScenario(Scenario):
    """Disjoint class groups per block; one shared head, no task ID."""

    def _classes(self, block):
        order = np.random.default_rng(SEED).permutation(self.num_classes)
        per = self.num_classes // self.num_blocks
        return order[block * per:(block + 1) * per].tolist()

    def block_name(self, b):
        cs = self._classes(b)
        return f"classes {cs[0]}..{cs[-1]}"

    def partition(self, cid, num_clients, block):
        if not 0 <= cid < num_clients:
            raise ValueError(f"cid {cid} outside [0, {num_clients - 1}]")
        tf = self.spec.train_transform()
        full = self.spec.build(train=True, transform=tf)
        targets = np.asarray(full.targets)
        idx = np.where(np.isin(targets, self._classes(block)))[0]
        perm = np.random.default_rng(SEED + block * 7).permutation(idx)
        size = len(perm) // num_clients
        shard = perm[cid * size:(cid + 1) * size].tolist()
        if len(shard) < BATCH_SIZE:
            raise ValueError(f"client {cid} got {len(shard)} samples; "
                             "check num_clients")
        return _split_train_val(Subset(full, shard), cid, block)

    def eval_units(self):
        tf = self.spec.eval_transform()
        test = self.spec.build(train=False, transform=tf)
        targets = np.asarray(test.targets)
        rng = np.random.default_rng(SEED + 9999)
        out = {}
        for b in range(self.num_blocks):
            idx = []
            per_class = max(1, self.eval_samples // (self.num_classes
                                                     // self.num_blocks))
            for c in self._classes(b):
                pool = np.where(targets == c)[0]
                idx += rng.choice(pool, size=min(per_class, len(pool)),
                                  replace=False).tolist()
            out[b] = _loader(Subset(test, idx), False)
        return out


# ── domain-incremental ──────────────────────────────────────────────────
class DomainScenario(Scenario):
    """Fixed label space, shifting inputs. Uses the dataset's native
    domains when it has them, otherwise synthetic corruptions."""

    def __init__(self, spec, num_blocks=4, **kw):
        native = spec.native_domains
        if native:
            num_blocks = min(num_blocks, len(native))
        super().__init__(spec, num_blocks, **kw)
        self.native = native

    def _corruption(self, block):
        if self.native:
            return None
        c = CORRUPTIONS[block % len(CORRUPTIONS)]
        return None if isinstance(c, Identity) else c

    def _domain(self, block):
        return self.native[block] if self.native else None

    def block_name(self, b):
        return (self.native[b] if self.native
                else CORRUPTIONS[b % len(CORRUPTIONS)].name)

    def partition(self, cid, num_clients, block):
        if not 0 <= cid < num_clients:
            raise ValueError(f"cid {cid} outside [0, {num_clients - 1}]")
        tf = self.spec.train_transform(self._corruption(block))
        full = self.spec.build(train=True, transform=tf,
                               domain=self._domain(block))
        # same image shard across blocks, so only the inputs change
        perm = np.random.default_rng(SEED).permutation(len(full))
        per = len(full) // num_clients
        size = min(self.samples_per_client, per)
        shard = perm[cid * per: cid * per + size].tolist()
        if len(shard) < BATCH_SIZE:
            raise ValueError(f"client {cid} got {len(shard)} samples; "
                             "check num_clients")
        return _split_train_val(Subset(full, shard), cid, block)

    def eval_units(self):
        rng = np.random.default_rng(SEED + 9999)
        out = {}
        for b in range(self.num_blocks):
            tf = self.spec.eval_transform(self._corruption(b))
            test = self.spec.build(train=False, transform=tf,
                                   domain=self._domain(b))
            idx = rng.choice(len(test),
                             size=min(self.eval_samples, len(test)),
                             replace=False).tolist()
            out[b] = _loader(Subset(test, idx), False)
        return out


# ── class migration ─────────────────────────────────────────────────────
class DirichletScenario(Scenario):
    """Each client independently resamples class proportions per block.
    Retained units are classes, so horizon.py runs in --mode class."""
    unit_mode = "class"

    def __init__(self, spec, num_blocks=4, alpha=0.5, eval_per_class=25,
                 **kw):
        super().__init__(spec, num_blocks, **kw)
        self.alpha = alpha
        self.eval_per_class = eval_per_class

    def distribution(self, cid, block):
        rng = np.random.default_rng(SEED + cid * 1000 + block * 7)
        return rng.dirichlet(self.alpha * np.ones(self.num_classes))

    def block_name(self, b):
        return f"block{b} (alpha={self.alpha})"

    def unit_name(self, u):
        names = self.spec.class_names
        return names[u] if u < len(names) else f"class{u}"

    def partition(self, cid, num_clients, block):
        if not 0 <= cid < num_clients:
            raise ValueError(f"cid {cid} outside [0, {num_clients - 1}]")
        props = self.distribution(cid, block)
        rng = np.random.default_rng(SEED + cid * 1000 + block * 7 + 1)
        tf = self.spec.train_transform()
        full = self.spec.build(train=True, transform=tf)
        targets = np.asarray(full.targets)

        n = self.samples_per_client
        alloc = (props * n).astype(int)
        fracs = props * n - alloc
        alloc[np.argsort(fracs)[::-1][:n - alloc.sum()]] += 1
        sel = []
        for c in range(self.num_classes):
            if alloc[c] == 0:
                continue
            pool = np.where(targets == c)[0]
            sel += rng.choice(pool, size=min(alloc[c], len(pool)),
                              replace=False).tolist()
        rng.shuffle(sel)
        return _split_train_val(Subset(full, sel), cid, block)

    def eval_units(self):
        tf = self.spec.eval_transform()
        test = self.spec.build(train=False, transform=tf)
        targets = np.asarray(test.targets)
        rng = np.random.default_rng(SEED + 9999)
        out = {}
        for c in range(self.num_classes):
            pool = np.where(targets == c)[0]
            idx = rng.choice(pool,
                             size=min(self.eval_per_class, len(pool)),
                             replace=False).tolist()
            out[c] = _loader(Subset(test, idx), False)
        return out


_SCENARIOS = {"split": SplitScenario,
              "domain": DomainScenario,
              "dirichlet": DirichletScenario}


def build_scenario(name, spec, **kw):
    if name not in _SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; "
                       f"available: {sorted(_SCENARIOS)}")
    return _SCENARIOS[name](spec, **kw)


def available() -> list:
    return sorted(_SCENARIOS)


if __name__ == "__main__":
    import argparse
    from datasets import get_dataset, available as ds_available

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cifar100", choices=ds_available())
    ap.add_argument("--scenario", default="split", choices=available())
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--clients", type=int, default=3)
    args = ap.parse_args()

    sc = build_scenario(args.scenario, get_dataset(args.dataset),
                        num_blocks=args.blocks)
    print(sc.describe())
    for b in range(sc.num_blocks):
        print(f"  block {b}: {sc.block_name(b)}")
    tr, va = sc.partition(0, args.clients, 0)
    print(f"client 0 block 0: train={len(tr.dataset)} val={len(va.dataset)}")
    ev = sc.eval_units()
    print(f"eval units: {len(ev)} ({sc.unit_mode} mode)")
