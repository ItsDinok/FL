"""
datasets.py — dataset registry.

Every dataset is described by a DatasetSpec, so scenarios and clients never
name a dataset directly. Adding a dataset means adding one entry here;
nothing else in the codebase changes.

    from datasets import get_dataset
    spec = get_dataset("cifar100")
    train = spec.build(train=True, transform=spec.train_transform())

Datasets that ship their own domains (PACS, DomainNet, OfficeHome) expose
them through `native_domains`; the domain-incremental scenario uses those
in preference to synthetic corruptions, which is what makes the
"real versus synthetic shift" ablation a one-line config change.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

DATA_ROOT = os.environ.get("FL_DATA_ROOT", "./data")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    builder: Callable                 # (root, train, transform) -> Dataset
    class_names: list = field(default_factory=list)
    mean: list = field(default_factory=lambda: IMAGENET_MEAN)
    std: list = field(default_factory=lambda: IMAGENET_STD)
    image_size: int = 224
    native_domains: Optional[list] = None   # set for PACS-style datasets
    needs_manual_download: bool = False
    notes: str = ""

    # ── transforms ──────────────────────────────────────────────────
    def _base(self, train: bool):
        steps = [transforms.Resize((self.image_size, self.image_size))]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.ToTensor())
        return steps

    def train_transform(self, corruption=None):
        steps = self._base(True)
        if corruption is not None:
            steps.append(corruption)
        steps.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(steps)

    def eval_transform(self, corruption=None):
        steps = self._base(False)
        if corruption is not None:
            steps.append(corruption)
        steps.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(steps)

    # ── access ──────────────────────────────────────────────────────
    def build(self, train: bool, transform=None, domain: str = None):
        """Returns a dataset exposing `.targets` as a list of int labels."""
        return self.builder(DATA_ROOT, train, transform, domain)

    def targets(self, train: bool = True) -> np.ndarray:
        """Label array without loading images — used by partitioners."""
        ds = self.build(train, transform=None)
        return np.asarray(ds.targets)

    def ensure(self):
        """Download/verify. Safe to call repeatedly; call once before
        launching clients so they do not race on the same archive."""
        if self.needs_manual_download:
            root = os.path.join(DATA_ROOT, self.name)
            if not os.path.isdir(root):
                raise SystemExit(
                    f"{self.name} must be fetched manually into {root}.\n"
                    f"{self.notes}")
            print(f"{self.name}: found at {os.path.abspath(root)}")
            return
        for train in (True, False):
            ds = self.build(train, transform=None)
            print(f"  {self.name} {'train' if train else 'test':<5}: "
                  f"{len(ds)} images")


# ── builders ────────────────────────────────────────────────────────────
def _cifar(cls):
    def build(root, train, transform, domain=None):
        return cls(root, train=train, download=True, transform=transform)
    return build


def _imagefolder(subdir, split_dirs=("train", "val")):
    """Generic ImageFolder dataset laid out as <root>/<subdir>/<split>/<class>/*"""
    def build(root, train, transform, domain=None):
        parts = [root, subdir]
        if domain is not None:
            parts.append(domain)
        parts.append(split_dirs[0] if train else split_dirs[1])
        path = os.path.join(*parts)
        ds = ImageFolder(path, transform=transform)
        ds.targets = [s[1] for s in ds.samples]
        return ds
    return build


def _pacs():
    """PACS: <root>/pacs/<domain>/<class>/*.jpg — no train/test split on
    disk, so the partitioner carves one deterministically."""
    def build(root, train, transform, domain=None):
        domain = domain or "photo"
        path = os.path.join(root, "pacs", domain)
        ds = ImageFolder(path, transform=transform)
        ds.targets = [s[1] for s in ds.samples]
        return ds
    return build



def _eurosat():
    """EuroSAT ships as a flat ImageFolder with no train/test split, so one
    is carved deterministically (80/20, seeded) to keep evaluation honest
    across runs."""
    def build(root, train, transform, domain=None):
        from torchvision.datasets import EuroSAT
        from torch.utils.data import Subset

        full = EuroSAT(root, download=True, transform=transform)
        targets = np.asarray([s[1] for s in full.samples])
        rng = np.random.default_rng(12345)
        keep = []
        for c in np.unique(targets):
            idx = np.where(targets == c)[0]
            idx = rng.permutation(idx)
            cut = int(0.8 * len(idx))
            keep += (idx[:cut] if train else idx[cut:]).tolist()
        keep.sort()
        ds = Subset(full, keep)
        ds.targets = [int(targets[i]) for i in keep]
        ds.classes = full.classes
        return ds
    return build


def _cifar100c():
    """CIFAR-100-C: the standard corruption benchmark (Hendrycks &
    Dietterich). Corruptions arrive as .npy arrays of 50,000 images, five
    severities of 10,000 each, sharing CIFAR-100's test labels.

    `domain` selects the corruption; severity is fixed by CIFAR100C_SEVERITY.
    Training data is the clean CIFAR-100 train split, since the benchmark
    only provides corrupted *test* images — so this spec is for evaluating
    domain robustness, not for training through corruptions. Use the
    synthetic corruptions in scenarios.py for the latter.
    """
    def build(root, train, transform, domain=None):
        from torch.utils.data import Dataset
        from PIL import Image

        if train or domain in (None, "clean"):
            return CIFAR100(root, train=train, download=True,
                            transform=transform)

        base = os.path.join(root, "CIFAR-100-C")
        arr = np.load(os.path.join(base, f"{domain}.npy"))
        labels = np.load(os.path.join(base, "labels.npy"))
        s = max(1, min(5, CIFAR100C_SEVERITY))
        lo, hi = (s - 1) * 10000, s * 10000

        class _C(Dataset):
            def __init__(self):
                self.data = arr[lo:hi]
                self.targets = labels[lo:hi].astype(int).tolist()
                self.transform = transform

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, i):
                img = Image.fromarray(self.data[i])
                if self.transform is not None:
                    img = self.transform(img)
                return img, self.targets[i]

        return _C()
    return build


CIFAR100C_SEVERITY = 3
CIFAR100C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]



def _medmnist(flag, n_classes):
    """MedMNIST v2 — medical imaging, far outside ImageNet's distribution,
    which is precisely why it is useful for the pretraining ablation.
    Installs with `pip install medmnist`; data comes from Zenodo on first
    use. Images are 28x28 greyscale or RGB and are upscaled by the usual
    transform pipeline.
    """
    def build(root, train, transform, domain=None):
        import medmnist
        from medmnist import INFO
        from torch.utils.data import Dataset

        info = INFO[flag]
        cls = getattr(medmnist, info["python_class"])
        split = "train" if train else "test"
        inner = cls(split=split, root=root, download=False,
                    transform=transform, as_rgb=True)

        class _Flat(Dataset):
            """MedMNIST yields labels of shape (1,); cross-entropy needs a
            scalar per sample, so squeeze on the way out. `targets` is
            exposed as a flat int list for the partitioners."""

            def __init__(self):
                self.inner = inner
                self.targets = [int(x) for x in inner.labels.reshape(-1)]
                self.classes = [info["label"][str(i)]
                                for i in range(n_classes)]

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, i):
                img, _ = self.inner[i]
                return img, self.targets[i]

        return _Flat()
    return build


# ── registry ────────────────────────────────────────────────────────────
_REGISTRY = {}


def register(spec: DatasetSpec):
    _REGISTRY[spec.name] = spec
    return spec


register(DatasetSpec(
    name="cifar100", num_classes=100, builder=_cifar(CIFAR100),
    notes="Primary benchmark. Adequate headroom under a pretrained ViT."))

register(DatasetSpec(
    name="cifar10", num_classes=10, builder=_cifar(CIFAR10),
    notes="Retained for the historical DIL runs; pretrained ceiling is "
          "too high for new work."))

register(DatasetSpec(
    name="tinyimagenet", num_classes=200,
    builder=_imagefolder("tiny-imagenet-200"),
    needs_manual_download=True,
    notes="wget http://cs231n.stanford.edu/tiny-imagenet-200.zip and "
          "unzip into the data root. Note the val/ split needs "
          "reorganising into class subdirectories."))

register(DatasetSpec(
    name="pacs", num_classes=7, builder=_pacs(),
    native_domains=["photo", "art_painting", "cartoon", "sketch"],
    needs_manual_download=True,
    notes="Four real domains, 7 classes, ~10k images. The cheap answer to "
          "the objection that synthetic corruptions are not domain shift."))

register(DatasetSpec(
    name="eurosat", num_classes=10, builder=_eurosat(),
    mean=[0.344, 0.380, 0.408], std=[0.203, 0.137, 0.115],
    notes="Satellite imagery, auto-downloaded. ImageNet pretraining "
          "transfers poorly here, which is the point: it tests whether the "
          "findings are properties of FCL or of a backbone that already "
          "knows the classes. 27k images, deterministic 80/20 split."))


register(DatasetSpec(
    name="cifar100c", num_classes=100, builder=_cifar100c(),
    native_domains=["clean"] + CIFAR100C_CORRUPTIONS,
    needs_manual_download=True,
    notes="Standard corruption benchmark (Hendrycks & Dietterich 2019). "
          "Download CIFAR-100-C.tar from https://zenodo.org/record/3555552 "
          "and extract so that <root>/CIFAR-100-C/*.npy exists. Training "
          "uses clean CIFAR-100; corruptions apply to evaluation only, so "
          "this measures robustness rather than training through shift."))


register(DatasetSpec(
    name="pathmnist", num_classes=9, builder=_medmnist("pathmnist", 9),
    notes="MedMNIST: 107k colon histology patches, 9 tissue classes. "
          "Needs `pip install medmnist`. The pretraining ablation: a "
          "domain ImageNet knows nothing about, so it separates findings "
          "about FCL from findings about pretrained backbones."))

register(DatasetSpec(
    name="dermamnist", num_classes=7, builder=_medmnist("dermamnist", 7),
    notes="MedMNIST: 10k dermatoscopic images, 7 lesion classes. Smaller "
          "and more imbalanced than pathmnist, so a useful second point "
          "for the class-migration scenario."))

register(DatasetSpec(
    name="bloodmnist", num_classes=8, builder=_medmnist("bloodmnist", 8),
    notes="MedMNIST: 17k blood cell microscopy images, 8 classes. Well "
          "balanced, quick to train, good for rapid ablations."))


def get_dataset(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; "
                       f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list:
    return sorted(_REGISTRY)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", nargs="*", default=None,
                    help="dataset names to fetch/verify (default: all "
                         "auto-downloadable ones)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or args.download is None:
        print(f"data root: {os.path.abspath(DATA_ROOT)}\n")
        for n in available():
            s = get_dataset(n)
            dom = f", {len(s.native_domains)} native domains" if s.native_domains else ""
            man = "  [manual download]" if s.needs_manual_download else ""
            print(f"  {n:<14} {s.num_classes:>4} classes{dom}{man}")
            if s.notes:
                print(f"                 {s.notes}")
        raise SystemExit(0)

    names = args.download or [n for n in available()
                              if not get_dataset(n).needs_manual_download]
    os.makedirs(DATA_ROOT, exist_ok=True)
    for n in names:
        get_dataset(n).ensure()
    print("done.")
