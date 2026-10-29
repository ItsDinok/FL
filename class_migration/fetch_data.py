#!/usr/bin/env python3
"""
fetch_data.py — download and prepare every reachable dataset.

The container has no wget or curl, so everything here goes through
urllib. Compute nodes also reach only some hosts, so each download is
preceded by a short reachability probe and an unreachable host is
reported rather than retried until the job times out.

    python fetch_data.py              # fetch everything reachable
    python fetch_data.py --probe      # report reachability only, no fetch
    python fetch_data.py --only cifar100 tinyimagenet

Anything unreachable from the compute node can be fetched on the login
node (which usually has wider access) or locally and copied up; the
summary prints the exact command for each.
"""

import argparse
import os
import shutil
import socket
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile

DATA = os.environ.get("FL_DATA_ROOT", "./data")
TIMEOUT = 20
UA = {"User-Agent": "Mozilla/5.0"}


# ── helpers ─────────────────────────────────────────────────────────────
def reachable(url, timeout=TIMEOUT):
    """True if the host answers. Uses a HEAD-ish GET since some hosts
    reject HEAD outright."""
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 400
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


def download(url, dest, label):
    """Stream to disk with progress. Returns True on success."""
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, \
                open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {label}: {pct:5.1f}% "
                          f"({done >> 20}/{total >> 20} MB)",
                          end="", flush=True)
        print()
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print(f"\n    FAILED: {type(e).__name__}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def extract(path, into):
    print(f"    extracting {os.path.basename(path)} ...")
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(into)
    else:
        with tarfile.open(path) as t:
            t.extractall(into)
    os.remove(path)


def size_of(path):
    if not os.path.exists(path):
        return "-"
    if os.path.isfile(path):
        return f"{os.path.getsize(path) >> 20} MB"
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs)
    return f"{total >> 20} MB"


# ── per-dataset handlers ────────────────────────────────────────────────
def fetch_torchvision(name):
    """CIFAR-10, CIFAR-100 and EuroSAT come through torchvision."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets import get_dataset
    spec = get_dataset(name)
    spec.ensure()
    return True


def fetch_tinyimagenet():
    root = os.path.join(DATA, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(root, "train")):
        print("    already present")
    else:
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        if not reachable(url):
            return False, url
        z = os.path.join(DATA, "tiny-imagenet-200.zip")
        if not download(url, z, "tiny-imagenet"):
            return False, url
        extract(z, DATA)

    # val/ ships flat with an annotations file; ImageFolder needs class dirs
    val = os.path.join(root, "val")
    ann = os.path.join(val, "val_annotations.txt")
    if os.path.isfile(ann) and os.path.isdir(os.path.join(val, "images")):
        print("    reorganising val/ into class subdirectories")
        with open(ann) as f:
            pairs = [l.split("\t")[:2] for l in f]
        for fname, cls in pairs:
            d = os.path.join(val, cls)
            os.makedirs(d, exist_ok=True)
            src = os.path.join(val, "images", fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(d, fname))
        shutil.rmtree(os.path.join(val, "images"), ignore_errors=True)
        os.remove(ann)
        print(f"    reorganised {len(pairs)} validation images")
    return True, None


def fetch_cifar100c():
    root = os.path.join(DATA, "CIFAR-100-C")
    if os.path.isfile(os.path.join(root, "labels.npy")):
        print("    already present")
        return True, None
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    if not reachable(url):
        return False, url
    tar = os.path.join(DATA, "CIFAR-100-C.tar")
    if not download(url, tar, "CIFAR-100-C"):
        return False, url
    extract(tar, DATA)
    return True, None


def fetch_medmnist(flag):
    """MedMNIST installs from PyPI and pulls its .npz from Zenodo."""
    def go():
        try:
            import medmnist
        except ImportError:
            print("    medmnist not installed; attempting pip install")
            import subprocess
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "medmnist"], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    pip install failed: {r.stderr.strip()[:200]}")
                return False, "pip install medmnist"
            import medmnist
        from medmnist import INFO
        info = INFO[flag]
        cls = getattr(medmnist, info["python_class"])
        for split in ("train", "test"):
            ds = cls(split=split, root=DATA, download=True, as_rgb=True)
            print(f"    {split}: {len(ds)} images")
        return True, None
    return go


def check_pacs():
    if os.path.isdir(os.path.join(DATA, "pacs", "photo")):
        print("    already present")
        return True, None
    return False, "https://drive.google.com (manual)"


# ── main ────────────────────────────────────────────────────────────────
HANDLERS = [
    ("cifar100", lambda: (fetch_torchvision("cifar100"), None),
     "https://www.cs.toronto.edu"),
    ("cifar10", lambda: (fetch_torchvision("cifar10"), None),
     "https://www.cs.toronto.edu"),
    ("eurosat", lambda: (fetch_torchvision("eurosat"), None),
     "https://madm.dfki.de"),
    ("tinyimagenet", fetch_tinyimagenet,
     "http://cs231n.stanford.edu"),
    ("cifar100c", fetch_cifar100c,
     "https://zenodo.org"),
    ("pacs", check_pacs, "https://drive.google.com"),
    ("pathmnist", fetch_medmnist("pathmnist"), "https://zenodo.org"),
    ("dermamnist", fetch_medmnist("dermamnist"), "https://zenodo.org"),
    ("bloodmnist", fetch_medmnist("bloodmnist"), "https://zenodo.org"),
]

MANUAL_HINT = {
    "pathmnist": "pip install medmnist  # then rerun; data comes from Zenodo",
    "dermamnist": "pip install medmnist  # then rerun",
    "bloodmnist": "pip install medmnist  # then rerun",
    "eurosat": "host madm.dfki.de appears to be down/blocked; "
               "consider pathmnist instead for the pretraining ablation",
    "tinyimagenet": "wget http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                    "&& unzip -q tiny-imagenet-200.zip -d data/",
    "cifar100c": "wget https://zenodo.org/record/3555552/files/"
                 "CIFAR-100-C.tar && tar -xf CIFAR-100-C.tar -C data/",
    "pacs": "download from the PACS project page, then:\n"
            "        scp -r pacs <user>@<login node>:<path>/data/",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="report host reachability and exit")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these dataset names")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    print(f"data root: {os.path.abspath(DATA)}\n")

    todo = [h for h in HANDLERS
            if args.only is None or h[0] in args.only]

    print("=== host reachability ===")
    reach = {}
    for name, _, host in todo:
        ok = reachable(host)
        reach[name] = ok
        print(f"  {name:<14} {host:<32} {'OK' if ok else 'UNREACHABLE'}")
    print()

    if args.probe:
        return

    done, failed = [], []
    for name, fn, host in todo:
        print(f"=== {name} ===")
        if not reach[name] and name != "pacs":
            print("    host unreachable from this node, skipping")
            failed.append(name)
            print()
            continue
        try:
            ok, _ = fn()
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            ok = False
        (done if ok else failed).append(name)
        print()

    print("=== summary ===")
    for name, _, _ in todo:
        path = {"cifar100": "cifar-100-python",
                "cifar10": "cifar-10-batches-py",
                "eurosat": "eurosat",
                "tinyimagenet": "tiny-imagenet-200",
                "cifar100c": "CIFAR-100-C",
                "pacs": "pacs",
                "pathmnist": "pathmnist.npz",
                "dermamnist": "dermamnist.npz",
                "bloodmnist": "bloodmnist.npz"}[name]
        mark = "ready" if name in done else "MISSING"
        print(f"  {name:<14} {mark:<9} {size_of(os.path.join(DATA, path))}")

    if failed:
        print("\n=== fetch these another way ===")
        print("Try the login node first; it usually has wider network "
              "access than compute nodes.\n")
        for name in failed:
            print(f"  {name}:\n        {MANUAL_HINT.get(name, '-')}")


if __name__ == "__main__":
    main()
