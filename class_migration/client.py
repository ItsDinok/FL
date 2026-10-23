"""
client.py — class-migration (Dirichlet) client with the FULL
intervention-signal benchmark wired in.

This is the pre-feedback Dirichlet experiment instrumented with every
signal family from preflight.py, so signal accuracy can be compared on the
absence-driven forgetting regime:

  per round:  pre-training signals -> train -> post-training signals
  per block:  end-of-block snapshot (anchors, prototypes, weights)
  logs:       metrics_client{cid}.csv       per-class retained accuracy
              distribution_client{cid}.csv  Dirichlet proportions per block
              signals_client{cid}.csv       all signals + per-group cost

Scoring afterwards (offline):
  python horizon.py --logs logs/ --mode class
which computes per-class t_forget (with per-class effective task-end
windows from the distribution CSV), t_signal per signal, delta-t,
detection rate, false alarms, and cost — i.e. exactly "which signals
most accurately predict forgetting".
"""

import argparse
import contextlib
import csv
import fcntl
import os
import time
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

import cifardata as data
from cifardata import NUM_CLASSES, CLASS_NAMES
from preflight import SignalMonitor, SIGNAL_COLUMNS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 10
LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-4
ROUNDS_PER_BLOCK = 5           # keep in sync with server.py
NUM_BLOCKS = 4
ALPHA = 0.5                    # Dirichlet concentration (K=100)
METRICS_DIR = "logs"

# ── GPU slot serialization ──────────────────────────────────────────────
# With many clients on one GPU, only GPU_SLOTS clients may occupy it at a
# time; the rest wait with their model parked on CPU. VRAM usage is then
# bounded by GPU_SLOTS x per-client footprint (+ one CUDA context per
# process, ~0.5GB each, which persists regardless).
GPU_SLOTS = 2
# Per-GPU lock pool: clients pinned to different GPUs (via
# CUDA_VISIBLE_DEVICES in the launcher) contend only with clients on the
# same card, so total concurrency = num_gpus x GPU_SLOTS.
_LOCK_DIR = os.path.join(
    METRICS_DIR,
    ".gpu_slots_" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "_"))


@contextlib.contextmanager
def gpu_slot(poll_s: float = 2.0):
    """Acquire one of GPU_SLOTS exclusive slots (cross-process flock)."""
    if DEVICE.type != "cuda":
        yield
        return
    os.makedirs(_LOCK_DIR, exist_ok=True)
    handle = None
    while handle is None:
        for i in range(GPU_SLOTS):
            f = open(os.path.join(_LOCK_DIR, f"slot{i}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = f
                break
            except OSError:
                f.close()
        if handle is None:
            time.sleep(poll_s)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Built on CPU; moved to GPU only while holding a gpu_slot."""
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def round_to_block(server_round: int) -> int:
    return min((server_round - 1) // ROUNDS_PER_BLOCK, NUM_BLOCKS - 1)


def is_block_final_round(server_round: int) -> bool:
    return server_round % ROUNDS_PER_BLOCK == 0


def train(model, loader, epochs: int, lr: float):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def evaluate_loader(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            loss += criterion(out, labels).item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss / total, correct / total


class MigrationSignalClient(fl.client.NumPyClient):
    def __init__(self, cid: int, alpha: float):
        self.cid = cid
        self.alpha = alpha
        self.model = build_model()
        self.monitor = SignalMonitor(self.model, lr=LEARNING_RATE, seed=cid)
        self._loader_cache = {}
        self.eval_loaders = data.load_per_class_eval_loaders()

        os.makedirs(METRICS_DIR, exist_ok=True)

        self.metrics_csv = os.path.join(METRICS_DIR,
                                        f"metrics_client{cid}.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "block", "dominant_class", "loss"]
                    + [f"acc_{n}" for n in CLASS_NAMES]
                    + ["acc_mean"])

        # Dirichlet proportions per block -> the effective task-end
        # windows for horizon.py --mode class
        self.dist_csv = os.path.join(METRICS_DIR,
                                     f"distribution_client{cid}.csv")
        if not os.path.exists(self.dist_csv):
            with open(self.dist_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["block"] + [f"p_{n}" for n in CLASS_NAMES])
                for b in range(NUM_BLOCKS):
                    p = data.get_block_distribution(cid, alpha, b)
                    csv.writer(f).writerow(
                        [b] + [f"{v:.6f}" for v in p])

        self.signals_csv = os.path.join(METRICS_DIR,
                                        f"signals_client{cid}.csv")
        if not os.path.exists(self.signals_csv):
            with open(self.signals_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "block", "dominant_class"] + SIGNAL_COLUMNS)

    def _loaders(self, block):
        if block not in self._loader_cache:
            self._loader_cache[block] = data.load_dirichlet_partition(
                self.cid, NUM_CLIENTS, self.alpha, block)
        return self._loader_cache[block]

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = self.model.state_dict().keys()
        self.model.load_state_dict(OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}),
            strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", 1))
        block = round_to_block(server_round)
        p = data.get_block_distribution(self.cid, self.alpha, block)
        dominant = CLASS_NAMES[int(p.argmax())]
        train_loader, _ = self._loaders(block)
        planned = LOCAL_EPOCHS * len(train_loader)

        with gpu_slot():
            self.model.to(DEVICE)

            # 1. pre-training signals
            sig = self.monitor.pre_training(train_loader,
                                            planned_steps=planned)
            print(f"[client {self.cid}] r{server_round} block {block} "
                  f"dominant={dominant} new_loss={sig['new_loss']:.3f} "
                  f"grad_cos={sig['grad_cos']:.3f}", flush=True)

            # 2. train
            train(self.model, train_loader, epochs=LOCAL_EPOCHS,
                  lr=LEARNING_RATE)

            # 3. post-training signals
            sig = self.monitor.post_training(sig)

            # 5a. snapshot needs the GPU (embeds anchors)
            if server_round == 1 or is_block_final_round(server_round):
                self.monitor.snapshot_task(block, train_loader.dataset)
                print(f"[client {self.cid}] snapshot stored for block "
                      f"{block} (round {server_round})", flush=True)

            # park the model and release the slot
            self.model.to("cpu")
            torch.cuda.empty_cache()

        # 4. log (no GPU needed)
        with open(self.signals_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [server_round, block, dominant]
                + [f"{sig[k]:.6f}" for k in SIGNAL_COLUMNS])

        # 5b. buffer add is CPU-only
        self.monitor.add_to_buffer(train_loader.dataset, n=128)

        fit_metrics = {f"c{self.cid}_block": block}
        for k in ("new_loss", "grad_cos", "pred_dL_old", "proto_drift",
                  "weight_div_task", "cka_final", "ref_acc"):
            v = sig[k]
            if v == v:
                fit_metrics[f"c{self.cid}_{k}"] = float(v)
        return (self.get_parameters(config={}),
                len(train_loader.dataset), fit_metrics)

    def evaluate(self, parameters, config):
        """Per-class retained accuracy — ground truth for per-class
        t_forget in horizon.py --mode class."""
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", 0))
        block = round_to_block(server_round) if server_round else 0
        p = data.get_block_distribution(self.cid, self.alpha, block)
        dominant = CLASS_NAMES[int(p.argmax())]

        per_class, total_loss, total_n = {}, 0.0, 0
        with gpu_slot():
            self.model.to(DEVICE)
            for c, loader in self.eval_loaders.items():
                loss, acc = evaluate_loader(self.model, loader)
                per_class[c] = float(acc)
                n = len(loader.dataset)
                total_loss += loss * n
                total_n += n
            self.model.to("cpu")
            torch.cuda.empty_cache()
        mean_loss = total_loss / total_n
        mean_acc = sum(per_class.values()) / NUM_CLASSES

        metrics = {f"c{self.cid}_acc_mean": float(mean_acc),
                   "accuracy": float(mean_acc)}

        with open(self.metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [server_round, block, dominant, f"{mean_loss:.4f}"]
                + [f"{per_class[c]:.4f}" for c in range(NUM_CLASSES)]
                + [f"{mean_acc:.4f}"])
        return float(mean_loss), total_n, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    fl.client.start_client(
        server_address=args.server,
        client=MigrationSignalClient(args.cid, args.alpha).to_client(),
    )
