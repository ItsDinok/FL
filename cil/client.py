"""
client_split.py — Split CIFAR-100 client with the full intervention-signal
benchmark (interference regime).

Per round:
  1. pre-training signals (gradient + predictive families)
  2. local training on the current task's partition
  3. post-training signals (performance, weight-space, representation)
  4. log every signal + per-group cost to logs_split/signals_client{cid}.csv
  5. snapshot at the end of round 1 (calibration reference) and at the
     final round of every task (the state that solved that task)
Per-task retained accuracy -> logs_split/metrics_client{cid}.csv

Carries all fixes from the class-migration condition: round-1 snapshot,
dynamic monitor device, CPU-parked models with per-GPU slot locks.

Scoring afterwards (offline):
    python horizon.py --logs logs_split/ --mode task --sweep

Uses a separate logs directory (logs_split/) and port (8081) so it can
coexist with, or even run alongside, the class-migration condition.
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
from cifardata import NUM_CLASSES, NUM_TASKS
from preflight import SignalMonitor, SIGNAL_COLUMNS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 3
LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-4
ROUNDS_PER_TASK = 5            # keep in sync with server_split.py
METRICS_DIR = "logs_split"

# minimal demonstration controller (off by default; phase-2 preview)
CONTROLLER_ENABLED = False
NEW_LOSS_GATE = 1.0            # trigger on zero-shot incoming loss
GATE_LR_SCALE = 0.25           # LR multiplier for gated rounds

# ── GPU slot serialization (per GPU) ────────────────────────────────────
GPU_SLOTS = 2
_LOCK_DIR = os.path.join(
    METRICS_DIR,
    ".gpu_slots_" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "_"))


@contextlib.contextmanager
def gpu_slot(poll_s: float = 2.0):
    """Acquire one of GPU_SLOTS exclusive slots on this client's GPU."""
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


def round_to_task(server_round: int) -> int:
    return min((server_round - 1) // ROUNDS_PER_TASK, NUM_TASKS - 1)


def is_task_final_round(server_round: int) -> bool:
    return server_round % ROUNDS_PER_TASK == 0


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


class SplitSignalClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.model = build_model()
        self.monitor = SignalMonitor(self.model, lr=LEARNING_RATE, seed=cid)
        self._loader_cache = {}
        self.eval_loaders = data.load_task_eval_loaders()

        os.makedirs(METRICS_DIR, exist_ok=True)
        self.metrics_csv = os.path.join(METRICS_DIR,
                                        f"metrics_client{cid}.csv")
        if not os.path.exists(self.metrics_csv):
            with open(self.metrics_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "task", "loss"]
                    + [f"acc_task{t}" for t in range(NUM_TASKS)]
                    + ["acc_mean"])

        self.signals_csv = os.path.join(METRICS_DIR,
                                        f"signals_client{cid}.csv")
        if not os.path.exists(self.signals_csv):
            with open(self.signals_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "task", "gated"] + SIGNAL_COLUMNS)

    def _loaders(self, task):
        if task not in self._loader_cache:
            self._loader_cache[task] = data.load_task_partition(
                self.cid, NUM_CLIENTS, task)
        return self._loader_cache[task]

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
        task = round_to_task(server_round)
        train_loader, _ = self._loaders(task)
        planned = LOCAL_EPOCHS * len(train_loader)

        with gpu_slot():
            self.model.to(DEVICE)

            # 1. pre-training signals
            sig = self.monitor.pre_training(train_loader,
                                            planned_steps=planned)

            # minimal controller: gate LR on a zero-shot loss spike
            lr, gated = LEARNING_RATE, 0
            if (CONTROLLER_ENABLED and sig["new_loss"] == sig["new_loss"]
                    and sig["new_loss"] > NEW_LOSS_GATE):
                lr, gated = LEARNING_RATE * GATE_LR_SCALE, 1
            print(f"[client {self.cid}] r{server_round} task {task} "
                  f"new_loss={sig['new_loss']:.3f} "
                  f"grad_cos={sig['grad_cos']:.3f} gated={gated}",
                  flush=True)

            # 2. train
            train(self.model, train_loader, epochs=LOCAL_EPOCHS, lr=lr)

            # 3. post-training signals
            sig = self.monitor.post_training(sig)

            # snapshot: round 1 (calibration reference) and task end
            if server_round == 1 or is_task_final_round(server_round):
                self.monitor.snapshot_task(task, train_loader.dataset)
                print(f"[client {self.cid}] snapshot stored for task "
                      f"{task} (round {server_round})", flush=True)

            self.model.to("cpu")
            torch.cuda.empty_cache()

        # 4. log (CPU only)
        with open(self.signals_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [server_round, task, gated]
                + [f"{sig[k]:.6f}" for k in SIGNAL_COLUMNS])

        # 5. buffer (CPU only)
        self.monitor.add_to_buffer(train_loader.dataset, n=128)

        fit_metrics = {f"c{self.cid}_task": task,
                       f"c{self.cid}_gated": gated}
        for k in ("new_loss", "grad_cos", "pred_dL_old", "ref_acc",
                  "weight_div_round", "centroid_drift", "cka_final"):
            v = sig[k]
            if v == v:
                fit_metrics[f"c{self.cid}_{k}"] = float(v)
        return (self.get_parameters(config={}),
                len(train_loader.dataset), fit_metrics)

    def evaluate(self, parameters, config):
        """Per-task retained accuracy — ground truth for t_forget."""
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", 0))
        task = round_to_task(server_round) if server_round else 0

        per_task, total_loss, total_n = {}, 0.0, 0
        with gpu_slot():
            self.model.to(DEVICE)
            for t, loader in self.eval_loaders.items():
                loss, acc = evaluate_loader(self.model, loader)
                per_task[t] = float(acc)
                n = len(loader.dataset)
                total_loss += loss * n
                total_n += n
            self.model.to("cpu")
            torch.cuda.empty_cache()
        mean_loss = total_loss / total_n
        mean_acc = sum(per_task.values()) / NUM_TASKS

        metrics = {f"c{self.cid}_acc_task{t}": a for t, a in per_task.items()}
        metrics[f"c{self.cid}_acc_mean"] = float(mean_acc)
        metrics["accuracy"] = float(mean_acc)

        with open(self.metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [server_round, task, f"{mean_loss:.4f}"]
                + [f"{per_task[t]:.4f}" for t in range(NUM_TASKS)]
                + [f"{mean_acc:.4f}"])
        return float(mean_loss), total_n, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", type=str, default="127.0.0.1:8081")
    args = parser.parse_args()

    fl.client.start_client(
        server_address=args.server,
        client=SplitSignalClient(args.cid).to_client(),
    )
