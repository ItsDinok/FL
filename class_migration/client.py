"""
client.py — generic federated continual-learning client.

Dataset- and scenario-agnostic: everything that varies comes from
config.py, datasets.py and scenarios.py, so a new dataset or scenario
requires no change here.

    python client.py --dataset cifar100 --scenario split --cid 0
    python client.py --dataset pacs --scenario domain --cid 0 \
                     --intra-round-eval

Per round each client:
  1. computes pre-training signals (gradient + predictive families)
  2. trains locally, optionally probing retained accuracy mid-round
  3. computes post-training signals (performance, weight-space,
     representation)
  4. writes signals, metrics and (optionally) step-level accuracy to CSV
  5. snapshots anchors at round 1 and at each block's final round

Scoring afterwards:
    python horizon.py --logs <logs dir> --mode <task|class> --sweep
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

import config
from preflight import SignalMonitor, SIGNAL_COLUMNS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── GPU slot serialisation (per GPU) ────────────────────────────────────
@contextlib.contextmanager
def gpu_slot(lock_dir, slots, poll_s=2.0):
    """Bound the number of clients holding GPU memory at once. Waiting
    clients keep their model parked on the CPU."""
    if DEVICE.type != "cuda":
        yield
        return
    os.makedirs(lock_dir, exist_ok=True)
    handle = None
    while handle is None:
        for i in range(slots):
            f = open(os.path.join(lock_dir, f"slot{i}.lock"), "w")
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


def train(model, loader, epochs, lr, probe=None, probe_every=50):
    """probe(step) fires at step 0 — before any update of this round —
    and every probe_every optimiser steps thereafter."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 0
    if probe is not None:
        probe(0)
        model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
            step += 1
            if probe is not None and step % probe_every == 0:
                probe(step)
                model.train()
    if probe is not None and step % probe_every != 0:
        probe(step)
        model.train()


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
    return loss / max(total, 1), correct / max(total, 1)


class SignalClient(fl.client.NumPyClient):
    def __init__(self, cid, args, scenario):
        self.cid, self.args, self.sc = cid, args, scenario
        self.model = config.build_model(args.model, scenario.num_classes)
        self.monitor = SignalMonitor(
            self.model, lr=args.lr, seed=cid,
            param_filters=config.MODELS[args.model]["filters"])
        self._cache = {}
        self.units = scenario.eval_units()
        self.unit_names = [scenario.unit_name(u) for u in self.units]

        self.dir = config.logs_dir(args)
        os.makedirs(self.dir, exist_ok=True)
        self.lock_dir = os.path.join(
            self.dir, ".gpu_slots_"
            + os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "_"))

        self.metrics_csv = self._init_csv(
            f"metrics_client{cid}.csv",
            ["round", "block", "loss"]
            + [f"acc_{n}" for n in self.unit_names] + ["acc_mean"])
        self.signals_csv = self._init_csv(
            f"signals_client{cid}.csv",
            ["round", "block", "gated"] + SIGNAL_COLUMNS)
        # class-migration scoring needs the per-block class proportions:
        # horizon.py --mode class derives each class's effective task end
        # from them. Deterministic, so written once at startup.
        if hasattr(scenario, "distribution"):
            path = os.path.join(self.dir, f"distribution_client{cid}.csv")
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["block"] + [f"p_{n}" for n in self.unit_names])
                    for b in range(scenario.num_blocks):
                        p = scenario.distribution(cid, b)
                        w.writerow([b] + [f"{v:.6f}" for v in p])

        self.steps_csv = self._init_csv(
            f"steps_client{cid}.csv",
            ["round", "block", "step", "global_step"]
            + [f"acc_{n}" for n in self.unit_names] + ["acc_mean"]
        ) if args.intra_round_eval else None

        # smaller probe set for intra-round evaluation
        self.probe_units = {}
        if args.intra_round_eval:
            from torch.utils.data import DataLoader, Subset
            for u, ld in self.units.items():
                n = min(args.probe_images, len(ld.dataset))
                self.probe_units[u] = DataLoader(
                    Subset(ld.dataset, list(range(n))),
                    batch_size=args.batch_size)

    def _init_csv(self, name, header):
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
        return path

    def _round_to_block(self, r):
        return min((r - 1) // self.args.rounds_per_block,
                   self.sc.num_blocks - 1)

    def _is_block_end(self, r):
        return r % self.args.rounds_per_block == 0

    def _loaders(self, block):
        if block not in self._cache:
            self._cache[block] = self.sc.partition(
                self.cid, self.args.clients, block)
        return self._cache[block]

    def get_parameters(self, config_):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = self.model.state_dict().keys()
        self.model.load_state_dict(OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}),
            strict=True)

    def fit(self, parameters, config_):
        self.set_parameters(parameters)
        rnd = int(config_.get("server_round", 1))
        block = self._round_to_block(rnd)
        train_loader, _ = self._loaders(block)
        a = self.args

        with gpu_slot(self.lock_dir, a.gpu_slots):
            self.model.to(DEVICE)
            steps = a.local_epochs * len(train_loader)
            sig = self.monitor.pre_training(train_loader, planned_steps=steps)

            lr, gated = a.lr, 0
            if a.controller:
                v = sig.get(a.gate_signal, float("nan"))
                if v == v and v > a.gate_threshold:
                    lr, gated = a.lr * a.gate_lr_scale, 1
            print(f"[c{self.cid}] r{rnd} block {block} "
                  f"({self.sc.block_name(block)}) "
                  f"new_loss={sig['new_loss']:.3f} gated={gated}", flush=True)

            probe = None
            if a.intra_round_eval:
                def probe(step):
                    accs = [evaluate_loader(self.model, ld)[1]
                            for ld in self.probe_units.values()]
                    with open(self.steps_csv, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [rnd, block, step, (rnd - 1) * steps + step]
                            + [f"{x:.4f}" for x in accs]
                            + [f"{sum(accs) / len(accs):.4f}"])

            train(self.model, train_loader, a.local_epochs, lr,
                  probe=probe, probe_every=a.probe_every_steps)

            sig = self.monitor.post_training(sig)
            if rnd == 1 or self._is_block_end(rnd):
                self.monitor.snapshot_task(block, train_loader.dataset)
                print(f"[c{self.cid}] snapshot: block {block} (r{rnd})",
                      flush=True)

            self.model.to("cpu")
            torch.cuda.empty_cache()

        with open(self.signals_csv, "a", newline="") as f:
            csv.writer(f).writerow([rnd, block, gated]
                                   + [f"{sig[k]:.6f}" for k in SIGNAL_COLUMNS])
        self.monitor.add_to_buffer(train_loader.dataset, n=128)

        m = {f"c{self.cid}_block": block, f"c{self.cid}_gated": gated}
        for k in ("new_loss", "grad_cos", "pred_dL_old", "ref_acc",
                  "weight_div_round", "centroid_drift", "cka_final"):
            if sig[k] == sig[k]:
                m[f"c{self.cid}_{k}"] = float(sig[k])
        return self.get_parameters({}), len(train_loader.dataset), m

    def evaluate(self, parameters, config_):
        self.set_parameters(parameters)
        rnd = int(config_.get("server_round", 0))
        block = self._round_to_block(rnd) if rnd else 0

        per_unit, tot_loss, tot_n = {}, 0.0, 0
        with gpu_slot(self.lock_dir, self.args.gpu_slots):
            self.model.to(DEVICE)
            for u, ld in self.units.items():
                loss, acc = evaluate_loader(self.model, ld)
                per_unit[u] = acc
                n = len(ld.dataset)
                tot_loss += loss * n
                tot_n += n
            self.model.to("cpu")
            torch.cuda.empty_cache()

        mean_acc = sum(per_unit.values()) / len(per_unit)
        with open(self.metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [rnd, block, f"{tot_loss / tot_n:.4f}"]
                + [f"{per_unit[u]:.4f}" for u in self.units]
                + [f"{mean_acc:.4f}"])

        metrics = {f"c{self.cid}_acc_{self.sc.unit_name(u)}": a
                   for u, a in per_unit.items()}
        metrics[f"c{self.cid}_acc_mean"] = float(mean_acc)
        metrics["accuracy"] = float(mean_acc)
        return float(tot_loss / tot_n), tot_n, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--server", default=None)
    config.add_arguments(ap)
    args = ap.parse_args()

    sc = config.make_scenario(args)
    if args.cid == 0:
        print(config.summarise(args, sc), flush=True)

    fl.client.start_client(
        server_address=args.server or f"127.0.0.1:{args.port}",
        client=SignalClient(args.cid, args, sc).to_client(),
    )
