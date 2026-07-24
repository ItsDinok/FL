"""
Flower client for heterogeneous Dirichlet class-distribution federated
continual learning with a Vision Transformer.

Each client independently resamples from Dir(ALPHA * 1_K) every block.
Evaluation reports per-class accuracy for all 10 CIFAR-10 classes so
class-level forgetting is visible round by round.

Outputs per client:
  logs/metrics_client{cid}.csv    — per-class accuracy each round
  logs/distribution_client{cid}.csv — Dirichlet proportions per block
  logs/preflight_client{cid}.csv  — pre-training gradient signals
"""

import argparse
import csv
import os
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore", message=".*cudnnException.*")

import flwr as fl
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

import cifardata
from cifardata import (NUM_CLASSES, CLASS_NAMES,
                       get_block_distribution,
                       load_dirichlet_partition,
                       load_per_class_eval_loaders)
from preflight import Preflight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS        = 3
LOCAL_EPOCHS       = 2
LEARNING_RATE      = 1e-4
ROUNDS_PER_BLOCK   = 5
NUM_BLOCKS         = 4
ALPHA              = 0.5    # 0.1 is pathologically sparse at K=100 classes
PREFLIGHT_ENABLED  = True
METRICS_DIR        = "logs"


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(DEVICE)


def round_to_block(server_round: int) -> int:
    return min((server_round - 1) // ROUNDS_PER_BLOCK, NUM_BLOCKS - 1)


def train(model, loader, epochs: int):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for ep in range(epochs):
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"  ep{ep+1} batch {i}/{len(loader)} "
                      f"loss={loss.item():.3f}", flush=True)


def eval_loader(model, loader):
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


class DirichletCLClient(fl.client.NumPyClient):
    def __init__(self, cid: int, alpha: float):
        self.cid   = cid
        self.alpha = alpha
        self.model = build_model()
        self._loader_cache = {}
        self.eval_loaders  = load_per_class_eval_loaders(samples_per_class=100)

        os.makedirs(METRICS_DIR, exist_ok=True)

        # print startup schedule
        print(f"[client {cid}] CIFAR-10 class distribution experiment  "
              f"alpha={alpha}", flush=True)
        for b in range(NUM_BLOCKS):
            p = get_block_distribution(cid, alpha, b)
            top3 = sorted(range(NUM_CLASSES), key=lambda c: -p[c])[:3]
            print(f"  block {b}: "
                  + ", ".join(f"{CLASS_NAMES[c]}={p[c]:.2f}" for c in top3),
                  flush=True)

        # distribution CSV (written once at startup)
        dist_csv = os.path.join(METRICS_DIR, f"distribution_client{cid}.csv")
        if not os.path.exists(dist_csv):
            with open(dist_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["block", "dominant_class"]
                    + [f"p_{n}" for n in CLASS_NAMES])
            for b in range(NUM_BLOCKS):
                p = get_block_distribution(cid, alpha, b)
                with open(dist_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [b, CLASS_NAMES[int(p.argmax())]]
                        + [f"{v:.4f}" for v in p])

        # per-round metrics CSV
        self.csv_path = os.path.join(METRICS_DIR, f"metrics_client{cid}.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "block", "dominant_class", "loss"]
                    + [f"acc_{n}" for n in CLASS_NAMES]
                    + ["acc_mean"])

        # preflight
        self.preflight = (Preflight(self.model, lr=LEARNING_RATE, seed=cid)
                          if PREFLIGHT_ENABLED else None)
        self.pf_csv = os.path.join(METRICS_DIR, f"preflight_client{cid}.csv")
        if PREFLIGHT_ENABLED and not os.path.exists(self.pf_csv):
            with open(self.pf_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "block", "dominant_class",
                     "cos_conflict", "inner_prod", "pred_dL_old_1st",
                     "fisher_overlap", "curvature_term",
                     "new_loss", "new_entropy", "ref_loss"])

    def _loaders(self, block: int):
        if block not in self._loader_cache:
            self._loader_cache[block] = load_dirichlet_partition(
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
        block        = round_to_block(server_round)
        p            = get_block_distribution(self.cid, self.alpha, block)
        dominant     = CLASS_NAMES[int(p.argmax())]

        print(f"[client {self.cid}] round {server_round} block {block} "
              f"dominant={dominant}({p.max():.2f})", flush=True)

        train_loader, _ = self._loaders(block)
        fit_metrics = {f"c{self.cid}_block": block}

        if self.preflight is not None:
            planned = LOCAL_EPOCHS * len(train_loader)
            report  = self.preflight.assess(train_loader, planned_steps=planned)
            print(f"[client {self.cid}] preflight: "
                  f"cos={report['cos_conflict']:.3f}  "
                  f"pred_dL={report['pred_dL_old_1st']:.4f}  "
                  f"new_loss={report['new_loss']:.3f}", flush=True)
            with open(self.pf_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [server_round, block, dominant]
                    + [f"{report[k]:.6f}" for k in
                       ("cos_conflict", "inner_prod", "pred_dL_old_1st",
                        "fisher_overlap", "curvature_term",
                        "new_loss", "new_entropy", "ref_loss")])
            for k in ("cos_conflict", "pred_dL_old_1st"):
                v = report[k]
                if v == v:
                    fit_metrics[f"c{self.cid}_pf_{k}"] = float(v)

        train(self.model, train_loader, epochs=LOCAL_EPOCHS)

        if self.preflight is not None:
            self.preflight.update_buffer(train_loader.dataset, n=128)

        return self.get_parameters({}), len(train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", 0))
        block        = round_to_block(server_round) if server_round else 0
        p            = get_block_distribution(self.cid, self.alpha, block)
        dominant     = CLASS_NAMES[int(p.argmax())]

        per_class, total_loss, total_n = {}, 0.0, 0
        for c, loader in self.eval_loaders.items():
            loss, acc = eval_loader(self.model, loader)
            per_class[c] = float(acc)
            n = len(loader.dataset)
            total_loss += loss * n
            total_n    += n

        mean_loss = total_loss / total_n
        mean_acc  = sum(per_class.values()) / NUM_CLASSES

        # per-client metric keys so server history keeps clients separate
        metrics = {f"c{self.cid}_acc_{CLASS_NAMES[c]}": a
                   for c, a in per_class.items()}
        metrics[f"c{self.cid}_acc_mean"] = float(mean_acc)
        metrics["accuracy"]              = float(mean_acc)

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [server_round, block, dominant, f"{mean_loss:.4f}"]
                + [f"{per_class[c]:.4f}" for c in range(NUM_CLASSES)]
                + [f"{mean_acc:.4f}"])

        return float(mean_loss), total_n, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid",   type=int,   required=True)
    parser.add_argument("--server",type=str,   default="127.0.0.1:8080")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()
    fl.client.start_client(
        server_address=args.server,
        client=DirichletCLClient(args.cid, args.alpha).to_client(),
    )
