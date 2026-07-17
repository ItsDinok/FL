"""
Flower client for domain-incremental federated continual learning,
configured to produce PRONOUNCED forgetting (baseline for
forgetting-prevention research).

Key knobs:
  DRIFT_LAG    rounds by which client cid's drift lags client 0.
               0  = fully correlated drift (max forgetting; all clients
                    leave a domain at the same time, so nothing keeps it
                    alive in aggregation)
               ROUNDS_PER_DOMAIN = fully staggered (implicit replay,
                    minimal forgetting — the regime from the previous run)
               Sweep 0..ROUNDS_PER_DOMAIN to get forgetting-vs-drift-
               correlation as a controlled axis.
  LEARNING_RATE / LOCAL_EPOCHS
               hotter optimization -> the model moves further toward the
               current domain each block -> more overwriting of old ones.

All clients share the same domain ORDER (clean -> rotate -> permute ->
noise); heterogeneity is in the TIMING via DRIFT_LAG. Per-client metrics
(c{cid}_acc_*) and per-round CSV logging are unchanged.
"""

import argparse
import csv
import os
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

import cifardata
from cifardata import NUM_DOMAINS, DOMAIN_NAMES
from preflight import Preflight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
NUM_CLIENTS = 3            # must match how many client processes you launch
LOCAL_EPOCHS = 2           # was 1: more steps per block -> more forgetting
LEARNING_RATE = 1e-4       # was 3e-5: hotter -> more overwriting
ROUNDS_PER_DOMAIN = 5      # keep in sync with server.py
DRIFT_LAG = 0              # 0 = correlated drift (max forgetting)
PREFLIGHT_ENABLED = True   # assess batch impact before each round's training
METRICS_DIR = "logs"


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(DEVICE)


def round_to_domain(server_round: int, cid: int,
                    drift_lag: int = DRIFT_LAG) -> int:
    """All clients traverse the same domain sequence; client cid starts
    each block `cid * drift_lag` rounds later. With drift_lag=0 every
    client is always on the same domain (fully correlated drift)."""
    effective_round = max(server_round - cid * drift_lag, 1)
    block = min((effective_round - 1) // ROUNDS_PER_DOMAIN, NUM_DOMAINS - 1)
    return block   # base order is 0,1,2,3 so block index == domain id


def train(model, loader, epochs: int):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss / total, correct / total


class ContinualViTClient(fl.client.NumPyClient):
    def __init__(self, cid: int, drift_lag: int):
        self.cid = cid
        self.drift_lag = drift_lag
        self.model = build_model()
        self._loader_cache = {}
        self.eval_loaders = cifardata.load_all_domain_eval_loaders(
            cid, NUM_CLIENTS
        )
        print(f"[client {cid}] drift_lag={drift_lag} "
              f"(starts each block {cid * drift_lag} rounds after client 0)",
              flush=True)

        os.makedirs(METRICS_DIR, exist_ok=True)
        self.csv_path = os.path.join(METRICS_DIR, f"metrics_client{cid}.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "trained_domain", "loss"]
                                + [f"acc_{name}" for name in DOMAIN_NAMES]
                                + ["acc_mean"])
        self._last_trained_domain = None

        # ---- preflight: assess incoming data BEFORE training on it ----
        self.preflight = Preflight(self.model, lr=LEARNING_RATE, seed=cid) \
            if PREFLIGHT_ENABLED else None
        self.preflight_csv = os.path.join(
            METRICS_DIR, f"preflight_client{cid}.csv")
        if PREFLIGHT_ENABLED and not os.path.exists(self.preflight_csv):
            with open(self.preflight_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["round", "incoming_domain", "cos_conflict",
                     "inner_prod", "pred_dL_old_1st", "fisher_overlap",
                     "curvature_term", "new_loss", "new_entropy",
                     "ref_loss"])

    def _loaders_for_domain(self, domain_id: int):
        if domain_id not in self._loader_cache:
            self._loader_cache[domain_id] = cifardata.load_domain_partition(
                self.cid, NUM_CLIENTS, domain_id
            )
        return self._loader_cache[domain_id]

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = self.model.state_dict().keys()
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", 1))
        domain = round_to_domain(server_round, self.cid, self.drift_lag)
        self._last_trained_domain = domain
        print(f"[client {self.cid}] round {server_round}: training on "
              f"domain {domain} ({DOMAIN_NAMES[domain]})", flush=True)

        train_loader, _ = self._loaders_for_domain(domain)

        fit_metrics = {f"c{self.cid}_trained_domain": domain}
        if self.preflight is not None:
            planned = LOCAL_EPOCHS * len(train_loader)
            report = self.preflight.assess(train_loader, planned_steps=planned)
            print(f"[client {self.cid}] preflight: "
                  f"cos={report['cos_conflict']:.3f} "
                  f"pred_dL_old={report['pred_dL_old_1st']:.3f} "
                  f"new_loss={report['new_loss']:.3f}", flush=True)
            with open(self.preflight_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [server_round, DOMAIN_NAMES[domain]]
                    + [f"{report[k]:.6f}" for k in
                       ("cos_conflict", "inner_prod", "pred_dL_old_1st",
                        "fisher_overlap", "curvature_term", "new_loss",
                        "new_entropy", "ref_loss")])
            for k in ("cos_conflict", "pred_dL_old_1st", "new_loss"):
                if report[k] == report[k]:   # skip NaN (round 1)
                    fit_metrics[f"c{self.cid}_pf_{k}"] = float(report[k])

        train(self.model, train_loader, epochs=LOCAL_EPOCHS)

        if self.preflight is not None:
            # fold the just-trained domain into the reference distribution
            self.preflight.update_buffer(train_loader.dataset, n=128)

        return (self.get_parameters(config={}),
                len(train_loader.dataset),
                fit_metrics)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", 0))

        per_domain, total_loss, total_n = {}, 0.0, 0
        for d, loader in self.eval_loaders.items():
            loss, acc = evaluate(self.model, loader)
            per_domain[d] = float(acc)
            n = len(loader.dataset)
            total_loss += loss * n
            total_n += n
        mean_loss = total_loss / total_n
        mean_acc = sum(per_domain.values()) / len(per_domain)

        metrics = {f"c{self.cid}_acc_{DOMAIN_NAMES[d]}": a
                   for d, a in per_domain.items()}
        metrics[f"c{self.cid}_acc_mean"] = float(mean_acc)
        metrics["accuracy"] = float(mean_acc)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            trained = (DOMAIN_NAMES[self._last_trained_domain]
                       if self._last_trained_domain is not None else "")
            writer.writerow([server_round, trained, f"{mean_loss:.4f}"]
                            + [f"{per_domain[d]:.4f}"
                               for d in range(NUM_DOMAINS)]
                            + [f"{mean_acc:.4f}"])

        return float(mean_loss), total_n, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--drift-lag", type=int, default=DRIFT_LAG,
                        help="rounds by which client cid lags client 0's "
                             "drift (0 = fully correlated, max forgetting)")
    args = parser.parse_args()

    fl.client.start_client(
        server_address=args.server,
        client=ContinualViTClient(args.cid, args.drift_lag).to_client(),
    )
