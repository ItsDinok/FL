"""
preflight.py — unified intervention-signal benchmark for federated continual
learning.

Research question this serves:
    Which intervention signal provides the earliest and most reliable
    indication of catastrophic forgetting, at the lowest computational cost?

Signal taxonomy (all computed per client per round under one protocol):

  Performance-based (post-training):
    ref_loss        loss on reference buffer of past-task data
    ref_acc         accuracy on reference buffer

  Gradient-based (pre-training):
    grad_norm_new   ||g_new||  on incoming probe batches
    grad_norm_old   ||g_old||  on reference buffer
    grad_cos        cos(g_old, g_new)
    grad_inner      <g_old, g_new>

  Weight-space (post-training):
    weight_div_round  L2 distance moved this round (param subset)
    weight_div_task   L2 distance from end-of-previous-task weights

  Representation-space (post-training, on stored anchor sets):
    proto_drift     mean L2 drift of per-class CLS prototypes vs task-end
    centroid_drift  L2 drift of overall embedding centroid vs task-end
    cka_block{3,7,11}, cka_final
                    linear CKA between current and task-end activations
                    at early/mid/late encoder blocks and final embedding

  Predictive (pre-training):
    new_loss        zero-shot loss on incoming data (drift trigger)
    new_entropy     mean prediction entropy on incoming data
    pred_dL_old     -lr * k * <g_old, g_new>  (1st-order forgetting estimate)

  Cost accounting:
    t_grad_ms, t_weight_ms, t_repr_ms, t_perf_ms  wall-clock per group

Anchors: at the final round of each task the monitor snapshots (a) a small
fixed set of that task's images, (b) their activations at probe layers,
(c) per-class CLS prototypes, and (d) the current weights (subset). All
later drift/CKA/divergence values are measured against these snapshots, so
each signal answers: "how far has the model moved from the state that
solved task T?"
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PARAM_FILTERS = (
    "heads",
    "encoder.layers.encoder_layer_10",
    "encoder.layers.encoder_layer_11",
    "encoder.pos_embedding",
    "class_token",
)
PROBE_LAYERS = {"block3": 3, "block7": 7, "block11": 11}

NAN = float("nan")

SIGNAL_COLUMNS = [
    # predictive / gradient (pre-training)
    "new_loss", "new_entropy",
    "grad_norm_new", "grad_norm_old", "grad_cos", "grad_inner", "pred_dL_old",
    # performance (post-training)
    "ref_loss", "ref_acc",
    # weight-space (post-training)
    "weight_div_round", "weight_div_task",
    # representation-space (post-training)
    "proto_drift", "centroid_drift",
    "cka_block3", "cka_block7", "cka_block11", "cka_final",
    # cost accounting
    "t_grad_ms", "t_perf_ms", "t_weight_ms", "t_repr_ms",
]


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two [N, D] activation matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = (Y.T @ X).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return (num / den).item() if den > 0 else NAN


class SignalMonitor:
    def __init__(self, model: nn.Module, lr: float,
                 param_filters=DEFAULT_PARAM_FILTERS,
                 buffer_capacity: int = 512,
                 anchors_per_task: int = 64,
                 probe_batches: int = 4,
                 batch_size: int = 32,
                 seed: int = 0):
        self.model = model
        self.lr = lr
        self.probe_batches = probe_batches
        self.batch_size = batch_size
        self.anchors_per_task = anchors_per_task
        self.device = next(model.parameters()).device
        self._rng = torch.Generator().manual_seed(seed)
        self._criterion = nn.CrossEntropyLoss()

        self._params = [(n, p) for n, p in model.named_parameters()
                        if p.requires_grad and any(f in n for f in param_filters)]
        if not self._params:
            raise ValueError("param_filters matched no parameters")

        # reference buffer of past-task samples (CPU tensors)
        self.buf_x, self.buf_y = [], []
        self.buffer_capacity = buffer_capacity
        self._seen = 0

        # per-task snapshots {task_id: {...}}
        self.snapshots = {}
        self._prev_round_w = None   # weights at end of previous round
        self._task_end_w = None     # weights at end of previous task

        # forward hooks capture CLS token at probe layers + final embedding
        self._acts = {}
        enc = model.encoder
        for name, idx in PROBE_LAYERS.items():
            enc.layers[idx].register_forward_hook(self._make_hook(name))
        enc.ln.register_forward_hook(self._make_hook("final"))

    def _make_hook(self, name):
        def hook(_m, _i, out):
            self._acts[name] = out[:, 0, :].detach()   # CLS token
        return hook

    # ── low-level helpers ────────────────────────────────────────────────
    def _flat_w(self):
        return torch.cat([p.detach().flatten() for _, p in self._params]).cpu()

    def _zero(self):
        for _, p in self._params:
            p.grad = None

    def _grad(self, batches):
        """Average subset gradient + mean loss over (x, y) batches."""
        was = self.model.training
        self.model.eval()
        self._zero()
        loss_sum, n = 0.0, 0
        for x, y in batches:
            loss = self._criterion(self.model(x), y)
            loss.backward()
            loss_sum += loss.item()
            n += 1
        if n == 0:
            self._zero()
            if was:
                self.model.train()
            return None, NAN
        g = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for _, p in self._params]).detach() / n
        self._zero()
        if was:
            self.model.train()
        return g, loss_sum / n

    def _buffer_batches(self, max_batches):
        idx = torch.randperm(len(self.buf_x), generator=self._rng).tolist()
        idx = idx[: self.batch_size * max_batches]
        for s in range(0, len(idx), self.batch_size):
            sel = idx[s:s + self.batch_size]
            x = torch.stack([self.buf_x[i] for i in sel]).to(self.device)
            y = torch.tensor([self.buf_y[i] for i in sel], device=self.device)
            yield x, y

    @torch.no_grad()
    def _embed(self, x):
        """Forward pass returning dict of CLS activations at probe layers."""
        was = self.model.training
        self.model.eval()
        self._acts.clear()
        self.model(x)
        out = {k: v.clone() for k, v in self._acts.items()}
        if was:
            self.model.train()
        return out

    # ── lifecycle ─────────────────────────────────────────────────────────
    def add_to_buffer(self, dataset, n: int = 128):
        """Reservoir-sample past-task data into the reference buffer."""
        import random
        rng = random.Random(self._seen)
        for i in rng.sample(range(len(dataset)), min(n, len(dataset))):
            img, label = dataset[i]
            self._seen += 1
            if len(self.buf_x) < self.buffer_capacity:
                self.buf_x.append(img.cpu())
                self.buf_y.append(int(label))
            else:
                j = rng.randrange(self._seen)
                if j < self.buffer_capacity:
                    self.buf_x[j] = img.cpu()
                    self.buf_y[j] = int(label)

    def snapshot_task(self, task_id: int, dataset):
        """Call at the END of a task's final round: store anchors,
        activations, prototypes, and weights for later drift measurement."""
        import random
        rng = random.Random(1000 + task_id)
        idx = rng.sample(range(len(dataset)),
                         min(self.anchors_per_task, len(dataset)))
        xs = torch.stack([dataset[i][0] for i in idx])
        ys = torch.tensor([int(dataset[i][1]) for i in idx])

        with torch.no_grad():
            embs = []
            for s in range(0, len(xs), self.batch_size):
                a = self._embed(xs[s:s + self.batch_size].to(self.device))
                embs.append(a)
            acts = {k: torch.cat([e[k] for e in embs]).cpu()
                    for k in embs[0]}
        final = acts["final"]
        protos = {}
        for c in ys.unique().tolist():
            protos[c] = final[ys == c].mean(dim=0)

        self.snapshots[task_id] = {
            "x": xs.cpu(), "y": ys,
            "acts": acts,
            "protos": protos,
            "centroid": final.mean(dim=0),
        }
        self._task_end_w = self._flat_w()

    # ── the two measurement calls ────────────────────────────────────────
    def pre_training(self, new_loader, planned_steps: int) -> dict:
        """Gradient + predictive signals, computed BEFORE training."""
        out = {k: NAN for k in SIGNAL_COLUMNS}
        t0 = time.perf_counter()

        probe = []
        for i, (x, y) in enumerate(new_loader):
            if i >= self.probe_batches:
                break
            probe.append((x.to(self.device), y.to(self.device)))
        if not probe:
            return out

        g_new, new_loss = self._grad(probe)
        out["new_loss"] = new_loss
        out["grad_norm_new"] = g_new.norm().item()

        with torch.no_grad():
            was = self.model.training
            self.model.eval()
            ents, n = 0.0, 0
            for x, _ in probe:
                p = F.softmax(self.model(x), dim=1)
                ents += (-p * (p + 1e-12).log()).sum(1).mean().item()
                n += 1
            if was:
                self.model.train()
        out["new_entropy"] = ents / n

        if self.buf_x:
            g_old, _ = self._grad(self._buffer_batches(self.probe_batches))
            out["grad_norm_old"] = g_old.norm().item()
            inner = torch.dot(g_old, g_new).item()
            den = (g_old.norm() * g_new.norm()).item()
            out["grad_inner"] = inner
            out["grad_cos"] = inner / den if den > 0 else NAN
            out["pred_dL_old"] = -self.lr * planned_steps * inner

        out["t_grad_ms"] = (time.perf_counter() - t0) * 1000
        return out

    def post_training(self, pre: dict) -> dict:
        """Performance, weight-space, and representation signals AFTER the
        round's training. Merges into and returns the pre dict."""
        out = pre

        # performance-based
        t0 = time.perf_counter()
        if self.buf_x:
            was = self.model.training
            self.model.eval()
            loss_sum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in self._buffer_batches(self.probe_batches):
                    logits = self.model(x)
                    loss_sum += self._criterion(logits, y).item() * y.size(0)
                    correct += (logits.argmax(1) == y).sum().item()
                    total += y.size(0)
            if was:
                self.model.train()
            if total:
                out["ref_loss"] = loss_sum / total
                out["ref_acc"] = correct / total
        out["t_perf_ms"] = (time.perf_counter() - t0) * 1000

        # weight-space
        t0 = time.perf_counter()
        w = self._flat_w()
        if self._prev_round_w is not None:
            out["weight_div_round"] = (w - self._prev_round_w).norm().item()
        if self._task_end_w is not None:
            out["weight_div_task"] = (w - self._task_end_w).norm().item()
        self._prev_round_w = w
        out["t_weight_ms"] = (time.perf_counter() - t0) * 1000

        # representation-space (vs all stored task snapshots)
        t0 = time.perf_counter()
        if self.snapshots:
            proto_ds, cent_ds = [], []
            ckas = {k: [] for k in list(PROBE_LAYERS) + ["final"]}
            for snap in self.snapshots.values():
                xs = snap["x"]
                cur_parts = []
                for s in range(0, len(xs), self.batch_size):
                    cur_parts.append(
                        self._embed(xs[s:s + self.batch_size].to(self.device)))
                cur = {k: torch.cat([c[k] for c in cur_parts]).cpu()
                       for k in cur_parts[0]}

                final = cur["final"]
                ys = snap["y"]
                ds = [(final[ys == c].mean(0) - proto).norm().item()
                      for c, proto in snap["protos"].items()
                      if (ys == c).any()]
                if ds:
                    proto_ds.append(sum(ds) / len(ds))
                cent_ds.append(
                    (final.mean(0) - snap["centroid"]).norm().item())
                for k in ckas:
                    ckas[k].append(_linear_cka(snap["acts"][k], cur[k]))
            out["proto_drift"] = (sum(proto_ds) / len(proto_ds)
                                  if proto_ds else NAN)
            out["centroid_drift"] = sum(cent_ds) / len(cent_ds)
            for k, vs in ckas.items():
                out[f"cka_{k}"] = sum(vs) / len(vs)
        out["t_repr_ms"] = (time.perf_counter() - t0) * 1000
        return out
