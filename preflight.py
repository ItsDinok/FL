"""
preflight.py — assess the impact of a new batch of data BEFORE training on it.

Theory (first-order): one SGD step theta' = theta - eta * g_B changes the
old-domain loss by
    dL_old ~= -eta * <g_old, g_B>  +  (eta^2 / 2) * g_B^T H_old g_B
so with quantities computable before the step:
    <g_old, g_B> < 0        -> the step provably (locally) increases old loss
    -eta * k * <g_old, g_B> -> predicted first-order damage over k planned steps
    g_B^T F g_B (F ~= diag Fisher of old data) -> curvature/importance term (EWC-style)

All signals below are computed from ONE probe pass over the incoming data and
a small reference buffer of past-domain samples. No training step is taken.

Signals returned by Preflight.assess():
    cos_conflict        cosine(g_old, g_B); negative = destructive direction
    inner_prod          <g_old, g_B> (raw, subset of parameters)
    pred_dL_old_1st     -lr * planned_steps * inner_prod
                        (predicted first-order change in old-domain loss;
                        positive = forgetting predicted. NOTE: leading-order
                        only — trust the SIGN and RANKING, not the magnitude,
                        over thousands of steps)
    fisher_overlap      normalized <F, g_B^2> in [0,1]; how much of the new
                        batch's gradient mass lands on parameters the old
                        domains are sensitive to
    curvature_term      0.5 * lr^2 * g_B^T F g_B (per-step 2nd-order term)
    new_loss            zero-shot loss of current model on incoming data
    new_entropy         mean prediction entropy on incoming data (drift signal)
    ref_loss            current loss on the reference buffer (baseline)

Gradients are restricted to a parameter subset (head + last blocks +
positional embedding by default) — this keeps cost/memory low, and it is
where domain adaptation concentrates in a pretrained ViT. The subset is a
design choice you may want to ablate.

Cost per call: ~2-3 forward+backward passes on small probe sets.

Usage in a Flower client:
    pf = Preflight(model, lr=LEARNING_RATE)
    ...
    # in fit(), BEFORE training:
    report = pf.assess(train_loader, planned_steps=LOCAL_EPOCHS * len(train_loader))
    # in fit(), AFTER training:
    pf.update_buffer(train_loader.dataset)
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PARAM_FILTERS = (
    "heads",                        # classification head
    "encoder.layers.encoder_layer_10",
    "encoder.layers.encoder_layer_11",
    "encoder.pos_embedding",        # where rotation-style shifts should land
    "class_token",
)


class ReferenceBuffer:
    """Reservoir of (image, label) samples from past training data.
    Stored on CPU; sampled into batches on demand."""

    def __init__(self, capacity: int = 512, seed: int = 0):
        self.capacity = capacity
        self.images, self.labels = [], []
        self._seen = 0
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.images)

    def add_from_dataset(self, dataset, n: int = 128):
        """Reservoir-sample n items from a dataset into the buffer."""
        idxs = self._rng.sample(range(len(dataset)), min(n, len(dataset)))
        for i in idxs:
            img, label = dataset[i]
            self._seen += 1
            if len(self.images) < self.capacity:
                self.images.append(img.cpu())
                self.labels.append(int(label))
            else:
                j = self._rng.randrange(self._seen)
                if j < self.capacity:
                    self.images[j] = img.cpu()
                    self.labels[j] = int(label)

    def batches(self, batch_size: int, max_batches: int, device):
        order = list(range(len(self.images)))
        self._rng.shuffle(order)
        order = order[: batch_size * max_batches]
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            x = torch.stack([self.images[i] for i in idx]).to(device)
            y = torch.tensor([self.labels[i] for i in idx], device=device)
            yield x, y


class Preflight:
    def __init__(self, model: nn.Module, lr: float,
                 param_filters=DEFAULT_PARAM_FILTERS,
                 buffer_capacity: int = 512,
                 probe_batches: int = 4,
                 seed: int = 0):
        self.model = model
        self.lr = lr
        self.probe_batches = probe_batches
        self.buffer = ReferenceBuffer(buffer_capacity, seed=seed)
        self.device = next(model.parameters()).device
        self._params = [
            (name, p) for name, p in model.named_parameters()
            if p.requires_grad and any(f in name for f in param_filters)
        ]
        if not self._params:
            raise ValueError("param_filters matched no parameters")
        self._criterion = nn.CrossEntropyLoss()
        # diag Fisher over the subset, refreshed on update_buffer()
        self._fisher = None

    # ------------------------------------------------------------------
    # gradient helpers (subset only)
    # ------------------------------------------------------------------
    def _zero(self):
        for _, p in self._params:
            if p.grad is not None:
                p.grad = None

    def _collect_flat_grad(self):
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for _, p in self._params
        ]).detach()

    def _grad_and_loss(self, batches):
        """Average gradient (subset) and mean loss over an iterable of
        (x, y) batches. Leaves model params untouched."""
        was_training = self.model.training
        self.model.eval()   # deterministic probe (no dropout)
        self._zero()
        total_loss, n_batches = 0.0, 0
        for x, y in batches:
            loss = self._criterion(self.model(x), y)
            loss.backward()          # grads ACCUMULATE across batches
            total_loss += loss.item()
            n_batches += 1
        if n_batches == 0:
            self._zero()
            return None, None
        g = self._collect_flat_grad() / n_batches
        self._zero()
        if was_training:
            self.model.train()
        return g, total_loss / n_batches

    @torch.no_grad()
    def _entropy(self, batches):
        ents, n = 0.0, 0
        was_training = self.model.training
        self.model.eval()
        for x, _ in batches:
            p = F.softmax(self.model(x), dim=1)
            ents += (-p * (p + 1e-12).log()).sum(dim=1).mean().item()
            n += 1
        if was_training:
            self.model.train()
        return ents / max(n, 1)

    def _refresh_fisher(self, batch_size: int = 32, max_batches: int = 8):
        """Diagonal empirical Fisher over the subset, estimated from the
        reference buffer using squared MINI-BATCH gradients (coarser than
        per-sample Fisher; documented approximation)."""
        if len(self.buffer) == 0:
            self._fisher = None
            return
        acc, n = None, 0
        for x, y in self.buffer.batches(batch_size, max_batches, self.device):
            g, _ = self._grad_and_loss([(x, y)])
            acc = g.pow(2) if acc is None else acc + g.pow(2)
            n += 1
        self._fisher = acc / n if n else None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def update_buffer(self, dataset, n: int = 128):
        """Call AFTER training on a round's data: fold samples of the
        just-trained domain into the reference distribution, then refresh
        the Fisher estimate at the new weights."""
        self.buffer.add_from_dataset(dataset, n=n)
        self._refresh_fisher()

    def assess(self, new_loader, planned_steps: int,
               batch_size: int = 32) -> dict:
        """Compute all preflight signals for the incoming data WITHOUT
        taking a training step. Returns a dict of floats (NaN where a
        signal is undefined, e.g. empty buffer on the first round)."""
        nan = float("nan")
        out = {"cos_conflict": nan, "inner_prod": nan,
               "pred_dL_old_1st": nan, "fisher_overlap": nan,
               "curvature_term": nan, "new_loss": nan,
               "new_entropy": nan, "ref_loss": nan}

        # probe batches from the incoming data
        probe = []
        for i, (x, y) in enumerate(new_loader):
            if i >= self.probe_batches:
                break
            probe.append((x.to(self.device), y.to(self.device)))
        if not probe:
            return out

        g_new, new_loss = self._grad_and_loss(probe)
        out["new_loss"] = new_loss
        out["new_entropy"] = self._entropy(probe)

        if len(self.buffer) == 0:
            return out   # no past data yet: drift signals only

        ref = self.buffer.batches(batch_size, self.probe_batches, self.device)
        g_old, ref_loss = self._grad_and_loss(ref)
        out["ref_loss"] = ref_loss

        inner = torch.dot(g_old, g_new).item()
        denom = (g_old.norm() * g_new.norm()).item()
        out["inner_prod"] = inner
        out["cos_conflict"] = inner / denom if denom > 0 else nan
        out["pred_dL_old_1st"] = -self.lr * planned_steps * inner

        if self._fisher is not None:
            g2 = g_new.pow(2)
            fg = torch.dot(self._fisher, g2).item()
            fnorm = (self._fisher.norm() * g2.norm()).item()
            out["fisher_overlap"] = fg / fnorm if fnorm > 0 else nan
            out["curvature_term"] = 0.5 * (self.lr ** 2) * fg

        # guard against any accidental non-finite values leaking into CSVs
        for k, v in out.items():
            if v is not None and not math.isfinite(v):
                out[k] = nan
        return out
