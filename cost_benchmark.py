#!/usr/bin/env python3
"""
cost_benchmark.py — marginal per-signal computational cost.

Addresses the limitation that cost is reported per computation *group*
(grad / perf / weight / repr), so signals inside a group cannot be
separated on cost. The group figure answers "what does this family cost
if I compute all of it?"; a deployment needs "what does it cost to add
this one signal to a monitor that computes nothing else?"

This script measures the second quantity. Each signal is timed in
isolation on a real ViT-B/16 forward/backward path, with the shared
prerequisites attributed to whichever signal requires them:

    new_loss            1 forward on the probe
    new_entropy         1 forward on the probe (softmax only)
    grad_norm_new       1 forward + backward on the probe
    grad_norm_old       1 forward + backward on the buffer
    grad_cos            both gradients (new + old)
    grad_inner          both gradients
    pred_dL_old         both gradients
    ref_loss / ref_acc  1 forward on the buffer
    weight_div_round    parameter arithmetic only
    weight_div_task     parameter arithmetic only
    proto_drift         1 forward on the anchors, per stored block
    centroid_drift      1 forward on the anchors, per stored block
    cka_block{3,7,11}   1 forward on the anchors (hooked), per block
    cka_final           1 forward on the anchors (hooked), per block

Reported alongside the marginal cost is the *incremental* cost of adding
each signal to a monitor that already computes the cheapest full
detector, which is the number that matters when choosing what to deploy.

Usage:
    python3 cost_benchmark.py                 # defaults, ~2 min on a GPU
    python3 cost_benchmark.py --repeats 20 --probe-batches 4
Writes cost_benchmark.csv.
"""

import argparse
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_LAYERS = {"block3": 3, "block7": 7, "block11": 11}
PARAM_FILTERS = ("heads", "encoder.layers.encoder_layer_10",
                 "encoder.layers.encoder_layer_11",
                 "encoder.pos_embedding", "class_token")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeats * 1000.0   # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--probe-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=3,
                    help="number of stored block snapshots to compare against")
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--out", default="cost_benchmark.csv")
    args = ap.parse_args()

    print(f"device: {DEVICE}, repeats: {args.repeats}")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features,
                                 args.num_classes)
    model = model.to(DEVICE).eval()
    criterion = nn.CrossEntropyLoss()

    params = [p for n, p in model.named_parameters()
              if p.requires_grad and any(f in n for f in PARAM_FILTERS)]

    # synthetic data of the right shape — cost is shape-dependent, not
    # content-dependent, so random tensors give the same timings
    def batches(n):
        return [(torch.randn(args.batch_size, 3, 224, 224, device=DEVICE),
                 torch.randint(0, args.num_classes, (args.batch_size,),
                               device=DEVICE))
                for _ in range(n)]

    probe = batches(args.probe_batches)
    buffer = batches(args.probe_batches)
    anchors = torch.randn(args.anchors, 3, 224, 224, device=DEVICE)

    acts = {}

    def hook(name):
        def f(_m, _i, out):
            acts[name] = out[:, 0, :].detach()
        return f

    handles = [model.encoder.layers[i].register_forward_hook(hook(n))
               for n, i in PROBE_LAYERS.items()]
    handles.append(model.encoder.ln.register_forward_hook(hook("final")))

    # ── primitive operations ────────────────────────────────────────
    def fwd_probe():
        with torch.no_grad():
            for x, _ in probe:
                model(x)

    def fwd_probe_entropy():
        with torch.no_grad():
            for x, _ in probe:
                p = F.softmax(model(x), dim=1)
                (-p * (p + 1e-12).log()).sum(1).mean()

    def grad(bs):
        for p in params:
            p.grad = None
        for x, y in bs:
            criterion(model(x), y).backward()
        g = torch.cat([(p.grad if p.grad is not None
                        else torch.zeros_like(p)).flatten()
                       for p in params]).detach()
        for p in params:
            p.grad = None
        return g

    def grad_new():
        grad(probe)

    def grad_old():
        grad(buffer)

    def grad_both():
        a = grad(probe)
        b = grad(buffer)
        torch.dot(a, b)

    def fwd_buffer():
        with torch.no_grad():
            for x, y in buffer:
                out = model(x)
                criterion(out, y)
                (out.argmax(1) == y).sum()

    def weight_arith():
        w = torch.cat([p.detach().flatten() for p in params]).cpu()
        (w - w).norm()

    def anchor_forward():
        with torch.no_grad():
            for _ in range(args.blocks):
                for s in range(0, len(anchors), args.batch_size):
                    acts.clear()
                    model(anchors[s:s + args.batch_size])

    prim = {
        "fwd_probe": timed(fwd_probe, args.repeats),
        "fwd_probe_entropy": timed(fwd_probe_entropy, args.repeats),
        "grad_new": timed(grad_new, args.repeats),
        "grad_old": timed(grad_old, args.repeats),
        "grad_both": timed(grad_both, args.repeats),
        "fwd_buffer": timed(fwd_buffer, args.repeats),
        "weight_arith": timed(weight_arith, args.repeats),
        "anchor_forward": timed(anchor_forward, args.repeats),
    }
    for h in handles:
        h.remove()

    # ── attribute primitives to signals ─────────────────────────────
    marginal = {
        "new_loss":          prim["fwd_probe"],
        "new_entropy":       prim["fwd_probe_entropy"],
        "grad_norm_new":     prim["grad_new"],
        "grad_norm_old":     prim["grad_old"],
        "grad_cos":          prim["grad_both"],
        "grad_inner":        prim["grad_both"],
        "pred_dL_old":       prim["grad_both"],
        "ref_loss":          prim["fwd_buffer"],
        "ref_acc":           prim["fwd_buffer"],
        "weight_div_round":  prim["weight_arith"],
        "weight_div_task":   prim["weight_arith"],
        "proto_drift":       prim["anchor_forward"],
        "centroid_drift":    prim["anchor_forward"],
        "cka_block3":        prim["anchor_forward"],
        "cka_block7":        prim["anchor_forward"],
        "cka_block11":       prim["anchor_forward"],
        "cka_final":         prim["anchor_forward"],
    }

    # incremental cost on top of a monitor already computing
    # weight_div_round (the cheapest full detector in both conditions)
    base = marginal["weight_div_round"]
    incremental = {k: max(v - base, 0.0) for k, v in marginal.items()}

    rows = sorted(marginal.items(), key=lambda kv: kv[1])
    print(f"\n{'signal':<20}{'marginal (ms)':>15}{'incremental (ms)':>19}")
    print("-" * 54)
    for name, ms in rows:
        print(f"{name:<20}{ms:>15.1f}{incremental[name]:>19.1f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "marginal_ms", "incremental_ms"])
        for name, ms in rows:
            w.writerow([name, f"{ms:.3f}", f"{incremental[name]:.3f}"])
    print(f"\nwritten to {args.out}")
    print("Marginal = cost of computing this signal alone. Incremental = "
          "cost of adding it to a monitor already computing the per-round "
          "weight divergence.")


if __name__ == "__main__":
    main()
