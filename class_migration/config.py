"""
config.py — one place for everything an experiment varies.

Client and server both build their configuration from here, so they can
never disagree about client count, block schedule or log location. A run
is identified by dataset, scenario and an optional tag; logs land in
logs/<dataset>_<scenario>[_<tag>]/ so parallel ablations cannot overwrite
each other.

    python client.py --dataset pacs --scenario domain --cid 0
    python server.py --dataset pacs --scenario domain

The model registry lives here too, because the signal monitor's parameter
filters are architecture-specific: swapping the backbone means swapping
the filters with it.
"""

import argparse
import os

MODELS = {
    # name: (builder, param filters for gradient/weight-space signals,
    #        probe layer indices for layer-wise CKA)
    "vit_b_16": dict(
        filters=("heads", "encoder.layers.encoder_layer_10",
                 "encoder.layers.encoder_layer_11",
                 "encoder.pos_embedding", "class_token"),
        probe_layers={"block3": 3, "block7": 7, "block11": 11},
    ),
    "vit_b_16_scratch": dict(
        filters=("heads", "encoder.layers.encoder_layer_10",
                 "encoder.layers.encoder_layer_11",
                 "encoder.pos_embedding", "class_token"),
        probe_layers={"block3": 3, "block7": 7, "block11": 11},
    ),
}


def build_model(name, num_classes):
    """Returns a CPU model; the client moves it to GPU inside a slot."""
    import torch.nn as nn
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    weights = None if name.endswith("_scratch") else ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def add_arguments(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("experiment")
    g.add_argument("--dataset", default="cifar100")
    g.add_argument("--scenario", default="split",
                   choices=["split", "domain", "dirichlet"])
    g.add_argument("--tag", default=None,
                   help="suffix for the log directory, e.g. seed1337")
    g.add_argument("--model", default="vit_b_16", choices=sorted(MODELS))
    g.add_argument("--clients", type=int, default=3)
    g.add_argument("--blocks", type=int, default=4)
    g.add_argument("--rounds-per-block", type=int, default=5)
    g.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet concentration (dirichlet scenario only)")
    g.add_argument("--samples-per-client", type=int, default=4000)
    g.add_argument("--eval-samples", type=int, default=1000)

    t = ap.add_argument_group("training")
    t.add_argument("--local-epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--seed", type=int, default=42)

    m = ap.add_argument_group("monitoring")
    m.add_argument("--intra-round-eval", action="store_true",
                   help="probe retained accuracy during local training, "
                        "giving a step-level prediction horizon")
    m.add_argument("--probe-every-steps", type=int, default=50)
    m.add_argument("--probe-images", type=int, default=200)
    m.add_argument("--gpu-slots", type=int, default=2,
                   help="clients allowed on one GPU at a time")

    c = ap.add_argument_group("controller (phase-2 preview)")
    c.add_argument("--controller", action="store_true")
    c.add_argument("--gate-signal", default="new_loss")
    c.add_argument("--gate-threshold", type=float, default=1.0)
    c.add_argument("--gate-lr-scale", type=float, default=0.25)

    ap.add_argument("--port", type=int, default=8080)
    return ap


def logs_dir(args) -> str:
    name = f"{args.dataset}_{args.scenario}"
    if args.tag:
        name += f"_{args.tag}"
    return os.path.join("logs", name)


def make_scenario(args):
    from datasets import get_dataset
    from scenarios import build_scenario

    spec = get_dataset(args.dataset)
    kw = dict(num_blocks=args.blocks,
              samples_per_client=args.samples_per_client,
              eval_samples=args.eval_samples)
    if args.scenario == "dirichlet":
        kw["alpha"] = args.alpha
    return build_scenario(args.scenario, spec, **kw)


def summarise(args, scenario) -> str:
    rounds = args.rounds_per_block * scenario.num_blocks
    lines = [
        f"dataset       {args.dataset} ({scenario.num_classes} classes)",
        f"scenario      {args.scenario} (mode={scenario.unit_mode})",
        f"model         {args.model}",
        f"clients       {args.clients}",
        f"schedule      {scenario.num_blocks} blocks x "
        f"{args.rounds_per_block} rounds = {rounds} rounds",
        f"training      {args.local_epochs} epochs, lr={args.lr}, "
        f"batch={args.batch_size}",
        f"logs          {logs_dir(args)}",
    ]
    if args.intra_round_eval:
        lines.append(f"intra-round   every {args.probe_every_steps} steps, "
                     f"{args.probe_images} images/unit")
    if args.controller:
        lines.append(f"controller    {args.gate_signal} > "
                     f"{args.gate_threshold} -> lr x {args.gate_lr_scale}")
    lines.append("blocks        " + ", ".join(
        f"{b}:{scenario.block_name(b)}" for b in range(scenario.num_blocks)))
    return "\n".join("  " + l for l in lines)
