"""
server.py — generic federated continual-learning server.

Reads the same configuration as the client, so the two cannot disagree
about client count or block schedule.

    python server.py --dataset cifar100 --scenario split
    python server.py --dataset pacs --scenario domain --port 8081

Broadcasts the round number (which drives every client's block schedule)
and aggregates whatever per-client metrics arrive, including the
per-client signal values used for cross-checking.
"""

import argparse

import flwr as fl
from flwr.common import ndarrays_to_parameters

import config


def weighted_average(metrics):
    """Per-key weighted mean over the clients that reported that key, so
    per-client keys (c0_*, c1_*) pass through untouched while shared keys
    ('accuracy') average across the fleet."""
    sums, counts = {}, {}
    for n, m in metrics:
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + n * v
            counts[k] = counts.get(k, 0) + n
    return {k: sums[k] / counts[k] for k in sorted(sums)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_arguments(ap)
    args = ap.parse_args()

    sc = config.make_scenario(args)
    rounds = args.rounds_per_block * sc.num_blocks

    print("Federated continual-learning benchmark")
    print(config.summarise(args, sc))
    print(f"\nScore afterwards with:\n  python horizon.py --logs "
          f"{config.logs_dir(args)} --mode {sc.unit_mode} --sweep\n",
          flush=True)

    model = config.build_model(args.model, sc.num_classes)
    initial = ndarrays_to_parameters(
        [v.cpu().numpy() for v in model.state_dict().values()])

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=initial,
        on_fit_config_fn=lambda r: {"server_round": r},
        on_evaluate_config_fn=lambda r: {"server_round": r},
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
