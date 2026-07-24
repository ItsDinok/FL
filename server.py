"""
Flower server for heterogeneous Dirichlet class-distribution FL.

Startup prints each client's dominant class per block so you can verify
the schedule before committing GPU hours. Per-class accuracy metrics
(c{cid}_acc_{classname}) are aggregated and logged to the server history.
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters

from client import (build_model, ROUNDS_PER_BLOCK, NUM_BLOCKS,
                    NUM_CLIENTS, ALPHA)
from cifardata import CLASS_NAMES, NUM_CLASSES, get_block_distribution

NUM_ROUNDS  = ROUNDS_PER_BLOCK * NUM_BLOCKS   # 20
MIN_CLIENTS = NUM_CLIENTS


def fit_config(server_round: int):
    return {"server_round": server_round}


def evaluate_config(server_round: int):
    return {"server_round": server_round}


def weighted_average(metrics):
    """Average every metric key weighted by n_examples.
    Handles heterogeneous keys (c0_*, c1_*, ...) correctly — each key
    is averaged only over the clients that reported it.
    """
    sums, counts = {}, {}
    for n, m in metrics:
        for k, v in m.items():
            sums[k]   = sums.get(k, 0.0) + n * v
            counts[k] = counts.get(k, 0)  + n
    return {k: sums[k] / counts[k] for k in sorted(sums)}


def get_initial_parameters():
    model = build_model()
    return ndarrays_to_parameters(
        [v.cpu().numpy() for v in model.state_dict().values()])


def main():
    print(f"Dirichlet class-distribution experiment  "
          f"alpha={ALPHA}  {NUM_CLIENTS} clients  {NUM_ROUNDS} rounds")
    print(f"\nDominant class per block (alpha={ALPHA}):")
    for cid in range(NUM_CLIENTS):
        row = []
        for b in range(NUM_BLOCKS):
            p = get_block_distribution(cid, ALPHA, b)
            row.append(f"b{b}:{CLASS_NAMES[int(p.argmax())]}({p.max():.2f})")
        print(f"  client {cid}: " + "  ->  ".join(row))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit         = 1.0,
        fraction_evaluate    = 1.0,
        min_fit_clients      = MIN_CLIENTS,
        min_evaluate_clients = MIN_CLIENTS,
        min_available_clients= MIN_CLIENTS,
        initial_parameters   = get_initial_parameters(),
        on_fit_config_fn     = fit_config,
        on_evaluate_config_fn= evaluate_config,
        evaluate_metrics_aggregation_fn = weighted_average,
        fit_metrics_aggregation_fn      = weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
