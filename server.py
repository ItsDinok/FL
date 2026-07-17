"""
Flower server for the high-forgetting DIL baseline.

NOTE: if you run clients with a nonzero --drift-lag, set DRIFT_LAG below
to the same value so the schedule is extended to let lagging clients
finish their final domain block.
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters

from client import (build_model, round_to_domain, ROUNDS_PER_DOMAIN,
                    NUM_CLIENTS, DRIFT_LAG)
from cifardata import NUM_DOMAINS, DOMAIN_NAMES

# Base blocks + extra rounds so the most-lagged client completes its sequence
NUM_ROUNDS = ROUNDS_PER_DOMAIN * NUM_DOMAINS + DRIFT_LAG * (NUM_CLIENTS - 1)
MIN_CLIENTS = NUM_CLIENTS


def fit_config(server_round: int):
    return {"server_round": server_round}


def evaluate_config(server_round: int):
    return {"server_round": server_round}


def heterogeneous_weighted_average(metrics):
    """Weighted average per key, over only the clients that reported it."""
    sums, counts = {}, {}
    for n, m in metrics:
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + n * v
            counts[k] = counts.get(k, 0) + n
    return {k: sums[k] / counts[k] for k in sorted(sums)}


def get_initial_parameters():
    model = build_model()
    return ndarrays_to_parameters(
        [v.cpu().numpy() for v in model.state_dict().values()]
    )


def main():
    print(f"Drift lag: {DRIFT_LAG} "
          f"({'fully correlated - max forgetting' if DRIFT_LAG == 0 else 'staggered'})")
    print("Per-client domain by round:")
    for r in range(1, NUM_ROUNDS + 1):
        domains = [DOMAIN_NAMES[round_to_domain(r, c, DRIFT_LAG)]
                   for c in range(NUM_CLIENTS)]
        print(f"  round {r:2d}: " + " | ".join(
            f"c{c}={d}" for c, d in enumerate(domains)))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=get_initial_parameters(),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=heterogeneous_weighted_average,
        fit_metrics_aggregation_fn=heterogeneous_weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
