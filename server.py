"""
Flower federated learning server for the Vision Transformer.

Run:
    python server.py
then start clients with client.py.
"""

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters

# Reuse the model definition so server and clients stay in sync
from client import build_model

NUM_ROUNDS = 5
MIN_CLIENTS = 2          # minimum clients needed per round
FRACTION_FIT = 1.0       # fraction of available clients sampled each round


def weighted_average(metrics):
    """Aggregate client 'accuracy' metrics weighted by number of examples."""
    total = sum(num_examples for num_examples, _ in metrics)
    acc = sum(num_examples * m["accuracy"] for num_examples, m in metrics)
    return {"accuracy": acc / total}


def get_initial_parameters():
    """Initialize global weights from the (pretrained) ViT so all clients
    start from the same point."""
    model = build_model()
    ndarrays = [v.cpu().numpy() for v in model.state_dict().values()]
    return ndarrays_to_parameters(ndarrays)


# ----------------------------------------------------------------------------
# OPTIONAL: centralized (server-side) evaluation
# <<< PLUG IN a held-out test set here if you have one on the server. >>>
# ----------------------------------------------------------------------------
# def get_evaluate_fn():
#     from client import MyDataset, evaluate  # your dataset class
#     from torch.utils.data import DataLoader
#
#     test_set = MyDataset("data/server_test", train=False)   # <-- your path
#     test_loader = DataLoader(test_set, batch_size=32)
#     model = build_model()
#
#     def evaluate_fn(server_round, parameters, config):
#         from collections import OrderedDict
#         keys = model.state_dict().keys()
#         model.load_state_dict(
#             OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
#         )
#         loss, accuracy = evaluate(model, test_loader)
#         return loss, {"accuracy": accuracy}
#
#     return evaluate_fn


def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        initial_parameters=get_initial_parameters(),
        evaluate_metrics_aggregation_fn=weighted_average,
        # evaluate_fn=get_evaluate_fn(),   # uncomment for server-side evaluation
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
