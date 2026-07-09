"""
Flower federated learning client with a Vision Transformer (ViT).

Run (after starting server.py):
    python client.py --cid 0
    python client.py --cid 1
    ...one process per client.
"""

import argparse
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import cifardata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10          # <-- set to the number of classes in YOUR dataset
BATCH_SIZE = 32
LOCAL_EPOCHS = 1
LEARNING_RATE = 3e-4
CLIENTS = 3 # TODO: Make this arguable


# ----------------------------------------------------------------------------
# 1. MODEL: Vision Transformer
# ----------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ViT-B/16, pretrained on ImageNet, with a new classification head."""
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # Replace the head for your number of classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(DEVICE)


# ----------------------------------------------------------------------------
# 2. DATASET  <<< PLUG YOUR DATA IN HERE >>>
# ----------------------------------------------------------------------------
# ViT-B/16 expects 224x224 RGB images normalized with ImageNet stats.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MyDataset(Dataset):
    """
    <<< REPLACE THIS with a Dataset that reads YOUR files. >>>

    It must return (image_tensor, label) pairs, where image_tensor is a
    3x224x224 float tensor (use TRANSFORM above) and label is an int.

    Common options instead of writing your own class:
      - Folder of images sorted by class:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder("path/to/client_data/train", transform=TRANSFORM)
      - CSV/parquet with file paths + labels: read it in __init__,
        load and transform the image in __getitem__.
    """

    def __init__(self, data_dir: str, train: bool = True):
        raise NotImplementedError("Plug in your own data loading here")

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...


def load_data(cid: int):
    """
    <<< PLUG IN: return this client's train and validation DataLoaders. >>>

    In federated learning each client should load ONLY its own partition,
    e.g. a per-client directory like data/client_{cid}/train and .../val.
    """
    #train_set = MyDataset(f"data/client_{cid}/train", train=True)   # <-- your path
    #val_set = MyDataset(f"data/client_{cid}/val", train=False)      # <-- your path

    #train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    #val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    train_loader, val_loader = cifardata.load_partition(cid, CLIENTS)

    return train_loader, val_loader


# ----------------------------------------------------------------------------
# 3. TRAIN / EVALUATE
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# 4. FLOWER CLIENT
# ----------------------------------------------------------------------------
class ViTClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.model = build_model()
        self.train_loader, self.val_loader = load_data(cid)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = self.model.state_dict().keys()
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=LOCAL_EPOCHS)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.val_loader)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    fl.client.start_client(
        server_address=args.server,
        client=ViTClient(args.cid).to_client(),
    )
