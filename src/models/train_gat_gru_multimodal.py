
from pathlib import Path
import random

import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from src.models.temporal_multimodal_dataset import (
    TemporalMultimodalDataset,
)
# pyrefly: ignore [missing-import]
from src.models.gat_gru_multimodal import MultimodalGATGRU


# --------------------------------------------------
# Configuration: keep consistent with baseline
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 7
SEQUENCE_LENGTH = 6

CHECKPOINT_PATH = Path(
    "results/models/gat_gru_multimodal_best.pt"
)
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")

print("=" * 68)
print("MULTIMODAL GAT + GRU TRAINING")
print("=" * 68)
print("Device:", device)
print("Input features per node: 26")
print("Sequence length:", SEQUENCE_LENGTH)


# --------------------------------------------------
# Datasets and loaders
# --------------------------------------------------
train_dataset = TemporalMultimodalDataset(
    split="train",
    sequence_length=SEQUENCE_LENGTH,
)
dev_dataset = TemporalMultimodalDataset(
    split="dev",
    sequence_length=SEQUENCE_LENGTH,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

if len(train_dataset) == 0 or len(dev_dataset) == 0:
    raise RuntimeError("Training or development dataset is empty.")


# --------------------------------------------------
# Calculate class weights from TRAIN labels only
# --------------------------------------------------
train_labels = np.array([
    int(sequence.iloc[-1]["label_graph"])
    for sequence in train_dataset.sequences
])

class_counts = np.bincount(train_labels, minlength=2)

if np.any(class_counts == 0):
    raise ValueError(
        f"Training set must contain both classes: {class_counts}"
    )

class_weights = len(train_labels) / (2.0 * class_counts)
class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device,
)

print("\nTraining class counts:")
print("Interictal (0):", int(class_counts[0]))
print("Preictal   (1):", int(class_counts[1]))
print("Class weights:", class_weights.tolist())


# --------------------------------------------------
# Model, optimizer, loss
# --------------------------------------------------
model = MultimodalGATGRU(input_dim=26).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# All graphs use the same topology.
adjacency = np.load("data/graph/adjacency.npy")
edge_index_np = np.vstack(np.where(adjacency > 0)).astype(np.int64)
edge_index = torch.tensor(
    edge_index_np,
    dtype=torch.long,
    device=device,
)


# --------------------------------------------------
# Evaluation on DEV only
# --------------------------------------------------
def evaluate(loader):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, batch_edges, y in loader:
            x = x.to(device)
            y = y.to(device)

            # DataLoader stacks each sample's identical edge index.
            # The model accepts one shared edge index.
            batch_edges = batch_edges.to(device)
            if not torch.all(batch_edges == batch_edges[0]):
                raise ValueError("Graph topologies differ within a batch.")

            logits = model(x, batch_edges[0])
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_correct += (
                (logits.argmax(dim=1) == y).sum().item()
            )
            total_samples += y.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


# --------------------------------------------------
# Training with early stopping on DEV loss
# --------------------------------------------------
best_dev_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, batch_edges, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        batch_edges = batch_edges.to(device)
        if not torch.all(batch_edges == batch_edges[0]):
            raise ValueError("Graph topologies differ within a batch.")

        optimizer.zero_grad()

        logits = model(x, batch_edges[0])
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (
            (logits.argmax(dim=1) == y).sum().item()
        )
        total_samples += y.size(0)

    train_loss = total_loss / total_samples
    train_accuracy = total_correct / total_samples

    dev_loss, dev_accuracy = evaluate(dev_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Dev Loss: {dev_loss:.4f} | "
        f"Dev Acc: {dev_accuracy:.4f}"
    )

    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_loss": best_dev_loss,
                "class_weights": class_weights.cpu(),
                "config": {
                    "input_dim": 26,
                    "sequence_length": SEQUENCE_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                },
            },
            CHECKPOINT_PATH,
        )

        print("  Saved new best checkpoint.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


print("\n" + "=" * 68)
print("TRAINING COMPLETE")
print("=" * 68)
print("Best epoch:", best_epoch)
print(f"Best dev loss: {best_dev_loss:.6f}")
print("Checkpoint:", CHECKPOINT_PATH)
print("Evaluation split was not used.")
