# pyrefly: ignore [missing-import]
import os
import random
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

# pyrefly: ignore [missing-import]
from temporal_graph_dataset import TemporalGraphDataset
# pyrefly: ignore [missing-import]
from gat_gru import EEGGATGRU


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 50
PATIENCE = 7

GRAPH_ROOT = "data/graph"
ADJACENCY_PATH = os.path.join(GRAPH_ROOT, "adjacency.npy")

CHECKPOINT_PATH = "results/models/gat_gru_best.pt"

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASETS
# ============================================================

print("=" * 60)
print("GAT + GRU TRAINING")
print("=" * 60)

train_dataset = TemporalGraphDataset(
    split="train",
    sequence_length=6
)

dev_dataset = TemporalGraphDataset(
    split="dev",
    sequence_length=6
)

print("\nDataset sizes:")
print("Train sequences:", len(train_dataset))
print("Dev sequences  :", len(dev_dataset))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_labels = np.array([
    train_dataset[i][1]
    for i in range(len(train_dataset))
])

negative_count = np.sum(train_labels == 0)
positive_count = np.sum(train_labels == 1)

print("\nTraining class distribution:")
print("Negative:", negative_count)
print("Positive:", positive_count)

# Same class-weighting strategy as GAT
total = negative_count + positive_count

class_weights = torch.tensor(
    [
        total / (2.0 * negative_count),
        total / (2.0 * positive_count)
    ],
    dtype=torch.float32,
    device=DEVICE
)

print("Class weights:", class_weights)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# GRAPH TOPOLOGY
# ============================================================

adjacency = np.load(ADJACENCY_PATH)

edge_index = np.stack(
    np.where(adjacency > 0),
    axis=0
)

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long,
    device=DEVICE
)

print("\nGraph topology:")
print("Nodes:", adjacency.shape[0])
print("Edges:", edge_index.shape[1])


# ============================================================
# MODEL
# ============================================================

model = EEGGATGRU(
    input_dim=16,
    gat_hidden=32,
    gat_output=64,
    gru_hidden=64,
    num_classes=2,
    dropout=0.3
).to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

best_dev_loss = float("inf")
epochs_without_improvement = 0

os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True
)


for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:

        x = x.float().to(DEVICE)
        y = y.long().to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            x,
            edge_index
        )

        loss = criterion(
            logits,
            y
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_correct += (
            predictions == y
        ).sum().item()

        train_total += y.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # --------------------------------------------------------
    # DEV
    # --------------------------------------------------------

    model.eval()

    dev_loss = 0.0
    dev_correct = 0
    dev_total = 0

    with torch.no_grad():

        for x, y in dev_loader:

            x = x.float().to(DEVICE)
            y = y.long().to(DEVICE)

            logits = model(
                x,
                edge_index
            )

            loss = criterion(
                logits,
                y
            )

            dev_loss += loss.item() * y.size(0)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            dev_correct += (
                predictions == y
            ).sum().item()

            dev_total += y.size(0)

    dev_loss /= dev_total
    dev_accuracy = dev_correct / dev_total


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Dev Loss: {dev_loss:.4f} | "
        f"Dev Acc: {dev_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if dev_loss < best_dev_loss:

        best_dev_loss = dev_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": dev_loss
            },
            CHECKPOINT_PATH
        )

        print(
            f"  -> Saved best model "
            f"(dev loss = {dev_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best dev loss:", best_dev_loss)
print("Checkpoint:", CHECKPOINT_PATH)