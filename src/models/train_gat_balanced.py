import os
from graph_dataset import np
from numpy import random

import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch_geometric.loader import DataLoader
# pyrefly: ignore [missing-import]
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, "src")

from models.graph_dataset import EEGGraphDataset
# pyrefly: ignore [missing-import]
from models.gat import EEGGAT


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 50
PATIENCE = 7

GRAPH_ROOT = "data/graph"
MODEL_PATH = "results/models/gat_balanced_best.pt"

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

train_dataset = EEGGraphDataset(
    GRAPH_ROOT,
    "train"
)

dev_dataset = EEGGraphDataset(
    GRAPH_ROOT,
    "dev"
)


print("=" * 60)
print("BALANCED-SAMPLING GAT DATASET")
print("=" * 60)

print(f"Train graphs: {len(train_dataset)}")
print(f"Dev graphs:   {len(dev_dataset)}")
print(f"Batch size:   {BATCH_SIZE}")


# ============================================================
# GET TRAIN LABELS
# ============================================================

train_labels = torch.tensor(
    [
        int(train_dataset[i].y.item())
        for i in range(len(train_dataset))
    ],
    dtype=torch.long
)

class_counts = torch.bincount(
    train_labels,
    minlength=2
).float()

print("\nOriginal training distribution:")
print(f"Negative (0): {int(class_counts[0])}")
print(f"Positive (1): {int(class_counts[1])}")


# ============================================================
# BALANCED SAMPLER
# ============================================================

# Give each sample weight inversely proportional
# to the frequency of its class.
class_weights = 1.0 / class_counts

sample_weights = class_weights[train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

# IMPORTANT:
# Dev remains completely natural / untouched.
dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = EEGGAT().to(DEVICE)


# ============================================================
# LOSS
# ============================================================

# No class weights here.
# The balancing is handled ONLY by the sampler.
criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

best_dev_loss = float("inf")
patience_counter = 0

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

print("\n" + "=" * 60)
print("BALANCED-SAMPLING GAT TRAINING")
print("=" * 60)


for epoch in range(1, MAX_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(
            logits,
            batch.y
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

        predictions = logits.argmax(dim=1)

        train_correct += (
            predictions == batch.y
        ).sum().item()

        train_total += batch.num_graphs

    train_loss /= train_total
    train_acc = train_correct / train_total


    # --------------------------------------------------------
    # DEV
    # --------------------------------------------------------

    model.eval()

    dev_loss = 0.0
    dev_correct = 0
    dev_total = 0

    with torch.no_grad():

        for batch in dev_loader:

            batch = batch.to(DEVICE)

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = criterion(
                logits,
                batch.y
            )

            dev_loss += loss.item() * batch.num_graphs

            predictions = logits.argmax(dim=1)

            dev_correct += (
                predictions == batch.y
            ).sum().item()

            dev_total += batch.num_graphs

    dev_loss /= dev_total
    dev_acc = dev_correct / dev_total


    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Dev Loss: {dev_loss:.4f} | "
        f"Dev Acc: {dev_acc:.4f}"
    )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if dev_loss < best_dev_loss:

        best_dev_loss = dev_loss
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": dev_loss,
                "dev_accuracy": dev_acc,
            },
            MODEL_PATH
        )

        print(
            f"  -> Saved best model "
            f"(dev loss = {dev_loss:.4f})"
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping after {epoch} epochs."
            )
            break


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best dev loss: {best_dev_loss:.4f}")
print(f"Best model: {MODEL_PATH}")