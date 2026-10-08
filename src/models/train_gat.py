import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch_geometric.loader import DataLoader

sys.path.insert(0, "src")

from models.graph_dataset import EEGGraphDataset
# pyrefly: ignore [missing-import]
from models.gat import EEGGAT


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 50
PATIENCE = 7

DEVICE = torch.device("cpu")

GRAPH_ROOT = "data/graph"

CHECKPOINT_DIR = Path("results/models")
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "gat_best.pt"
)


# ============================================================
# Dataset
# ============================================================

train_dataset = EEGGraphDataset(
    root=GRAPH_ROOT,
    split="train"
)

dev_dataset = EEGGraphDataset(
    root=GRAPH_ROOT,
    split="dev"
)

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


print("=" * 60)
print("GAT DATASET")
print("=" * 60)

print(
    f"Train graphs: {len(train_dataset)}"
)

print(
    f"Dev graphs:   {len(dev_dataset)}"
)

print(
    f"Batch size:   {BATCH_SIZE}"
)

print(
    f"Device:       {DEVICE}"
)


# ============================================================
# Class weights
# ============================================================

train_labels = []

for i in range(len(train_dataset)):
    train_labels.append(
        train_dataset[i].y.item()
    )

train_labels = torch.tensor(
    train_labels
)

num_negative = (
    train_labels == 0
).sum().item()

num_positive = (
    train_labels == 1
).sum().item()

total = len(train_labels)
num_classes = 2

class_weights = torch.tensor(
    [
        total / (
            num_classes * num_negative
        ),
        total / (
            num_classes * num_positive
        )
    ],
    dtype=torch.float32,
    device=DEVICE
)


print("\nClass distribution:")
print(
    f"Negative (0): {num_negative}"
)

print(
    f"Positive (1): {num_positive}"
)

print("\nClass weights:")
print(
    f"Weight for class 0: "
    f"{class_weights[0]:.4f}"
)

print(
    f"Weight for class 1: "
    f"{class_weights[1]:.4f}"
)


# ============================================================
# Model
# ============================================================

model = EEGGAT(
    in_channels=16,
    hidden_channels=32,
    out_channels=64,
    num_classes=2,
    dropout=0.3,
    heads=1
).to(DEVICE)


# ============================================================
# Loss + optimizer
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
# Training
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

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

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == batch.y
        ).sum().item()

        total += batch.num_graphs

    avg_loss = (
        total_loss / total
    )

    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

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

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == batch.y
        ).sum().item()

        total += batch.num_graphs

    avg_loss = (
        total_loss / total
    )

    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
# Training loop
# ============================================================

print("\n" + "=" * 60)
print("GAT TRAINING")
print("=" * 60)

best_dev_loss = float("inf")
epochs_without_improvement = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    train_loss, train_acc = (
        train_one_epoch()
    )

    dev_loss, dev_acc = (
        evaluate(dev_loader)
    )

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Dev Loss: {dev_loss:.4f} | "
        f"Dev Acc: {dev_acc:.4f}"
    )

    # --------------------------------------------------------
    # Save best model using dev loss
    # --------------------------------------------------------

    if dev_loss < best_dev_loss:

        best_dev_loss = dev_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "dev_loss": dev_loss,
                "dev_accuracy": dev_acc,
                "class_weights":
                    class_weights.cpu(),
            },
            BEST_MODEL_PATH
        )

        print(
            f"  -> Saved best model "
            f"(dev loss = "
            f"{dev_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping after "
                f"{epoch} epochs."
            )

            break


print("\n" + "=" * 60)
print("GAT TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best dev loss: "
    f"{best_dev_loss:.4f}"
)

print(
    f"Best model: "
    f"{BEST_MODEL_PATH}"
)