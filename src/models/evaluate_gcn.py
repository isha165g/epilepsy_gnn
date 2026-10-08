import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, "src")

from models.graph_dataset import EEGGraphDataset
# pyrefly: ignore [missing-import]
from models.gcn import EEGGCN


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")

GRAPH_ROOT = "data/graph"
CHECKPOINT = "results/models/gcn_best.pt"

BATCH_SIZE = 32


# ============================================================
# Load model
# ============================================================

model = EEGGCN(
    in_channels=16,
    hidden_channels=32,
    out_channels=64,
    num_classes=2,
    dropout=0.3
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 60)
print("GCN EVALUATION")
print("=" * 60)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Checkpoint dev loss: {checkpoint['dev_loss']:.4f}")
print(f"Checkpoint dev accuracy: {checkpoint['dev_accuracy']:.4f}")


# ============================================================
# Evaluation function
# ============================================================

@torch.no_grad()
def evaluate(split):

    dataset = EEGGraphDataset(
        root=GRAPH_ROOT,
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in loader:

        batch = batch.to(DEVICE)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        predictions = logits.argmax(dim=1)

        all_labels.extend(
            batch.y.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

    y_true = torch.tensor(all_labels).numpy()
    y_pred = torch.tensor(all_predictions).numpy()
    y_prob = torch.tensor(all_probabilities).numpy()

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    print("\n" + "-" * 60)
    print(f"{split.upper()} SET")
    print("-" * 60)

    print(f"Samples:          {len(y_true)}")
    print(f"Actual negative:  {(y_true == 0).sum()}")
    print(f"Actual positive:  {(y_true == 1).sum()}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nBalanced accuracy:")
    print(f"{balanced_acc:.4f}")

    print("\nROC-AUC:")
    print(f"{roc_auc:.4f}")

    print("\nPR-AUC:")
    print(f"{pr_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Interictal",
                "Preictal"
            ],
            digits=4
        )
    )


# ============================================================
# Evaluate train and dev
# ============================================================

evaluate("train")
evaluate("dev")