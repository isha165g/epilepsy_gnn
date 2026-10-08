# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)

# pyrefly: ignore [missing-import]
from temporal_graph_dataset import TemporalGraphDataset
# pyrefly: ignore [missing-import]
from gat_gru import EEGGATGRU


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32

GRAPH_ROOT = "data/graph"
MANIFEST_PATH = "data/metadata/graph_manifest.csv"

CHECKPOINT_PATH = "results/models/gat_gru_best.pt"

DEVICE = torch.device("cpu")


# ============================================================
# LOAD GRAPH TOPOLOGY
# ============================================================

adjacency = np.load(
    os.path.join(
        GRAPH_ROOT,
        "adjacency.npy"
    )
)

edge_index = np.stack(
    np.where(adjacency > 0),
    axis=0
)

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long,
    device=DEVICE
)


# ============================================================
# LOAD MODEL
# ============================================================

model = EEGGATGRU(
    input_dim=16,
    gat_hidden=32,
    gat_output=64,
    gru_hidden=64,
    num_classes=2,
    dropout=0.3
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 60)
print("GAT + GRU EVALUATION")
print("=" * 60)

print("\nLoaded checkpoint:")
print("Epoch:", checkpoint["epoch"])
print("Best dev loss:", checkpoint["dev_loss"])


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    dataset,
    split_name
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for x, y in loader:

            x = x.float().to(DEVICE)
            y = y.long().to(DEVICE)

            logits = model(
                x,
                edge_index
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    accuracy = accuracy_score(
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

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print(f"{split_name.upper()} SET")
    print("-" * 60)

    print("Samples:", len(y_true))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nOverall Metrics:")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")

    print("\nClass Metrics:")

    print("\nInterictal (0):")
    print(f"Precision : {precision[0]:.4f}")
    print(f"Recall    : {recall[0]:.4f}")
    print(f"F1        : {f1[0]:.4f}")
    print(f"Support   : {support[0]}")

    print("\nPreictal (1):")
    print(f"Precision : {precision[1]:.4f}")
    print(f"Recall    : {recall[1]:.4f}")
    print(f"F1        : {f1[1]:.4f}")
    print(f"Support   : {support[1]}")

    print("\nPrediction Distribution:")
    print(
        f"Predicted interictal : "
        f"{np.sum(y_pred == 0)}"
    )

    print(
        f"Predicted preictal   : "
        f"{np.sum(y_pred == 1)}"
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


# ============================================================
# TRAIN EVALUATION
# ============================================================

train_dataset = TemporalGraphDataset(
    split="train",
    sequence_length=6
)

train_results = evaluate(
    train_dataset,
    "Train"
)


# ============================================================
# DEV EVALUATION
# ============================================================

dev_dataset = TemporalGraphDataset(
    split="dev",
    sequence_length=6
)

dev_results = evaluate(
    dev_dataset,
    "Dev"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("GAT + GRU SUMMARY")
print("=" * 60)

print(
    f"\n{'Metric':<20}"
    f"{'Train':>12}"
    f"{'Dev':>12}"
)

print("-" * 44)

print(
    f"{'Accuracy':<20}"
    f"{train_results['accuracy']:>12.4f}"
    f"{dev_results['accuracy']:>12.4f}"
)

print(
    f"{'Balanced Accuracy':<20}"
    f"{train_results['balanced_accuracy']:>12.4f}"
    f"{dev_results['balanced_accuracy']:>12.4f}"
)

print(
    f"{'ROC-AUC':<20}"
    f"{train_results['roc_auc']:>12.4f}"
    f"{dev_results['roc_auc']:>12.4f}"
)

print(
    f"{'PR-AUC':<20}"
    f"{train_results['pr_auc']:>12.4f}"
    f"{dev_results['pr_auc']:>12.4f}"
)

print("\nEvaluation complete.")
print("EVAL split was NOT used.")