import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

sys.path.insert(0, "src")

from models.graph_dataset import EEGGraphDataset


GRAPH_ROOT = "data/graph"


def load_split(split):
    """
    Load graph node features and flatten each graph.

    Original:
        X = (20 nodes, 16 features)

    Flattened:
        X = (320 features)
    """

    dataset = EEGGraphDataset(
        root=GRAPH_ROOT,
        split=split
    )

    X = []
    y = []

    for i in range(len(dataset)):

        graph = dataset[i]

        # 20 × 16 -> 320
        features = graph.x.numpy().reshape(-1)

        X.append(features)
        y.append(graph.y.item())

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X, y


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("FLAT 320-FEATURE BASELINE")
print("=" * 60)

X_train, y_train = load_split("train")
X_dev, y_dev = load_split("dev")

print("\nData shapes:")
print("Train X:", X_train.shape)
print("Train y:", y_train.shape)
print("Dev X:  ", X_dev.shape)
print("Dev y:  ", y_dev.shape)

print("\nClass distribution:")
print(
    f"Train -> negative: {(y_train == 0).sum()}, "
    f"positive: {(y_train == 1).sum()}"
)

print(
    f"Dev   -> negative: {(y_dev == 0).sum()}, "
    f"positive: {(y_dev == 1).sum()}"
)


# ============================================================
# Logistic Regression
# ============================================================

print("\nTraining logistic regression...")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(X, y, split):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(
        y,
        predictions
    )

    balanced_acc = balanced_accuracy_score(
        y,
        predictions
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    print("\n" + "-" * 60)
    print(f"{split.upper()} SET")
    print("-" * 60)

    print(f"Samples: {len(y)}")

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
            y,
            predictions,
            target_names=[
                "Interictal",
                "Preictal"
            ],
            digits=4
        )
    )


evaluate(
    X_train,
    y_train,
    "train"
)

evaluate(
    X_dev,
    y_dev,
    "dev"
)