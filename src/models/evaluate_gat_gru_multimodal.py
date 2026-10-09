


import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.models.temporal_multimodal_dataset import (
    TemporalMultimodalDataset,
)
# pyrefly: ignore [missing-import]
from src.models.gat_gru_multimodal import MultimodalGATGRU


DEVICE = torch.device("cpu")
CHECKPOINT = "results/models/gat_gru_multimodal_best.pt"
BATCH_SIZE = 32

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model = MultimodalGATGRU(input_dim=26).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

adjacency = np.load("data/graph/adjacency.npy")
edge_index = torch.tensor(
    np.vstack(np.where(adjacency > 0)),
    dtype=torch.long,
    device=DEVICE,
)


def evaluate_split(split):
    dataset = TemporalMultimodalDataset(
        split=split,
        sequence_length=6,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for x, batch_edges, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Every graph has the same topology.
            batch_edges = batch_edges.to(DEVICE)
            if not torch.all(batch_edges == batch_edges[0]):
                raise ValueError("Inconsistent graph topology.")

            logits = model(x, edge_index)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    print(f"\n{'=' * 60}")
    print(f"{split.upper()} RESULTS — BEST EPOCH {checkpoint['epoch']}")
    print("=" * 60)
    print("Samples:", len(y_true))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.4f}"
    )

    if len(np.unique(y_true)) == 2:
        print(f"ROC-AUC:           {roc_auc_score(y_true, y_score):.4f}")
        print(
            f"PR-AUC (average precision): "
            f"{average_precision_score(y_true, y_score):.4f}"
        )
    else:
        print("ROC-AUC / PR-AUC unavailable: only one class present.")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["interictal", "preictal"],
            digits=4,
            zero_division=0,
        )
    )


# Train metrics are diagnostic; DEV is used for model comparison.
evaluate_split("train")
evaluate_split("dev")

# Deliberately do not evaluate the held-out eval split here.
print("\nHeld-out evaluation split remains untouched.")
