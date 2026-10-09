from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.models.temporal_ablation_dataset import TemporalAblationDataset
from src.models.gat_gru_ablation import GATGRUAblation


DEVICE = torch.device("cpu")
BATCH_SIZE = 32

FEATURE_MODES = {
    "handcrafted": 16,
    "time_frequency": 10,
    "multimodal": 26,
}

MODEL_DIR = Path("results/ablation/models")
OUTPUT_DIR = Path("results/ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(feature_mode, input_dim, edge_index):
    checkpoint_path = MODEL_DIR / f"{feature_mode}_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    dataset = TemporalAblationDataset(
        split="dev",
        feature_mode=feature_mode,
        sequence_length=6,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = GATGRUAblation(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for x, batch_edges, y in loader:
            if not torch.all(batch_edges == batch_edges[0]):
                raise ValueError("Different graph topologies within batch.")

            logits = model(x.to(DEVICE), edge_index)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            y_true_all.extend(y.numpy().tolist())
            y_prob_all.extend(probabilities.cpu().numpy().tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_prob = np.asarray(y_prob_all, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")

    results = {
        "feature_mode": feature_mode,
        "input_features_per_node": input_dim,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_dev_loss": float(checkpoint["best_dev_loss"]),
        "n_dev": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_count": int(y_pred.sum()),
        "predicted_negative_count": int((y_pred == 0).sum()),
    }

    print("\n" + "=" * 72)
    print(f"DEV EVALUATION: {feature_mode.upper()}")
    print("=" * 72)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    print("\nClassification report (threshold = 0.5):")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["interictal/negative", "preictal/positive"],
        zero_division=0,
    ))

    return results


def main():
    adjacency = np.load("data/graph/adjacency.npy")
    edge_np = np.vstack(np.where(adjacency > 0)).astype(np.int64)
    edge_index = torch.tensor(edge_np, dtype=torch.long, device=DEVICE)

    results = []
    for feature_mode, input_dim in FEATURE_MODES.items():
        results.append(
            evaluate_model(feature_mode, input_dim, edge_index)
        )

    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "dev_ablation_metrics.csv"
    json_path = OUTPUT_DIR / "dev_ablation_metrics.json"

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("ABLATION COMPARISON — DEVELOPMENT SET ONLY")
    print("=" * 72)
    columns = [
        "feature_mode",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
    print(results_df[columns].to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print("Held-out evaluation split was not used.")


if __name__ == "__main__":
    main()
