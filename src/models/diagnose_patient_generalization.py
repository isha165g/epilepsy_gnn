from pathlib import Path

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
)

from src.models.temporal_ablation_dataset import TemporalAblationDataset
from src.models.gat_gru_ablation import GATGRUAblation


DEVICE = torch.device("cpu")
BATCH_SIZE = 32
SEQUENCE_LENGTH = 6

FEATURE_MODES = {
    "handcrafted": 16,
    "time_frequency": 10,
    "multimodal": 26,
}

MODEL_DIR = Path("results/ablation/models")
OUTPUT_DIR = Path("results/ablation/patient_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def metrics_for_group(group):
    y = group["label"].to_numpy(dtype=int)
    p = group["probability"].to_numpy(dtype=float)
    pred = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    result = {
        "n_sequences": len(y),
        "negative_count": int((y == 0).sum()),
        "positive_count": int((y == 1).sum()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # ROC-AUC/AP are undefined when a group contains only one class.
    if len(np.unique(y)) == 2:
        result["roc_auc"] = roc_auc_score(y, p)
        result["average_precision"] = average_precision_score(y, p)
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan

    return result


def main():
    adjacency = np.load("data/graph/adjacency.npy")
    edge_index = torch.tensor(
        np.vstack(np.where(adjacency > 0)).astype(np.int64),
        dtype=torch.long,
        device=DEVICE,
    )

    patient_rows = []
    recording_rows = []
    pooled_rows = []

    for mode, input_dim in FEATURE_MODES.items():
        checkpoint_path = MODEL_DIR / f"{mode}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        dataset = TemporalAblationDataset(
            split="dev",
            feature_mode=mode,
            sequence_length=SEQUENCE_LENGTH,
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

        labels, probabilities = [], []

        with torch.no_grad():
            for x, batch_edges, y in loader:
                if not torch.all(batch_edges == batch_edges[0]):
                    raise ValueError("Different graph topologies in batch")

                logits = model(x.to(DEVICE), edge_index)
                prob = torch.softmax(logits, dim=1)[:, 1]

                labels.extend(y.cpu().numpy().tolist())
                probabilities.extend(prob.cpu().numpy().tolist())

        if len(dataset.sequences) != len(labels):
            raise RuntimeError("Prediction/metadata length mismatch")

        metadata_rows = []
        for seq, label, probability in zip(
            dataset.sequences, labels, probabilities
        ):
            last = seq.iloc[-1]
            metadata_rows.append({
                "feature_mode": mode,
                "patient": str(last["patient"]),
                "recording": str(last["recording"]),
                "start_sec": float(last["start_sec"]),
                "label": int(label),
                "probability": float(probability),
            })

        predictions = pd.DataFrame(metadata_rows)
        predictions.to_csv(
            OUTPUT_DIR / f"{mode}_dev_predictions.csv",
            index=False,
        )

        pooled = metrics_for_group(predictions)
        pooled["feature_mode"] = mode
        pooled_rows.append(pooled)

        for patient, group in predictions.groupby("patient"):
            row = metrics_for_group(group)
            row.update({"feature_mode": mode, "patient": patient})
            patient_rows.append(row)

        for recording, group in predictions.groupby("recording"):
            row = metrics_for_group(group)
            row.update({"feature_mode": mode, "recording": recording})
            recording_rows.append(row)

        print(
            f"{mode}: saved {len(predictions)} development predictions; "
            f"{predictions['patient'].nunique()} patients, "
            f"{predictions['recording'].nunique()} recordings"
        )

    pooled_df = pd.DataFrame(pooled_rows)
    patient_df = pd.DataFrame(patient_rows)
    recording_df = pd.DataFrame(recording_rows)

    pooled_df.to_csv(OUTPUT_DIR / "pooled_metrics.csv", index=False)
    patient_df.to_csv(OUTPUT_DIR / "per_patient_metrics.csv", index=False)
    recording_df.to_csv(OUTPUT_DIR / "per_recording_metrics.csv", index=False)

    print("\nPOOLED DEVELOPMENT METRICS")
    print(pooled_df.to_string(index=False))

    print("\nPATIENT-LEVEL SUMMARY")
    for mode, group in patient_df.groupby("feature_mode"):
        valid_auc = group["roc_auc"].dropna()
        print(
            f"{mode}: patients={len(group)}, "
            f"patients with both classes={len(valid_auc)}, "
            f"median patient AUC="
            f"{valid_auc.median() if len(valid_auc) else np.nan:.4f}, "
            f"patients with BAcc > 0.5="
            f"{(group['balanced_accuracy'] > 0.5).sum()}/{len(group)}"
        )

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Held-out evaluation split was not used.")


if __name__ == "__main__":
    main()
