from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from src.models.temporal_ablation_dataset import TemporalAblationDataset

SEED = 42
SEQUENCE_LENGTH = 6
OUTPUT_DIR = Path("results/ablation/baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sequences(split):
    dataset = TemporalAblationDataset(
        split=split,
        feature_mode="time_frequency",
        sequence_length=SEQUENCE_LENGTH,
    )

    features, labels = [], []

    for i in range(len(dataset)):
        x, _, y = dataset[i]

        # (6 time steps, 20 nodes, 10 features) -> 1200 features
        features.append(x.numpy().reshape(-1))
        labels.append(y.item())

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


print("=" * 65)
print("SEQUENCE-LEVEL TIME-FREQUENCY LOGISTIC REGRESSION")
print("=" * 65)

X_train, y_train = load_sequences("train")
X_dev, y_dev = load_sequences("dev")

print("Train shape:", X_train.shape)
print("Dev shape:  ", X_dev.shape)
print("Train class counts:", np.bincount(y_train, minlength=2).tolist())
print("Dev class counts:  ", np.bincount(y_dev, minlength=2).tolist())

# Fit the classifier on training sequences only.
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    solver="lbfgs",
    random_state=SEED,
)
model.fit(X_train, y_train)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Evaluate on training data only
train_prob = model.predict_proba(X_train)[:, 1]
train_pred = model.predict(X_train)

tn, fp, fn, tp = confusion_matrix(
    y_train, train_pred, labels=[0, 1]
).ravel()

print("\n" + "=" * 60)
print("TRAINING RESULTS — TIME-FREQUENCY LOGISTIC REGRESSION")
print("=" * 60)
print(f"ROC-AUC:            {roc_auc_score(y_train, train_prob):.4f}")
print(f"Average Precision:  {average_precision_score(y_train, train_prob):.4f}")
print(f"Balanced Accuracy:  {balanced_accuracy_score(y_train, train_pred):.4f}")
print(f"Sensitivity:        {tp / (tp + fn):.4f}")
print(f"Specificity:        {tn / (tn + fp):.4f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

probabilities = model.predict_proba(X_dev)[:, 1]
predictions = (probabilities >= 0.5).astype(np.int64)

tn, fp, fn, tp = confusion_matrix(
    y_dev, predictions, labels=[0, 1]
).ravel()

metrics = {
    "model": "sequence_logistic_time_frequency",
    "sequence_length": SEQUENCE_LENGTH,
    "input_features": X_train.shape[1],
    "train_samples": len(y_train),
    "dev_samples": len(y_dev),
    "roc_auc": roc_auc_score(y_dev, probabilities),
    "average_precision": average_precision_score(y_dev, probabilities),
    "balanced_accuracy": balanced_accuracy_score(y_dev, predictions),
    "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
    "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

print("\nDEVELOPMENT RESULTS")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

pd.DataFrame([metrics]).to_csv(
    OUTPUT_DIR / "sequence_logistic_time_frequency_metrics.csv",
    index=False,
)

pd.DataFrame({
    "label": y_dev,
    "probability": probabilities,
    "prediction": predictions,
}).to_csv(
    OUTPUT_DIR / "sequence_logistic_time_frequency_dev_predictions.csv",
    index=False,
)

print("\nSaved baseline metrics and development predictions to", OUTPUT_DIR)
