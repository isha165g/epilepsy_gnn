from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

# pyrefly: ignore [missing-import]
from src.models.temporal_ablation_dataset import (
    TemporalAblationDataset,
)
# pyrefly: ignore [missing-import]
from src.models.gat_gru_ablation import GATGRUAblation


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 7
SEQUENCE_LENGTH = 6

FEATURE_MODES = {
    "handcrafted": 16,
    "time_frequency": 10,
    "multimodal": 26,
}

OUTPUT_DIR = Path("results/ablation")
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_model(feature_mode, input_dim):
    seed_everything(SEED)

    print("\n" + "=" * 72)
    print(f"ABLATION: {feature_mode.upper()}")
    print("=" * 72)

    train_dataset = TemporalAblationDataset(
        split="train",
        feature_mode=feature_mode,
        sequence_length=SEQUENCE_LENGTH,
    )
    dev_dataset = TemporalAblationDataset(
        split="dev",
        feature_mode=feature_mode,
        sequence_length=SEQUENCE_LENGTH,
    )

    # Ensure each experiment uses the expected controlled dataset.
    expected_counts = {"train": 2764, "dev": 5409}
    if len(train_dataset) != expected_counts["train"]:
        raise RuntimeError(
            f"Unexpected train count: {len(train_dataset)}"
        )
    if len(dev_dataset) != expected_counts["dev"]:
        raise RuntimeError(
            f"Unexpected dev count: {len(dev_dataset)}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    train_labels = np.array([
        int(seq.iloc[-1]["label_graph"])
        for seq in train_dataset.sequences
    ])
    class_counts = np.bincount(train_labels, minlength=2)

    if np.any(class_counts == 0):
        raise ValueError(f"Missing class: {class_counts}")

    class_weights = len(train_labels) / (2.0 * class_counts)
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=DEVICE
    )

    print("Input features:", input_dim)
    print("Train class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    model = GATGRUAblation(input_dim=input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    adjacency = np.load("data/graph/adjacency.npy")
    edge_np = np.vstack(np.where(adjacency > 0)).astype(np.int64)
    shared_edge_index = torch.tensor(
        edge_np, dtype=torch.long, device=DEVICE
    )

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, batch_edges, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                batch_edges = batch_edges.to(DEVICE)

                if not torch.all(batch_edges == batch_edges[0]):
                    raise ValueError("Different graph topologies in batch.")

                logits = model(x, shared_edge_index)
                loss = criterion(logits, y)

                total_loss += loss.item() * y.size(0)
                total_correct += (
                    (logits.argmax(dim=1) == y).sum().item()
                )
                total_samples += y.size(0)

        return (
            total_loss / total_samples,
            total_correct / total_samples,
        )

    checkpoint_path = MODEL_DIR / f"{feature_mode}_best.pt"
    log_path = LOG_DIR / f"{feature_mode}_training.csv"

    history = []
    best_dev_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, batch_edges, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            batch_edges = batch_edges.to(DEVICE)

            if not torch.all(batch_edges == batch_edges[0]):
                raise ValueError("Different graph topologies in batch.")

            optimizer.zero_grad()
            logits = model(x, shared_edge_index)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (
                (logits.argmax(dim=1) == y).sum().item()
            )
            total_samples += y.size(0)

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        dev_loss, dev_accuracy = evaluate(dev_loader)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_accuracy:.4f} | "
            f"Dev loss {dev_loss:.4f} | "
            f"Dev acc {dev_accuracy:.4f}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dev_loss": best_dev_loss,
                    "class_weights": class_weights.cpu(),
                    "config": {
                        "feature_mode": feature_mode,
                        "input_dim": input_dim,
                        "sequence_length": SEQUENCE_LENGTH,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "seed": SEED,
                    },
                },
                checkpoint_path,
            )
            print("  Saved best checkpoint.")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(log_path, index=False)

    summary = {
        "feature_mode": feature_mode,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "train_sequences": len(train_dataset),
        "dev_sequences": len(dev_dataset),
        "train_class_counts": class_counts.tolist(),
        "checkpoint": str(checkpoint_path),
        "training_log": str(log_path),
    }

    print("\nBest epoch:", best_epoch)
    print(f"Best dev loss: {best_dev_loss:.6f}")
    print("Checkpoint:", checkpoint_path)
    print("Training log:", log_path)

    return summary


def main():
    summaries = []

    for feature_mode, input_dim in FEATURE_MODES.items():
        summaries.append(train_one_model(feature_mode, input_dim))

    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 72)
    print("ALL ABLATION TRAINING RUNS COMPLETED")
    print("=" * 72)
    print("Summary:", summary_path)
    print("Evaluation split was not used.")


if __name__ == "__main__":
    main()
