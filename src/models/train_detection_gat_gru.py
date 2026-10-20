#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Domain-Adversarial Invariant GAT-GRU for Patient-Independent Seizure Detection (T=1 Window).
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

REPO_DIR = Path('/home/isha/epilepsy_gnn')
import sys
sys.path.append(str(REPO_DIR / 'src'))

from models.detection_graph_dataset import DetectionGraphDataset
from models.gat_gru_invariant import InvariantGATGRU

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
PATIENCE = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH = REPO_DIR / 'results/models/gat_gru_detection_best.pt'
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
ADJACENCY_PATH = REPO_DIR / 'data/graph/adjacency.npy'


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    print("=" * 80)
    print("TRAINING PATIENT-INDEPENDENT INVARIANT GAT-GRU FOR SEIZURE DETECTION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    train_ds = DetectionGraphDataset(split='train', sequence_length=1)
    dev_ds = DetectionGraphDataset(split='dev', sequence_length=1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Adjacency edge index
    adj = np.load(ADJACENCY_PATH)
    edge_index = torch.tensor(np.vstack(np.where(adj > 0)).astype(np.int64), dtype=torch.long, device=DEVICE)

    model = InvariantGATGRU(
        input_dim=26,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_patients=train_ds.num_train_patients,
        dropout=0.3
    ).to(DEVICE)

    criterion_task = nn.CrossEntropyLoss()
    criterion_patient = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_dev_acc = 0.0
    best_dev_bacc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Training Loop...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_batches = len(train_loader)
        train_task_loss = 0.0

        for b_idx, (x, y, pt_idx) in enumerate(train_loader):
            p = float(b_idx + (epoch - 1) * total_batches) / (MAX_EPOCHS * total_batches)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pt_idx = pt_idx.to(DEVICE)

            optimizer.zero_grad()
            task_logits, patient_logits, _ = model(x, edge_index, alpha=alpha)

            loss_task = criterion_task(task_logits, y)
            loss_patient = criterion_patient(patient_logits, pt_idx)
            loss_total = loss_task + 0.5 * alpha * loss_patient

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_task_loss += loss_task.item()

        train_task_loss /= total_batches

        # Evaluate on 17 Unseen Patients (Dev)
        model.eval()
        dev_y_true, dev_y_pred, dev_y_prob = [], [], []

        with torch.no_grad():
            for x, y, pt_idx in dev_loader:
                x = x.to(DEVICE)
                task_logits, _, _ = model(x, edge_index, alpha=0.0)
                probs = torch.softmax(task_logits, dim=1)[:, 1]
                preds = (probs >= 0.5).long()

                dev_y_true.extend(y.numpy().tolist())
                dev_y_pred.extend(preds.cpu().numpy().tolist())
                dev_y_prob.extend(probs.cpu().numpy().tolist())

        y_true = np.array(dev_y_true)
        y_pred = np.array(dev_y_pred)
        y_prob = np.array(dev_y_prob)

        dev_acc = accuracy_score(y_true, y_pred)
        dev_bacc = balanced_accuracy_score(y_true, y_pred)
        dev_auc = roc_auc_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS:2d} (alpha={alpha:.3f}) | "
              f"Train Loss: {train_task_loss:.4f} | "
              f"Dev Acc: {dev_acc*100:5.2f}% | "
              f"Dev BAcc: {dev_bacc*100:5.2f}% | "
              f"Dev Sens: {sens*100:5.2f}% | "
              f"Dev Spec: {spec*100:5.2f}% | "
              f"Dev AUC: {dev_auc:.4f}")

        # Checkpoint if best balanced accuracy
        if dev_bacc > best_dev_bacc:
            best_dev_bacc = dev_bacc
            best_dev_acc = dev_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dev_bacc': best_dev_bacc,
                'best_dev_acc': best_dev_acc,
                'best_dev_auc': dev_auc,
                'sensitivity': sens,
                'specificity': spec,
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best Dev BAcc: {best_dev_bacc*100:.2f}% (Epoch {best_epoch})")
                break

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Best Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Best Epoch: {best_epoch} | Best Dev Acc: {best_dev_acc*100:.2f}% | Best Dev BAcc: {best_dev_bacc*100:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
