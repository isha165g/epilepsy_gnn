#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Invariant Gated Multimodal GAT-GRU for Patient-Independent Seizure Detection
and Log Multimodal Gating Statistics (Phase 2 Empirical Verification).
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
sys.path.append(str(REPO_DIR))

from src.models.detection_graph_dataset import DetectionGraphDataset
from src.models.gat_gru_invariant import InvariantGatedMultimodalGATGRU

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
PATIENCE = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH = REPO_DIR / 'results/models/gated_multimodal_invariant_best.pt'
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
    print("TRAINING INVARIANT GATED MULTIMODAL SPATIOTEMPORAL GNN (GMU + GAT-GRU)")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    train_ds = DetectionGraphDataset(split='train', sequence_length=1)
    dev_ds = DetectionGraphDataset(split='dev', sequence_length=1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Adjacency edge index
    adj = np.load(ADJACENCY_PATH)
    edge_index = torch.tensor(np.vstack(np.where(adj > 0)).astype(np.int64), dtype=torch.long, device=DEVICE)

    # In the invariant dataset: 16 handcrafted (time stream) + 10 STFT (freq stream) = 26 features
    model = InvariantGatedMultimodalGATGRU(
        time_dim=16,
        freq_dim=10,
        fused_dim=32,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_patients=train_ds.num_train_patients,
        dropout=0.3
    ).to(DEVICE)

    criterion_task = nn.CrossEntropyLoss()
    criterion_patient = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_dev_bacc = 0.0
    best_dev_acc = 0.0
    best_dev_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting Gated Multimodal Training Loop...")
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
            task_logits, patient_logits, _, _ = model(x, edge_index, alpha=alpha)

            loss_task = criterion_task(task_logits, y)
            loss_patient = criterion_patient(patient_logits, pt_idx)
            loss_total = loss_task + 0.5 * alpha * loss_patient

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_task_loss += loss_task.item()

        train_task_loss /= total_batches

        # Evaluate on Held-out Patients (Dev split)
        model.eval()
        dev_y_true, dev_y_pred, dev_y_prob = [], [], []
        ictal_gates, interictal_gates = [], []

        with torch.no_grad():
            for x, y, pt_idx in dev_loader:
                x = x.to(DEVICE)
                task_logits, _, _, gate = model(x, edge_index, alpha=0.0)

                probs = torch.softmax(task_logits, dim=1)[:, 1]
                preds = (probs >= 0.5).long()

                dev_y_true.extend(y.numpy().tolist())
                dev_y_pred.extend(preds.cpu().numpy().tolist())
                dev_y_prob.extend(probs.cpu().numpy().tolist())

                # Collect gate values per class
                gate_mean = gate.mean(dim=(1, 2)).cpu().numpy()  # (batch,)
                for g_val, lbl in zip(gate_mean, y.numpy()):
                    if lbl == 1:
                        ictal_gates.append(g_val)
                    else:
                        interictal_gates.append(g_val)

        y_true = np.array(dev_y_true)
        y_pred = np.array(dev_y_pred)
        y_prob = np.array(dev_y_prob)

        dev_acc = accuracy_score(y_true, y_pred)
        dev_bacc = balanced_accuracy_score(y_true, y_pred)
        dev_auc = roc_auc_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        mean_ictal_gate = float(np.mean(ictal_gates)) if ictal_gates else 0.0
        mean_interictal_gate = float(np.mean(interictal_gates)) if interictal_gates else 0.0

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS:2d} (alpha={alpha:.3f}) | "
              f"Train Loss: {train_task_loss:.4f} | "
              f"Dev BAcc: {dev_bacc*100:5.2f}% | "
              f"Sens: {sens*100:5.2f}% | "
              f"Spec: {spec*100:5.2f}% | "
              f"AUC: {dev_auc:.4f} | "
              f"Gate (Ictal: {mean_ictal_gate:.3f}, Interictal: {mean_interictal_gate:.3f})")

        if dev_bacc > best_dev_bacc:
            best_dev_bacc = dev_bacc
            best_dev_acc = dev_acc
            best_dev_auc = dev_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dev_bacc': best_dev_bacc,
                'best_dev_acc': best_dev_acc,
                'best_dev_auc': best_dev_auc,
                'sensitivity': sens,
                'specificity': spec,
                'mean_ictal_gate': mean_ictal_gate,
                'mean_interictal_gate': mean_interictal_gate,
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best Dev BAcc: {best_dev_bacc*100:.2f}% (Epoch {best_epoch})")
                break

    print("\n" + "=" * 80)
    print(f"GATED MULTIMODAL TRAINING COMPLETE")
    print(f"Best Dev Balanced Accuracy: {best_dev_bacc*100:.2f}% | AUC: {best_dev_auc:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    main()
