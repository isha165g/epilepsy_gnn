#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Controlled Ablation Matrix Benchmark for Brain Connectivity Topologies.
Compares:
1. Static Distance Graph (Anatomical 10-20 system)
2. Tang et al. Dynamic Correlation Graph (Normalized cross-correlation, top-tau=3)
3. Dynamic Phase-Locking Value (PLV) Coherence Graph (Hilbert phase synchrony, top-tau=3)
4. Learnable Adaptive Graph Topology (A_adapt = Softmax(ReLU(E_1 * E_2^T)))
"""

from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

REPO_DIR = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(REPO_DIR))

from src.models.detection_graph_dataset import DetectionGraphDataset
from src.models.invariant_topology_gnn import InvariantTopologyGNN

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 10
PATIENCE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = REPO_DIR / 'results/ablation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = REPO_DIR / 'results/models'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval_topology(topology_name, topology_mode):
    set_seed(SEED)
    print("\n" + "=" * 80)
    print(f"BENCHMARKING TOPOLOGY: {topology_name.upper()} (mode='{topology_mode}')")
    print("=" * 80)

    train_ds = DetectionGraphDataset(split='train', sequence_length=1, topology_mode=topology_mode)
    dev_ds = DetectionGraphDataset(split='dev', sequence_length=1, topology_mode=topology_mode)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = InvariantTopologyGNN(
        time_dim=16,
        freq_dim=10,
        fused_dim=32,
        gnn_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_nodes=20,
        num_patients=train_ds.num_train_patients,
        topology_mode=topology_mode,
        dropout=0.3,
    ).to(DEVICE)

    criterion_task = nn.CrossEntropyLoss()
    criterion_patient = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_dev_bacc = 0.0
    best_metrics = {}
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_batches = len(train_loader)
        train_task_loss = 0.0
        train_y_true, train_y_prob = [], []

        for b_idx, (x, adj, y, pt_idx) in enumerate(train_loader):
            p = float(b_idx + (epoch - 1) * total_batches) / (MAX_EPOCHS * total_batches)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            x = x.to(DEVICE)
            adj = adj.to(DEVICE)
            y = y.to(DEVICE)
            pt_idx = pt_idx.to(DEVICE)

            optimizer.zero_grad()
            task_logits, patient_logits, _, _, _ = model(x, adj, alpha=alpha)

            loss_task = criterion_task(task_logits, y)
            loss_patient = criterion_patient(patient_logits, pt_idx)
            loss_total = loss_task + 0.5 * alpha * loss_patient

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_task_loss += loss_task.item()
            train_probs = torch.softmax(task_logits, dim=1)[:, 1].detach()
            train_y_true.extend(y.cpu().numpy().tolist())
            train_y_prob.extend(train_probs.cpu().numpy().tolist())

        train_task_loss /= total_batches
        train_auc = roc_auc_score(train_y_true, train_y_prob)

        # Evaluation on held-out development patients
        model.eval()
        dev_y_true, dev_y_pred, dev_y_prob = [], [], []

        with torch.no_grad():
            for x, adj, y, pt_idx in dev_loader:
                x = x.to(DEVICE)
                adj = adj.to(DEVICE)
                task_logits, _, _, _, _ = model(x, adj, alpha=0.0)

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
        dev_ap = average_precision_score(y_true, y_prob)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS:2d} | Train Loss: {train_task_loss:.4f} | Train AUC: {train_auc:.4f} | "
              f"Dev BAcc: {dev_bacc*100:5.2f}% | Sens: {sens*100:5.2f}% | Spec: {spec*100:5.2f}% | Dev AUC: {dev_auc:.4f}")

        if dev_bacc > best_dev_bacc:
            best_dev_bacc = dev_bacc
            patience_counter = 0
            best_metrics = {
                'topology': topology_name,
                'topology_mode': topology_mode,
                'best_epoch': epoch,
                'train_auc': train_auc,
                'dev_auc': dev_auc,
                'dev_ap': dev_ap,
                'dev_bacc': dev_bacc,
                'dev_acc': dev_acc,
                'sensitivity': sens,
                'specificity': spec,
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
            }
            # Save best checkpoint
            ckpt_path = CHECKPOINT_DIR / f"{topology_mode}_topology_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': best_metrics,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered for {topology_name} at epoch {epoch}.")
                break

    best_metrics['training_time_sec'] = round(time.time() - t0, 2)
    return best_metrics


def main():
    topologies = [
        ("Static Distance Graph", "static"),
        ("Dynamic Correlation Graph (Tang et al.)", "correlation"),
        ("Dynamic PLV Coherence Graph", "plv"),
        ("Learnable Adaptive Graph Topology", "adaptive"),
    ]

    results = []
    for topo_name, topo_mode in topologies:
        res = train_and_eval_topology(topo_name, topo_mode)
        results.append(res)

    df_results = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / 'topology_ablation_matrix.csv'
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("PHASE 3: CONTROLLED TOPOLOGY ABLATION MATRIX (HELD-OUT 17 UNSEEN PATIENTS)")
    print("=" * 90)
    cols = ['topology', 'train_auc', 'dev_auc', 'dev_ap', 'dev_bacc', 'sensitivity', 'specificity', 'training_time_sec']
    display_df = df_results[cols].copy()
    display_df['dev_bacc'] = (display_df['dev_bacc'] * 100).map('{:.2f}%'.format)
    display_df['sensitivity'] = (display_df['sensitivity'] * 100).map('{:.2f}%'.format)
    display_df['specificity'] = (display_df['specificity'] * 100).map('{:.2f}%'.format)
    display_df['dev_auc'] = display_df['dev_auc'].map('{:.4f}'.format)
    display_df['dev_ap'] = display_df['dev_ap'].map('{:.4f}'.format)
    display_df['train_auc'] = display_df['train_auc'].map('{:.4f}'.format)

    print(display_df.to_string(index=False))
    print("\n" + "=" * 90)
    print(f"Results saved to: {out_csv}")
    print("=" * 90)


if __name__ == '__main__':
    main()
