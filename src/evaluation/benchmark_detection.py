#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Patient-Independent Seizure Detection on Held-Out Unseen Patients (T=1 Window).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    precision_recall_curve
)

REPO_DIR = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(REPO_DIR / 'src'))

from models.detection_graph_dataset import DetectionGraphDataset
from models.gat_gru_invariant import InvariantGATGRU

CHECKPOINT_PATH = REPO_DIR / 'results/models/gat_gru_detection_best.pt'
OUTPUT_DIR = REPO_DIR / 'results/detection'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADJACENCY_PATH = REPO_DIR / 'data/graph/adjacency.npy'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 80)
    print("BENCHMARKING PATIENT-INDEPENDENT SEIZURE DETECTION")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    dev_ds = DetectionGraphDataset(split='dev', sequence_length=1)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    adj = np.load(ADJACENCY_PATH)
    edge_index = torch.tensor(np.vstack(np.where(adj > 0)).astype(np.int64), dtype=torch.long, device=DEVICE)

    # Load model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = InvariantGATGRU(
        input_dim=26,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_patients=dev_ds.num_train_patients,
        dropout=0.3
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    y_true_all = []
    y_prob_all = []
    patients_all = []
    recordings_all = []

    with torch.no_grad():
        for i, (x, y, _) in enumerate(dev_loader):
            x = x.to(DEVICE)
            logits, _, _ = model(x, edge_index, alpha=0.0)
            probs = torch.softmax(logits, dim=1)[:, 1]

            y_true_all.extend(y.numpy().tolist())
            y_prob_all.extend(probs.cpu().numpy().tolist())

    for s in dev_ds.samples:
        patients_all.append(s.iloc[-1]['patient'])
        recordings_all.append(s.iloc[-1]['recording'])

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    overall_metrics = {
        'Model': 'Patient-Independent Invariant GAT-GRU',
        'Evaluation': 'Held-Out Unseen Patients (Zero Calibration)',
        'Dev_Accuracy_%': round(acc * 100, 2),
        'Dev_Balanced_Accuracy_%': round(bacc * 100, 2),
        'Dev_Sensitivity_%': round(sens * 100, 2),
        'Dev_Specificity_%': round(spec * 100, 2),
        'Dev_F1_Score_%': round(f1 * 100, 2),
        'Dev_ROC_AUC': round(auc, 4),
        'Dev_PR_AUC': round(ap, 4),
        'True_Positives': int(tp),
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'Total_Sequences': len(y_true),
        'Unseen_Patients': len(set(patients_all))
    }

    df_overall = pd.DataFrame([overall_metrics])
    df_overall.to_csv(OUTPUT_DIR / 'detection_metrics.csv', index=False)

    # Per-Patient Breakdown
    df_preds = pd.DataFrame({
        'patient': patients_all,
        'recording': recordings_all,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred
    })
    df_preds.to_csv(OUTPUT_DIR / 'detection_predictions.csv', index=False)

    patient_rows = []
    for pt, group in df_preds.groupby('patient'):
        pt_yt = group['y_true'].to_numpy()
        pt_yp = group['y_pred'].to_numpy()
        pt_acc = accuracy_score(pt_yt, pt_yp)
        pt_pos = int((pt_yt == 1).sum())
        pt_neg = int((pt_yt == 0).sum())
        patient_rows.append({
            'patient': pt,
            'total_sequences': len(group),
            'ictal_count': pt_pos,
            'interictal_count': pt_neg,
            'accuracy_%': round(pt_acc * 100, 2)
        })

    df_pt = pd.DataFrame(patient_rows)
    df_pt.to_csv(OUTPUT_DIR / 'detection_patient_breakdown.csv', index=False)

    # Plot ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: ROC Curve
    ax1.plot(fpr, tpr, color='#2563eb', linewidth=2.5, label=f'Invariant GAT-GRU (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', label='Random Chance (AUC = 0.500)')
    ax1.scatter([1 - spec], [sens], color='#ef4444', s=80, zorder=5, label=f'Operating Point (Sens={sens*100:.1f}%, Spec={spec*100:.1f}%)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax1.set_title('Cross-Patient ROC Curve (17 Unseen Patients)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', framealpha=0.9)

    # Panel B: Precision-Recall Curve
    ax2.plot(rec, prec, color='#16a34a', linewidth=2.5, label=f'PR Curve (AP = {ap:.3f})')
    ax2.axhline(y_true.mean(), color='#94a3b8', linestyle='--', label=f'Baseline Chance ({y_true.mean()*100:.1f}%)')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('Cross-Patient Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', framealpha=0.9)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / 'detection_roc_curve.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve figure: {fig_path}")

    print("\n" + "=" * 90)
    print("PATIENT-INDEPENDENT SEIZURE DETECTION BENCHMARK RESULTS")
    print("=" * 90)
    for k, v in overall_metrics.items():
        print(f"{k:30s}: {v}")
    print("=" * 90)


if __name__ == '__main__':
    main()
