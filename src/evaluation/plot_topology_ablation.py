#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Publication-Quality Figures for Phase 3: Topology Ablation Matrix.
Plots:
1. Panel (a): Performance Metrics across Topologies (BAcc, AUC, Sens, Spec).
2. Panel (b): Cross-Patient ROC Curves for all 4 Graph Topologies on 17 Unseen Patients.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

REPO_DIR = Path('/home/isha/epilepsy_gnn')
import sys
sys.path.append(str(REPO_DIR))

from src.models.detection_graph_dataset import DetectionGraphDataset
from src.models.invariant_topology_gnn import InvariantTopologyGNN

CSV_PATH = REPO_DIR / 'results/ablation/topology_ablation_matrix.csv'
CHECKPOINT_DIR = REPO_DIR / 'results/models'
OUTPUT_PNG = REPO_DIR / 'results/ablation/topology_ablation_comparison.png'
OUTPUT_PDF = REPO_DIR / 'results/ablation/topology_ablation_comparison.pdf'
DEVICE = torch.device('cpu')


def main():
    print("=" * 80)
    print("GENERATING PHASE 3 TOPOLOGY ABLATION FIGURES")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)

    # Style configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # --------------------------------------------------------------------------
    # Panel (a): Metric Bar Chart
    # --------------------------------------------------------------------------
    topologies = [
        "Static Distance",
        "Dynamic Corr\n(Tang et al.)",
        "Dynamic PLV\nCoherence",
        "Adaptive\nLearned",
    ]
    bacc = df['dev_bacc'].to_numpy() * 100
    auc = df['dev_auc'].to_numpy() * 100
    sens = df['sensitivity'].to_numpy() * 100
    spec = df['specificity'].to_numpy() * 100

    x = np.arange(len(topologies))
    width = 0.18

    c_bacc = '#1f77b4'
    c_auc = '#2ca02c'
    c_sens = '#d62728'
    c_spec = '#9467bd'

    rects1 = ax1.bar(x - 1.5 * width, bacc, width, label='Balanced Acc', color=c_bacc, alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x - 0.5 * width, auc, width, label='ROC-AUC', color=c_auc, alpha=0.9, edgecolor='black', linewidth=0.8)
    rects3 = ax1.bar(x + 0.5 * width, sens, width, label='Sensitivity', color=c_sens, alpha=0.9, edgecolor='black', linewidth=0.8)
    rects4 = ax1.bar(x + 1.5 * width, spec, width, label='Specificity', color=c_spec, alpha=0.9, edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Score (%)', fontweight='bold')
    ax1.set_title('(a) Cross-Patient Detection Metrics (17 Unseen Patients)', fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topologies, fontweight='semibold')
    ax1.set_ylim([60, 105])
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.95)

    # Add numeric labels
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(f'{h:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, rotation=90)

    # --------------------------------------------------------------------------
    # Panel (b): ROC Curves of All 4 Topologies
    # --------------------------------------------------------------------------
    topo_configs = [
        ("Static Distance Graph", "static", "#1f77b4", "-"),
        ("Dynamic Correlation Graph (Tang et al.)", "correlation", "#ff7f0e", "--"),
        ("Dynamic PLV Coherence Graph", "plv", "#2ca02c", "-"),
        ("Learnable Adaptive Graph Topology", "adaptive", "#d62728", "-."),
    ]

    for name, mode, color, ls in topo_configs:
        ckpt_path = CHECKPOINT_DIR / f"{mode}_topology_best.pt"
        if not ckpt_path.exists():
            continue

        dev_ds = DetectionGraphDataset(split='dev', sequence_length=1, topology_mode=mode)
        loader = DataLoader(dev_ds, batch_size=32, shuffle=False)

        model = InvariantTopologyGNN(
            time_dim=16, freq_dim=10, fused_dim=32, gnn_hidden=32,
            embedding_dim=64, gru_hidden=64, num_nodes=20,
            num_patients=30, topology_mode=mode, dropout=0.0
        ).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        y_true, y_prob = [], []
        with torch.no_grad():
            for x_b, adj_b, y_b, _ in loader:
                logits, _, _, _, _ = model(x_b.to(DEVICE), adj_b.to(DEVICE), alpha=0.0)
                prob = torch.softmax(logits, dim=1)[:, 1]
                y_true.extend(y_b.numpy().tolist())
                y_prob.extend(prob.cpu().numpy().tolist())

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax2.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.2, label=f"{name} (AUC = {auc_val:.4f})")

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.50)')
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax2.set_title('(b) Held-Out ROC Curves across Brain Topologies', fontweight='bold', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved high-resolution figures:")
    print(f"  PNG: {OUTPUT_PNG}")
    print(f"  PDF: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
