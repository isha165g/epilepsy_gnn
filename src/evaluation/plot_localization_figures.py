#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Publication-Quality Visualizations for Phase 5: Quantitative Localization & Interpretability.

Methodologies Visualized:
- Panel (a): Tang et al. (ICLR 2022) Full Perturbation Occlusion
- Panel (b): Raw GAT Spatial Attention
- Panel (c): Feature-Energy Weighted Graph Attention (Real-Time Bedside Mode)
- Panel (d): Energy-Guided Focal Coarse-to-Fine Occlusion (Physician Diagnostic Review Mode)
- Panel (e): Quantitative Interpretability Metric Comparison (Coverage, Precision, Precise Clips > 0.8)
- Panel (f): Inference Latency vs. Localization Precision Pareto Frontier
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_DIR / 'results/ablation/saliency_heatmaps_sample.npz'
CSV_PATH = REPO_DIR / 'results/ablation/quantitative_localization_benchmark.csv'
OUTPUT_PNG = REPO_DIR / 'results/ablation/quantitative_localization_visualization.png'
OUTPUT_PDF = REPO_DIR / 'results/ablation/quantitative_localization_visualization.pdf'

from src.preprocessing.montage import BIPOLAR_PAIRS
BIPOLAR_NAMES = [name for name, _, _ in BIPOLAR_PAIRS]


def main():
    print("=" * 80)
    print("GENERATING PHASE 5 QUANTITATIVE LOCALIZATION VISUALIZATIONS")
    print("=" * 80)

    data = np.load(DATA_PATH, allow_pickle=True)
    records = data['records']
    occ_cov = data['occ_coverages']
    occ_loc = data['occ_localizations']
    gat_cov = data['gat_coverages']
    gat_loc = data['gat_localizations']
    ew_cov = data['ew_coverages']
    ew_loc = data['ew_localizations']
    focal_cov = data['focal_coverages']
    focal_loc = data['focal_localizations']

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9.5,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'figure.titlesize': 14,
    })

    fig = plt.figure(figsize=(20, 11), dpi=300)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.35)

    sample = records[0]
    M_annot = sample['M_annot']      # (20, 5)
    occ_norm = sample['occ_norm']    # (20, 5)
    gat_norm = sample['gat_norm']    # (20, 5)
    ew_norm = sample['ew_norm']      # (20, 5)
    focal_norm = sample['focal_norm'] # (20, 5)

    seiz_chans, seiz_times = np.where(M_annot > 0)

    def add_ground_truth(ax):
        if len(seiz_chans) > 0:
            c_min, c_max = np.min(seiz_chans), np.max(seiz_chans)
            t_min, t_max = np.min(seiz_times), np.max(seiz_times)
            rect = patches.Rectangle(
                (t_min - 0.45, c_min - 0.45), t_max - t_min + 0.9, c_max - c_min + 0.9,
                linewidth=2.0, edgecolor='#0000ff', facecolor='none', linestyle='--', label='Neurologist Ground Truth'
            )
            ax.add_patch(rect)
            ax.legend(loc='upper right', framealpha=0.9, fontsize=7.5)

    # --------------------------------------------------------------------------
    # Panel (a): Tang et al. Full Perturbation Occlusion Map
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(occ_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_a.set_title("(a) Tang et al. Full Occlusion\n(101 Passes, ~504 ms)", fontweight='bold', pad=8)
    ax_a.set_yticks(np.arange(len(BIPOLAR_NAMES)))
    ax_a.set_yticklabels(BIPOLAR_NAMES, fontsize=7.5)
    ax_a.set_xlabel("Time (Seconds)", fontweight='bold')
    ax_a.set_ylabel("Bipolar EEG Channels", fontweight='bold')
    add_ground_truth(ax_a)
    plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04, label='Normalized Saliency')

    # --------------------------------------------------------------------------
    # Panel (b): Raw GAT Spatial Attention Map
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(gat_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_b.set_title("(b) Raw GAT Spatial Attention\n(1 Pass, 6.2 ms, Diffuse)", fontweight='bold', pad=8)
    ax_b.set_yticks(np.arange(len(BIPOLAR_NAMES)))
    ax_b.set_yticklabels(BIPOLAR_NAMES, fontsize=7.5)
    ax_b.set_xlabel("Time (Seconds)", fontweight='bold')
    add_ground_truth(ax_b)
    plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04, label='Normalized Attention')

    # --------------------------------------------------------------------------
    # Panel (c): Feature-Energy Weighted Graph Attention
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(ew_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_c.set_title("(c) Feature-Energy Weighted GAT\n(1 Pass, 6.2 ms, Real-Time Bedside)", fontweight='bold', pad=8)
    ax_c.set_yticks(np.arange(len(BIPOLAR_NAMES)))
    ax_c.set_yticklabels(BIPOLAR_NAMES, fontsize=7.5)
    ax_c.set_xlabel("Time (Seconds)", fontweight='bold')
    add_ground_truth(ax_c)
    plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04, label='Energy-Weighted Attention')

    # --------------------------------------------------------------------------
    # Panel (d): Energy-Guided Focal Coarse-to-Fine Occlusion
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[0, 3])
    im_d = ax_d.imshow(focal_norm, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_d.set_title("(d) Energy-Guided Focal Occlusion\n(16 Passes, ~78 ms, Diagnostic Review)", fontweight='bold', pad=8)
    ax_d.set_yticks(np.arange(len(BIPOLAR_NAMES)))
    ax_d.set_yticklabels(BIPOLAR_NAMES, fontsize=7.5)
    ax_d.set_xlabel("Time (Seconds)", fontweight='bold')
    add_ground_truth(ax_d)
    plt.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04, label='Focal Guided Saliency')

    # --------------------------------------------------------------------------
    # Panel (e): Quantitative Metrics Comparison Bar Chart
    # --------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 0:2])
    labels = [
        'Tang et al.\nFull Occlusion',
        'Raw GAT\nSpatial Attention',
        'Feature-Energy\nWeighted GAT',
        'Energy-Guided\nFocal Occlusion'
    ]
    
    cov_means = [np.mean(occ_cov) * 100, np.mean(gat_cov) * 100, np.mean(ew_cov) * 100, np.mean(focal_cov) * 100]
    loc_means = [np.mean(occ_loc) * 100, np.mean(gat_loc) * 100, np.mean(ew_loc) * 100, np.mean(focal_loc) * 100]
    prec_rates = [np.mean(np.array(occ_loc) > 0.8) * 100, np.mean(np.array(gat_loc) > 0.8) * 100, np.mean(np.array(ew_loc) > 0.8) * 100, np.mean(np.array(focal_loc) > 0.8) * 100]

    x = np.arange(len(labels))
    width = 0.25

    rects1 = ax_e.bar(x - width, cov_means, width, label='Coverage (Recall %)', color='#1f77b4', edgecolor='black')
    rects2 = ax_e.bar(x, loc_means, width, label='Localization (Precision %)', color='#ff7f0e', edgecolor='black')
    rects3 = ax_e.bar(x + width, prec_rates, width, label='Precise Clips (>0.8 %)', color='#2ca02c', edgecolor='black')

    ax_e.set_ylabel('Percentage (%)', fontweight='bold')
    ax_e.set_title('(e) Interpretability Performance across 100 Unseen Seizures', fontweight='bold', pad=8)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(labels, fontweight='bold', fontsize=8.5)
    ax_e.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax_e.grid(axis='y', linestyle='--', alpha=0.5)
    ax_e.set_ylim([0, 105])

    for rect_group in [rects1, rects2, rects3]:
        for r in rect_group:
            h = r.get_height()
            ax_e.annotate(f"{h:.1f}%",
                          xy=(r.get_x() + r.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # --------------------------------------------------------------------------
    # Panel (f): Latency vs. Precision Pareto Frontier
    # --------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2:4])
    latencies = [504.10, 6.17, 6.20, 77.85]  # ms
    precisions = loc_means
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    markers = ['s', 'o', '*', '^']
    sizes = [160, 180, 320, 260]

    for i in range(4):
        ax_f.scatter(latencies[i], precisions[i], color=colors[i], marker=markers[i], s=sizes[i],
                     edgecolor='black', linewidth=1.5, zorder=5, label=labels[i].replace('\n', ' '))
        ax_f.annotate(f"{labels[i].replace(chr(10), ' ')}\n({latencies[i]:.1f} ms, {precisions[i]:.1f}%)",
                      xy=(latencies[i], precisions[i]),
                      xytext=(10 if i not in [2, 3] else (-45 if i == 2 else -35), -20 if i != 2 else 12), textcoords="offset points",
                      fontweight='bold', fontsize=8,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor=colors[i]))

    ax_f.set_xscale('log')
    ax_f.set_xlim([2, 1200])
    ax_f.set_ylim([20, 105])
    ax_f.set_xlabel('Inference Latency per Clip (ms, log scale) [Lower is Better]', fontweight='bold')
    ax_f.set_ylabel('Localization Precision (%) [Higher is Better]', fontweight='bold')
    ax_f.set_title('(f) Speed vs. Precision Pareto Frontier (Two Clinical Modes)', fontweight='bold', pad=8)
    ax_f.grid(True, linestyle='--', alpha=0.5)

    ax_f.text(0.04, 0.84, "Dual-Mode Clinical Deployment:\n• Real-Time Bedside (6.2 ms): Continuous 24/7 ICU Monitoring\n• Diagnostic Review (78 ms): High-Precision (80%+) Physician Drill-Down",
              transform=ax_f.transAxes, fontsize=8.5, fontweight='bold', color='#155724',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda', edgecolor='#c3e6cb'))

    plt.suptitle("Quantitative Seizure Localization & Interpretability Benchmark",
                 fontweight='bold', y=0.99, fontsize=14)

    plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved localization figures:")
    print(f"  PNG: {OUTPUT_PNG}")
    print(f"  PDF: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
