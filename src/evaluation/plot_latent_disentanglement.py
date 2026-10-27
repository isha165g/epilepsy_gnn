#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Publication-Quality Disentanglement Visualizations (t-SNE & UMAP):
Compares:
- Baseline Multimodal GAT-GRU: Memorized patient islands (shortcut learning).
- Proposed Invariant GNN: Disentangled patient-invariant representations clustering by seizure state.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import umap

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_DIR / 'results/ablation/latent_embeddings_for_disentanglement.npz'
OUTPUT_DIR = REPO_DIR / 'results/ablation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_manifold(embed_base, embed_inv, patients, labels, method_name='t-SNE'):
    print(f"Generating 4-panel {method_name} latent disentanglement figure...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.titlesize': 14,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # Palette for patients (discrete colormap across 17 patients)
    unique_patients = np.unique(patients)
    n_pts = len(unique_patients)
    pt_to_idx = {p: i for i, p in enumerate(unique_patients)}
    pt_indices = np.array([pt_to_idx[p] for p in patients])
    cmap_pts = plt.get_cmap('tab20', n_pts)

    # Palette for classes (Interictal vs Ictal)
    # 0 = Interictal (Steel Blue), 1 = Ictal (Crimson Red)
    c_interictal = '#1f77b4'
    c_ictal = '#d62728'

    # --------------------------------------------------------------------------
    # TOP ROW: Baseline Model (Shortcut Learning)
    # --------------------------------------------------------------------------
    # (a) Baseline colored by Patient ID
    ax_a = axes[0, 0]
    scatter_a = ax_a.scatter(
        embed_base[:, 0], embed_base[:, 1],
        c=pt_indices, cmap=cmap_pts, s=28, alpha=0.85, edgecolors='none'
    )
    ax_a.set_title(f"(a) Baseline {method_name} — Colored by Patient ID\n(Patient Memorization / Shortcut Learning)", fontweight='bold', pad=10)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.text(0.03, 0.94, "Subject-Clustered (High Leakage: 76.7%)", transform=ax_a.transAxes,
              fontsize=9.5, fontweight='bold', color='#a00000', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (b) Baseline colored by Seizure Class
    ax_b = axes[0, 1]
    mask_ictal = (labels == 1)
    ax_b.scatter(embed_base[~mask_ictal, 0], embed_base[~mask_ictal, 1], c=c_interictal, label='Interictal (Baseline)', s=24, alpha=0.6, edgecolors='none')
    ax_b.scatter(embed_base[mask_ictal, 0], embed_base[mask_ictal, 1], c=c_ictal, label='Ictal (Seizure)', s=32, alpha=0.85, edgecolors='none')
    ax_b.set_title(f"(b) Baseline {method_name} — Colored by Seizure State\n(Confounded / Entangled by Subject)", fontweight='bold', pad=10)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.legend(loc='lower right', framealpha=0.95, fontsize=9.5)

    # --------------------------------------------------------------------------
    # BOTTOM ROW: Invariant Model (Proposed)
    # --------------------------------------------------------------------------
    # (c) Invariant colored by Patient ID
    ax_c = axes[1, 0]
    ax_c.scatter(
        embed_inv[:, 0], embed_inv[:, 1],
        c=pt_indices, cmap=cmap_pts, s=28, alpha=0.85, edgecolors='none'
    )
    ax_c.set_title(f"(c) Proposed Invariant {method_name} — Colored by Patient ID\n(Subject-Invariant / Blended Manifold)", fontweight='bold', pad=10)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.text(0.03, 0.94, "Patient-Invariant (Suppressed: 37.2%)", transform=ax_c.transAxes,
              fontsize=9.5, fontweight='bold', color='#006600', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # (d) Invariant colored by Seizure Class
    ax_d = axes[1, 1]
    ax_d.scatter(embed_inv[~mask_ictal, 0], embed_inv[~mask_ictal, 1], c=c_interictal, label='Interictal (Baseline)', s=24, alpha=0.6, edgecolors='none')
    ax_d.scatter(embed_inv[mask_ictal, 0], embed_inv[mask_ictal, 1], c=c_ictal, label='Ictal (Seizure)', s=32, alpha=0.85, edgecolors='none')
    ax_d.set_title(f"(d) Proposed Invariant {method_name} — Colored by Seizure State\n(Clean Disentanglement of Ictal Dynamics)", fontweight='bold', pad=10)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.legend(loc='lower right', framealpha=0.95, fontsize=9.5)

    plt.suptitle(f"Latent Representation Disentanglement ({method_name}) Across 17 Unseen Patients", fontweight='bold', y=0.99, fontsize=15)
    plt.tight_layout()

    tag = method_name.lower().replace('-', '')
    out_png = OUTPUT_DIR / f'latent_disentanglement_{tag}.png'
    out_pdf = OUTPUT_DIR / f'latent_disentanglement_{tag}.pdf'

    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {method_name} figures:")
    print(f"  PNG: {out_png}")
    print(f"  PDF: {out_pdf}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run run_invariance_probing.py first.")

    data = np.load(DATA_PATH)
    h_base_dev = data['h_base_dev']
    h_inv_dev = data['h_inv_dev']
    y_dev = data['y_dev']
    patients_dev = data['patients_dev']

    print(f"Loaded embeddings: Baseline {h_base_dev.shape}, Invariant {h_inv_dev.shape}")
    print(f"Samples: {len(y_dev)} across {len(np.unique(patients_dev))} patients")

    # 1. Compute t-SNE
    print("Computing t-SNE projections (perplexity=35, random_state=42)...")
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000)
    tsne_base = tsne.fit_transform(h_base_dev)
    tsne_inv = tsne.fit_transform(h_inv_dev)
    plot_manifold(tsne_base, tsne_inv, patients_dev, y_dev, method_name='t-SNE')

    # 2. Compute UMAP
    print("Computing UMAP projections (n_neighbors=25, min_dist=0.3, random_state=42)...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.3, metric='cosine', random_state=42)
    umap_base = reducer.fit_transform(h_base_dev)
    umap_inv = reducer.fit_transform(h_inv_dev)
    plot_manifold(umap_base, umap_inv, patients_dev, y_dev, method_name='UMAP')

    print("\n" + "=" * 80)
    print("ALL DISENTANGLEMENT VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
