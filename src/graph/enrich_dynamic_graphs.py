#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich all detection graph archives (.npz) with:
1. Dynamic Cross-Correlation Adjacency (Tang et al. ICLR 2022)
2. Dynamic Phase-Locking Value (PLV) Adjacency (Theta/Beta Synchrony)
"""

import sys
import time
from pathlib import Path
import mne
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_DIR))

from src.preprocessing.montage import create_bipolar_montage, resample_data
from src.preprocessing.filtering import filter_eeg
from src.graph.dynamic_topology import (
    compute_correlation_adjacency,
    compute_plv_adjacency,
    compute_static_distance_adjacency,
)

MANIFEST_PATH = REPO_DIR / 'data/metadata/detection_graph_manifest.csv'
TARGET_SFREQ = 256
WINDOW_SAMPLES = int(5.0 * TARGET_SFREQ)  # 1280 samples


def main():
    print("=" * 80)
    print("ENRICHING DETECTION DATASET WITH DYNAMIC CORRELATION & PLV TOPOLOGIES")
    print("=" * 80)

    df_manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Total windows: {len(df_manifest)}")
    print(f"Total recordings: {df_manifest['recording'].nunique()}")

    static_adj = compute_static_distance_adjacency()

    recordings = df_manifest.groupby('recording', sort=True)
    total_recs = len(recordings)
    t_start = time.time()

    processed_count = 0
    rec_idx = 0

    for rec_name, group in recordings:
        rec_idx += 1
        edf_rel = group['edf_local'].iloc[0]
        edf_path = REPO_DIR / edf_rel

        if not edf_path.exists():
            print(f"[{rec_idx}/{total_recs}] Skipping missing EDF: {edf_path}")
            continue

        # Load EDF
        try:
            raw_mne = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            orig_sfreq = raw_mne.info['sfreq']
            bipolar_data, _ = create_bipolar_montage(raw_mne)
            bipolar_resampled = resample_data(bipolar_data, orig_sfreq, TARGET_SFREQ)
            filtered_eeg = filter_eeg(bipolar_resampled, TARGET_SFREQ)
        except Exception as e:
            print(f"[{rec_idx}/{total_recs}] Error reading {rec_name}: {e}")
            continue

        for _, row in group.iterrows():
            g_path = REPO_DIR / row['graph_path']
            if not g_path.exists():
                continue

            # Load existing archive
            data = np.load(g_path)
            x = data['x']
            y = data['y']
            adj_static = data.get('adj_static', data.get('adjacency', static_adj))

            s_sec = float(row['start_sec'])
            e_sec = float(row['end_sec'])

            start_idx = int(round(s_sec * TARGET_SFREQ))
            end_idx = int(round(e_sec * TARGET_SFREQ))

            if start_idx >= filtered_eeg.shape[1]:
                seg = np.zeros((20, WINDOW_SAMPLES), dtype=np.float32)
            else:
                seg = filtered_eeg[:, start_idx:end_idx]
                if seg.shape[1] < WINDOW_SAMPLES:
                    pad = np.zeros((20, WINDOW_SAMPLES), dtype=np.float32)
                    pad[:, :seg.shape[1]] = seg
                    seg = pad
                else:
                    seg = seg[:, :WINDOW_SAMPLES]

            # Compute dynamic correlation & PLV
            adj_corr = compute_correlation_adjacency(seg, top_tau=3)
            adj_plv = compute_plv_adjacency(seg, sfreq=TARGET_SFREQ, band=(4.0, 30.0), top_tau=3)

            # Save updated archive with all 3 discrete topologies
            np.savez_compressed(
                g_path,
                x=x,
                adj_static=adj_static,
                adj_corr=adj_corr,
                adj_plv=adj_plv,
                y=y,
            )
            processed_count += 1

        if rec_idx % 10 == 0 or rec_idx == total_recs:
            elapsed = time.time() - t_start
            print(f"[{rec_idx:2d}/{total_recs}] Processed {processed_count}/{len(df_manifest)} windows ({elapsed:.1f}s)")

    print("\n" + "=" * 80)
    print(f"COMPLETED ENRICHMENT: {processed_count} files enriched with Dynamic Correlation & PLV")
    print(f"Total time elapsed: {time.time() - t_start:.2f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
