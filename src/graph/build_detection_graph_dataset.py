#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract EEG windows and build artifact-invariant 20-node multimodal graphs
for Patient-Independent Seizure Detection (Ictal vs. Interictal).
"""

import os
import sys
from pathlib import Path
import mne
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_DIR / 'src'))
sys.path.append(str(REPO_DIR / 'src/preprocessing'))
sys.path.append(str(REPO_DIR / 'src/graph'))

from montage import create_bipolar_montage, resample_data
from filtering import filter_eeg
from node_features import extract_node_features
from time_frequency_features import extract_time_frequency_features

INPUT_MANIFEST = REPO_DIR / 'data/metadata/detection_manifest.csv'
OUTPUT_MANIFEST = REPO_DIR / 'data/metadata/detection_graph_manifest.csv'
GRAPH_DIR = REPO_DIR / 'data/graph_detection'
ADJACENCY_PATH = REPO_DIR / 'data/graph/adjacency.npy'

TARGET_SFREQ = 256
WINDOW_SAMPLES = int(5.0 * TARGET_SFREQ)  # 1280 samples


def main():
    print("=" * 80)
    print("BUILDING DETECTION INVARIANT MULTIMODAL GRAPH DATASET")
    print("=" * 80)

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing {INPUT_MANIFEST}")

    df_manifest = pd.read_csv(INPUT_MANIFEST)
    print(f"Total detection windows: {len(df_manifest)}")
    print(f"Unique recordings:      {df_manifest['recording'].nunique()}")
    print(f"Unique patients:        {df_manifest['patient'].nunique()}")

    adjacency = np.load(ADJACENCY_PATH).astype(np.float32)

    for split in ['train', 'dev', 'eval']:
        (GRAPH_DIR / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    recordings = df_manifest.groupby('recording', sort=True)
    total_recs = len(recordings)

    rec_idx = 0
    for rec_name, group in recordings:
        rec_idx += 1
        edf_rel = group['edf_local'].iloc[0]
        edf_path = REPO_DIR / edf_rel
        split = group['split'].iloc[0]
        patient = group['patient'].iloc[0]

        if not edf_path.exists():
            print(f"[{rec_idx}/{total_recs}] Skipping missing EDF: {edf_path}")
            continue

        print(f"[{rec_idx}/{total_recs}] Processing {rec_name} ({split}) - {len(group)} windows...")

        try:
            raw_mne = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            orig_sfreq = raw_mne.info['sfreq']
            bipolar_data, ch_names = create_bipolar_montage(raw_mne)
        except Exception as e:
            print(f"  Error loading {rec_name}: {e}")
            continue

        # Extract features for all windows in this recording
        rec_hc_list = []
        rec_tf_list = []
        valid_indices = []

        for row_i, row in group.iterrows():
            s_sec = row['start_sec']
            e_sec = row['end_sec']

            native_start = int(round(s_sec * orig_sfreq))
            native_end = int(round(e_sec * orig_sfreq))

            if native_end > bipolar_data.shape[1]:
                native_end = bipolar_data.shape[1]

            segment = bipolar_data[:, native_start:native_end]
            if segment.shape[1] == 0:
                continue

            # Resample & filter
            segment_resampled = resample_data(segment, orig_sfreq, TARGET_SFREQ)
            filtered_seg = filter_eeg(segment_resampled, TARGET_SFREQ)

            if filtered_seg.shape[1] < WINDOW_SAMPLES:
                # Zero-pad if slight boundary truncation
                pad = np.zeros((20, WINDOW_SAMPLES), dtype=filtered_seg.dtype)
                pad[:, :filtered_seg.shape[1]] = filtered_seg
                filtered_seg = pad
            else:
                filtered_seg = filtered_seg[:, :WINDOW_SAMPLES]

            try:
                hc_feats, _ = extract_node_features(filtered_seg)  # (20, 16)
                tf_feats = extract_time_frequency_features(filtered_seg)  # (20, 10)
                rec_hc_list.append(hc_feats.astype(np.float32))
                rec_tf_list.append(tf_feats.astype(np.float32))
                valid_indices.append(row_i)
            except Exception as e:
                continue

        if not rec_hc_list:
            continue

        # Convert to arrays: (N_windows, 20, 16) and (N_windows, 20, 10)
        hc_arrs = np.stack(rec_hc_list, axis=0)
        tf_arrs = np.stack(rec_tf_list, axis=0)

        # Per-recording standardization of dynamic features
        dyn_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        hc_dyn = hc_arrs[:, :, dyn_idx]
        hc_mean = hc_dyn.mean(axis=(0, 1), keepdims=True)
        hc_std = hc_dyn.std(axis=(0, 1), keepdims=True)
        hc_std = np.where(hc_std < 1e-8, 1.0, hc_std)
        hc_normed_dyn = (hc_dyn - hc_mean) / hc_std

        rel_idx = [5, 11, 12, 13, 14, 15]
        hc_rel = hc_arrs[:, :, rel_idx]
        hc_invariant = np.concatenate([hc_rel, hc_normed_dyn], axis=-1)

        tf_mean = tf_arrs.mean(axis=(0, 1), keepdims=True)
        tf_std = tf_arrs.std(axis=(0, 1), keepdims=True)
        tf_std = np.where(tf_std < 1e-8, 1.0, tf_std)
        tf_normed = (tf_arrs - tf_mean) / tf_std

        # Fuse into 26-dimensional invariant node feature vectors
        fused_features = np.concatenate([hc_invariant, tf_normed], axis=-1)  # (N_windows, 20, 26)

        # Save individual graph archives
        for k_idx, orig_row_idx in enumerate(valid_indices):
            row_data = group.loc[orig_row_idx]
            w_id = int(row_data['window_id'])
            lbl = int(row_data['label'])

            save_path = GRAPH_DIR / split / f"{w_id:06d}.npz"
            np.savez_compressed(
                save_path,
                x=fused_features[k_idx],
                adjacency=adjacency,
                y=lbl
            )

            row_dict = row_data.to_dict()
            row_dict['graph_path'] = str(save_path.relative_to(REPO_DIR))
            manifest_rows.append(row_dict)

    df_out = pd.DataFrame(manifest_rows)
    df_out.to_csv(OUTPUT_MANIFEST, index=False)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Created {len(df_out)} detection graphs in {GRAPH_DIR}")
    print(f"Saved manifest: {OUTPUT_MANIFEST}")
    print("\nBreakdown by split and label:")
    print(df_out.groupby(['split', 'label_type']).size().unstack(fill_value=0))
    print("=" * 80)


if __name__ == '__main__':
    main()
