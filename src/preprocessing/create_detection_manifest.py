#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Detection Manifest (Ictal vs. Interictal) for Patient-Independent Seizure Detection.

Class Definition:
- Positive (label=1, label_type='ictal'): 5-second windows occurring during active seizure intervals.
- Negative (label=0, label_type='interictal'): 5-second windows occurring during non-seizure baseline,
  strictly outside all seizure exclusion zones ([seizure_start - 60s, seizure_end + 120s]).

Partitions strictly follow patient disjoint splits:
- train (30 patients)
- dev (17 patients)
- eval (6 patients)
"""

from pathlib import Path
import pandas as pd
import numpy as np

REPO_DIR = Path('/home/isha/epilepsy_gnn')
METADATA_PATH = REPO_DIR / 'data/metadata/recordings_metadata.csv'
ELIGIBLE_EDFS_PATH = REPO_DIR / 'data/metadata/eligible_edfs.csv'
OUTPUT_MANIFEST_PATH = REPO_DIR / 'data/metadata/detection_manifest.csv'

WINDOW_SIZE_SEC = 5.0
PRE_EXCLUSION_SEC = 60.0    # 1 min pre-seizure buffer
POST_EXCLUSION_SEC = 120.0  # 2 min postictal buffer


def main():
    print("=" * 80)
    print("GENERATING PATIENT-INDEPENDENT SEIZURE DETECTION MANIFEST (ICTAL VS INTERICTAL)")
    print("=" * 80)

    metadata = pd.read_csv(METADATA_PATH)
    eligible_edfs = pd.read_csv(ELIGIBLE_EDFS_PATH)

    rec_map = {
        (row.split, row.patient, row.recording): row.edf_local
        for row in eligible_edfs.itertuples()
    }

    metadata['rec_key'] = list(zip(metadata['split'], metadata['patient'], metadata['recording']))
    meta_el = metadata[metadata['rec_key'].isin(rec_map.keys())].copy()

    records = []
    window_counter = 0

    for key, group in meta_el.groupby('rec_key', sort=True):
        split, pt, rec = key
        duration = float(group['duration_sec'].iloc[0])
        edf_local = rec_map[key]

        seizures = group[group['seizure_start_sec'].notna()]
        seizure_intervals = [
            (float(sz['seizure_start_sec']), float(sz['seizure_end_sec']))
            for _, sz in seizures.iterrows()
        ]

        # 1. Generate Ictal Windows (label=1)
        n_ictal = 0
        for s_start, s_end in seizure_intervals:
            t = s_start
            while t + WINDOW_SIZE_SEC <= s_end:
                records.append({
                    'window_id': window_counter,
                    'split': split,
                    'patient': pt,
                    'recording': rec,
                    'start_sec': round(t, 4),
                    'end_sec': round(t + WINDOW_SIZE_SEC, 4),
                    'duration_sec': duration,
                    'label': 1,
                    'label_type': 'ictal',
                    'edf_local': edf_local,
                })
                window_counter += 1
                n_ictal += 1
                t += WINDOW_SIZE_SEC

        # 2. Identify Safe Interictal Candidate Windows (label=0)
        # Exclude: [s_start - PRE_EXCLUSION_SEC, s_end + POST_EXCLUSION_SEC]
        interictal_candidates = []
        t = 10.0  # Avoid initial 10s amplifier settling
        while t + WINDOW_SIZE_SEC <= duration - 5.0:
            in_exclusion = False
            for s_start, s_end in seizure_intervals:
                if not (t + WINDOW_SIZE_SEC <= s_start - PRE_EXCLUSION_SEC or t >= s_end + POST_EXCLUSION_SEC):
                    in_exclusion = True
                    break
            if not in_exclusion:
                interictal_candidates.append(t)
            t += WINDOW_SIZE_SEC

        # Sample balanced interictal windows
        n_sample = max(n_ictal, 5) if n_ictal > 0 else 20
        if len(interictal_candidates) > 0:
            step = max(1, len(interictal_candidates) // n_sample)
            chosen_times = interictal_candidates[::step][:n_sample]
            for ct in chosen_times:
                records.append({
                    'window_id': window_counter,
                    'split': split,
                    'patient': pt,
                    'recording': rec,
                    'start_sec': round(ct, 4),
                    'end_sec': round(ct + WINDOW_SIZE_SEC, 4),
                    'duration_sec': duration,
                    'label': 0,
                    'label_type': 'interictal',
                    'edf_local': edf_local,
                })
                window_counter += 1

    df_manifest = pd.DataFrame(records)
    OUTPUT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(OUTPUT_MANIFEST_PATH, index=False)

    print(f"\nManifest successfully created: {OUTPUT_MANIFEST_PATH}")
    print(f"Total Windows: {len(df_manifest)}")
    print("\nClass breakdown by split:")
    print(df_manifest.groupby(['split', 'label_type']).size().unstack(fill_value=0))
    print("\nPatient counts per split:")
    print(df_manifest.groupby('split')['patient'].nunique())


if __name__ == '__main__':
    main()
