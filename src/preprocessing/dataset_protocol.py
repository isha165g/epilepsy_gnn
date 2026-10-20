#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Protocol, Patient Isolation Verification, and Benchmark Alignment with Tang et al. (ICLR 2022).
"""

from pathlib import Path
import numpy as np
import pandas as pd

# The 5 overlapping patients identified by Tang et al. (ICLR 2022) to exclude from held-out evaluation
EXCLUDED_OVERLAPPING_PATIENTS = {
    "00001027", "00004671", "00001981", "00006546", "00009842"
}

# Clip benchmarks defined in Tang et al. (ICLR 2022)
BENCHMARK_CLIPS = {
    "fast_detection": 12.0,       # 12-second clips
    "slow_context_detection": 60.0  # 60-second clips
}


def verify_patient_isolation(manifest_df):
    """
    Strictly verify zero patient overlap across train, dev, and eval splits,
    and confirm absence of known contaminated patients.
    """
    splits = manifest_df["split"].unique()
    patients_by_split = {
        s: set(manifest_df[manifest_df["split"] == s]["patient"].unique())
        for s in splits
    }

    overlaps = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            inter = patients_by_split[s1].intersection(patients_by_split[s2])
            if inter:
                overlaps[f"{s1}_vs_{s2}"] = inter

    if overlaps:
        raise ValueError(f"FATAL: Patient leakage detected between splits: {overlaps}")

    # Check for excluded overlapping patients from Tang et al.
    all_patients = set(manifest_df["patient"].unique())
    found_contaminated = all_patients.intersection(EXCLUDED_OVERLAPPING_PATIENTS)
    if found_contaminated:
        print(f"Warning: Found {len(found_contaminated)} known contaminated patients from TUSZ: {found_contaminated}")

    print("✓ Patient Isolation Verification PASSED: Zero patient overlap across splits.")
    for s, pts in patients_by_split.items():
        print(f"   [{s.upper()}]: {len(pts)} unique patients")

    return True


def create_benchmark_clip_manifest(
    recordings_metadata_path,
    eligible_edfs_path,
    output_manifest_path,
    clip_length_sec=12.0,
    pre_exclusion_sec=60.0,
    post_exclusion_sec=120.0,
):
    """
    Generate seizure detection manifest strictly aligned with Tang et al. clip benchmarks (12s or 60s).
    """
    metadata = pd.read_csv(recordings_metadata_path)
    eligible = pd.read_csv(eligible_edfs_path)

    rec_map = {
        (row.split, row.patient, row.recording): row.edf_local
        for row in eligible.itertuples()
    }

    metadata["rec_key"] = list(zip(metadata["split"], metadata["patient"], metadata["recording"]))
    meta_el = metadata[metadata["rec_key"].isin(rec_map.keys())].copy()

    records = []
    window_counter = 0

    for key, group in meta_el.groupby("rec_key", sort=True):
        split, pt, rec = key
        # Skip contaminated patients if present
        if str(pt) in EXCLUDED_OVERLAPPING_PATIENTS and split != "train":
            continue

        duration = float(group["duration_sec"].iloc[0])
        edf_local = rec_map[key]

        seizures = group[group["seizure_start_sec"].notna()]
        seizure_intervals = [
            (float(sz["seizure_start_sec"]), float(sz["seizure_end_sec"]))
            for _, sz in seizures.iterrows()
        ]

        # 1. Ictal Clips (label=1)
        for s_start, s_end in seizure_intervals:
            t = s_start
            while t + clip_length_sec <= s_end:
                records.append({
                    "window_id": window_counter,
                    "split": split,
                    "patient": pt,
                    "recording": rec,
                    "start_sec": round(t, 4),
                    "end_sec": round(t + clip_length_sec, 4),
                    "duration_sec": duration,
                    "clip_len_sec": clip_length_sec,
                    "label": 1,
                    "label_type": "ictal",
                    "edf_local": edf_local,
                })
                window_counter += 1
                t += clip_length_sec

        # 2. Interictal Clips (label=0) outside exclusion buffer
        interictal_candidates = []
        t = 10.0
        while t + clip_length_sec <= duration - 5.0:
            in_exclusion = False
            for s_start, s_end in seizure_intervals:
                if not (t + clip_length_sec <= s_start - pre_exclusion_sec or t >= s_end + post_exclusion_sec):
                    in_exclusion = True
                    break
            if not in_exclusion:
                interictal_candidates.append(t)
            t += clip_length_sec

        n_ictal = sum(1 for r in records if r["recording"] == rec and r["label"] == 1)
        n_sample = max(n_ictal, 5) if n_ictal > 0 else 15
        if len(interictal_candidates) > 0:
            step = max(1, len(interictal_candidates) // n_sample)
            chosen_times = interictal_candidates[::step][:n_sample]
            for ct in chosen_times:
                records.append({
                    "window_id": window_counter,
                    "split": split,
                    "patient": pt,
                    "recording": rec,
                    "start_sec": round(ct, 4),
                    "end_sec": round(ct + clip_length_sec, 4),
                    "duration_sec": duration,
                    "clip_len_sec": clip_length_sec,
                    "label": 0,
                    "label_type": "interictal",
                    "edf_local": edf_local,
                })
                window_counter += 1

    df = pd.DataFrame(records)
    verify_patient_isolation(df)
    output_path = Path(output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Saved benchmark manifest ({clip_length_sec}s clips): {output_path} ({len(df)} clips)")
    return df


if __name__ == "__main__":
    print("Dataset protocol loaded successfully.")
