import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("src"))

# pyrefly: ignore [missing-import]
from preprocessing.labels import (
    get_exclusion_interval,
    get_positive_interval,
)


# ============================================================
# Configuration
# ============================================================

MANIFEST_PATH = "data/metadata/window_manifest.csv"
METADATA_PATH = "data/metadata/recordings_metadata.csv"

SPH_SEC = 5 * 60
SOP_SEC = 5 * 60
POSTICTAL_SEC = 10 * 60
WINDOW_SIZE_SEC = 5

TOLERANCE = 1e-6


# ============================================================
# Helpers
# ============================================================

def recording_key(row):
    return (
        str(row["split"]),
        str(row["patient"]),
        str(row["recording"]),
    )


def intervals_overlap(start1, end1, start2, end2):
    return (
        start1 < end2 - TOLERANCE
        and end1 > start2 + TOLERANCE
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("AUDITING TUSZ WINDOW MANIFEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    manifest = pd.read_csv(MANIFEST_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    print("\nLoaded:")
    print(f"Manifest rows : {len(manifest):,}")
    print(f"Metadata rows : {len(metadata):,}")

    # --------------------------------------------------------
    # Basic columns
    # --------------------------------------------------------

    required_manifest = {
        "split",
        "patient",
        "recording",
        "start_sec",
        "end_sec",
        "duration_sec",
        "label",
        "label_type",
        "target_seizure_start_sec",
    }

    missing = required_manifest - set(manifest.columns)

    if missing:
        raise RuntimeError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # TEST 1 — Label consistency
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 1: LABEL CONSISTENCY")
    print("-" * 80)

    errors = []

    for _, row in manifest.iterrows():

        label = int(row["label"])
        label_type = row["label_type"]

        if label == 1 and label_type != "preictal":
            errors.append(
                f"Positive row has label_type={label_type}"
            )

        if label == 0 and label_type != "interictal":
            errors.append(
                f"Negative row has label_type={label_type}"
            )

    if errors:
        print(f"FAIL: {len(errors)} label inconsistencies")
        for error in errors[:10]:
            print(" ", error)
        raise RuntimeError("Label consistency test failed.")

    print("PASS ✓ Labels and label_type are consistent.")


    # --------------------------------------------------------
    # TEST 2 — Window duration
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 2: WINDOW DURATIONS")
    print("-" * 80)

    durations = (
        manifest["end_sec"] - manifest["start_sec"]
    )

    bad_duration = ~np.isclose(
        durations,
        WINDOW_SIZE_SEC,
        atol=TOLERANCE,
    )

    if bad_duration.any():

        print(
            f"FAIL: {bad_duration.sum()} windows "
            f"do not have duration {WINDOW_SIZE_SEC}s."
        )

        print(
            manifest.loc[
                bad_duration,
                ["split", "patient", "recording",
                 "start_sec", "end_sec"]
            ].head(10)
        )

        raise RuntimeError("Window duration test failed.")

    print(
        f"PASS ✓ All {len(manifest):,} windows "
        f"are exactly {WINDOW_SIZE_SEC}s."
    )


    # --------------------------------------------------------
    # TEST 3 — Recording boundaries
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 3: RECORDING BOUNDARIES")
    print("-" * 80)

    invalid_start = manifest["start_sec"] < -TOLERANCE
    invalid_end = (
        manifest["end_sec"]
        > manifest["duration_sec"] + TOLERANCE
    )

    invalid = invalid_start | invalid_end

    if invalid.any():

        print(
            f"FAIL: {invalid.sum()} windows "
            "fall outside recording boundaries."
        )

        print(
            manifest.loc[
                invalid,
                ["split", "patient", "recording",
                 "start_sec", "end_sec",
                 "duration_sec"]
            ].head(10)
        )

        raise RuntimeError(
            "Recording-boundary test failed."
        )

    print("PASS ✓ All windows are inside recordings.")


    # --------------------------------------------------------
    # TEST 4 — Positive event count
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 4: POSITIVE EVENT COUNTS")
    print("-" * 80)

    positive = manifest[
        manifest["label"] == 1
    ].copy()

    positive_events = (
        positive
        .groupby(
            [
                "split",
                "patient",
                "recording",
                "target_seizure_start_sec",
            ]
        )
        .size()
    )

    expected_per_event = SPH_SEC // WINDOW_SIZE_SEC

    bad_events = positive_events[
        positive_events != expected_per_event
    ]

    if not bad_events.empty:

        print(
            "FAIL: Some forecast events do not have "
            f"{expected_per_event} positive windows."
        )

        print(bad_events)

        raise RuntimeError(
            "Positive event count test failed."
        )

    print(
        f"PASS ✓ {len(positive_events)} forecast events "
        f"each contain exactly {expected_per_event} windows."
    )


    # --------------------------------------------------------
    # TEST 5 — Positive windows lie in target interval
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 5: POSITIVE INTERVAL CORRECTNESS")
    print("-" * 80)

    bad_positive = []

    for _, row in positive.iterrows():

        seizure_start = float(
            row["target_seizure_start_sec"]
        )

        interval = get_positive_interval(
            seizure_start=seizure_start,
            sph_sec=SPH_SEC,
            sop_sec=SOP_SEC,
        )

        start = float(row["start_sec"])
        end = float(row["end_sec"])

        if (
            start < interval.start - TOLERANCE
            or end > interval.end + TOLERANCE
        ):
            bad_positive.append(
                {
                    "split": row["split"],
                    "patient": row["patient"],
                    "recording": row["recording"],
                    "window_start": start,
                    "window_end": end,
                    "target_seizure": seizure_start,
                    "positive_start": interval.start,
                    "positive_end": interval.end,
                }
            )

    if bad_positive:

        print(
            f"FAIL: {len(bad_positive)} positive windows "
            "fall outside their target preictal interval."
        )

        print(pd.DataFrame(bad_positive).head(10))

        raise RuntimeError(
            "Positive interval test failed."
        )

    print(
        "PASS ✓ Every positive window lies inside "
        "its target [T-10min, T-5min] interval."
    )


    # --------------------------------------------------------
    # Build seizure lookup
    # --------------------------------------------------------

    metadata["recording_key"] = metadata.apply(
        recording_key,
        axis=1,
    )

    seizures_by_recording = {}

    for key, group in metadata.groupby(
        "recording_key"
    ):

        seizures_by_recording[key] = [
            (
                float(row["seizure_start_sec"]),
                float(row["seizure_end_sec"]),
            )
            for _, row in group.iterrows()
        ]


    # --------------------------------------------------------
    # TEST 6 — Positive windows do not overlap seizures
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 6: POSITIVE WINDOWS VS ACTUAL SEIZURES")
    print("-" * 80)

    seizure_overlap_errors = []

    for _, row in positive.iterrows():

        key = (
            str(row["split"]),
            str(row["patient"]),
            str(row["recording"]),
        )

        start = float(row["start_sec"])
        end = float(row["end_sec"])

        for seizure_start, seizure_end in (
            seizures_by_recording.get(key, [])
        ):

            if intervals_overlap(
                start,
                end,
                seizure_start,
                seizure_end,
            ):
                seizure_overlap_errors.append(
                    {
                        "split": row["split"],
                        "patient": row["patient"],
                        "recording": row["recording"],
                        "window_start": start,
                        "window_end": end,
                        "seizure_start": seizure_start,
                        "seizure_end": seizure_end,
                    }
                )

    if seizure_overlap_errors:

        print(
            f"FAIL: {len(seizure_overlap_errors)} "
            "positive windows overlap an actual seizure."
        )

        print(
            pd.DataFrame(
                seizure_overlap_errors
            ).head(10)
        )

        raise RuntimeError(
            "Positive/seizure overlap test failed."
        )

    print(
        "PASS ✓ No positive window overlaps an actual seizure."
    )


    # --------------------------------------------------------
    # TEST 7 — Negative windows are clean
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 7: NEGATIVE WINDOW CLEANLINESS")
    print("-" * 80)

    negative = manifest[
        manifest["label"] == 0
    ].copy()

    negative_errors = []

    for _, row in negative.iterrows():

        key = (
            str(row["split"]),
            str(row["patient"]),
            str(row["recording"]),
        )

        start = float(row["start_sec"])
        end = float(row["end_sec"])

        for seizure_start, seizure_end in (
            seizures_by_recording.get(key, [])
        ):

            exclusion = get_exclusion_interval(
                seizure_start=seizure_start,
                seizure_end=seizure_end,
                postictal_sec=POSTICTAL_SEC,
                sph_sec=SPH_SEC,
                sop_sec=SOP_SEC,
            )

            exclusion_start = max(
                0.0,
                exclusion.start,
            )

            exclusion_end = min(
                float(row["duration_sec"]),
                exclusion.end,
            )

            if intervals_overlap(
                start,
                end,
                exclusion_start,
                exclusion_end,
            ):
                negative_errors.append(
                    {
                        "split": row["split"],
                        "patient": row["patient"],
                        "recording": row["recording"],
                        "window_start": start,
                        "window_end": end,
                        "exclusion_start": exclusion_start,
                        "exclusion_end": exclusion_end,
                    }
                )

                break

    if negative_errors:

        print(
            f"FAIL: {len(negative_errors)} negative windows "
            "fall inside seizure exclusion intervals."
        )

        print(
            pd.DataFrame(
                negative_errors
            ).head(10)
        )

        raise RuntimeError(
            "Negative cleanliness test failed."
        )

    print(
        "PASS ✓ All negative windows are outside "
        "seizure exclusion intervals."
    )


    # --------------------------------------------------------
    # TEST 8 — No duplicate windows
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 8: DUPLICATE WINDOWS")
    print("-" * 80)

    duplicate_columns = [
        "split",
        "patient",
        "recording",
        "start_sec",
        "end_sec",
        "label",
        "target_seizure_start_sec",
    ]

    duplicates = manifest[
        manifest.duplicated(
            subset=duplicate_columns,
            keep=False,
        )
    ]

    if not duplicates.empty:

        print(
            f"FAIL: {len(duplicates)} duplicate rows detected."
        )

        print(
            duplicates[
                duplicate_columns
            ].head(20)
        )

        raise RuntimeError(
            "Duplicate-window test failed."
        )

    print("PASS ✓ No duplicate windows.")


    # --------------------------------------------------------
    # TEST 9 — EDF paths
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 9: EDF PATHS")
    print("-" * 80)

    path_columns = [
        column
        for column in manifest.columns
        if "path" in column.lower() or "edf" in column.lower()
    ]

    if not path_columns:

        raise RuntimeError(
            "No EDF path column found."
        )

    for column in path_columns:

        missing = manifest[column].isna().sum()

        if missing:

            print(
                f"FAIL: {missing} rows have missing "
                f"EDF path in {column}."
            )

            raise RuntimeError(
                "EDF path test failed."
            )

    print(
        "PASS ✓ Every manifest row has an EDF path."
    )


    # --------------------------------------------------------
    # TEST 10 — Split patient isolation
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 10: PATIENT SPLIT ISOLATION")
    print("-" * 80)

    patients_by_split = (
        manifest
        .groupby("split")["patient"]
        .apply(set)
        .to_dict()
    )

    split_names = sorted(
        patients_by_split.keys()
    )

    split_overlap_found = False

    for i in range(len(split_names)):

        for j in range(i + 1, len(split_names)):

            split_a = split_names[i]
            split_b = split_names[j]

            overlap = (
                patients_by_split[split_a]
                & patients_by_split[split_b]
            )

            if overlap:

                split_overlap_found = True

                print(
                    f"FAIL: {len(overlap)} patients overlap "
                    f"between {split_a} and {split_b}:"
                )

                print(sorted(overlap)[:20])

    if split_overlap_found:

        raise RuntimeError(
            "Patient split isolation test failed."
        )

    print(
        "PASS ✓ No patient occurs across multiple splits."
    )


    # --------------------------------------------------------
    # TEST 11 — Target seizure association
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 11: TARGET SEIZURE ASSOCIATION")
    print("-" * 80)

    missing_target = positive[
        positive["target_seizure_start_sec"].isna()
    ]

    if not missing_target.empty:

        raise RuntimeError(
            f"{len(missing_target)} positive windows "
            "have no target seizure."
        )

    negative_with_target = negative[
        negative["target_seizure_start_sec"].notna()
    ]

    if not negative_with_target.empty:

        raise RuntimeError(
            f"{len(negative_with_target)} negative windows "
            "incorrectly have a target seizure."
        )

    print(
        "PASS ✓ Positive windows have target seizures; "
        "negative windows do not."
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    print("\nTotal windows:")
    print(f"  {len(manifest):,}")

    print("\nLabels:")

    print(
        manifest["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nForecast events:")
    print(f"  {len(positive_events)}")

    print("\nPositive windows/event:")
    print(
        f"  {expected_per_event}"
    )

    print("\nRecordings:")
    print(
        manifest["recording"].nunique()
    )

    print("\nPatients:")

    print(
        manifest["patient"].nunique()
    )

    print("\nSplit distribution:")

    print(
        pd.crosstab(
            manifest["split"],
            manifest["label"],
        )
    )

    print("\n" + "=" * 80)
    print("ALL AUDIT TESTS PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()