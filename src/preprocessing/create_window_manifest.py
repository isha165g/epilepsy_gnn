import os
import sys

import pandas as pd

# Allow imports from src/
sys.path.append(os.path.abspath("src"))

# pyrefly: ignore [missing-import]
from preprocessing.labels import (
    get_exclusion_interval,
    get_positive_interval,
    is_valid_negative,
)

# pyrefly: ignore [missing-import]
from preprocessing.windowing import (
    generate_positive_windows,
    generate_windows,
)


# ============================================================
# Configuration
# ============================================================

WINDOW_SIZE_SEC = 5

SPH_SEC = 5 * 60
SOP_SEC = 5 * 60

# Conservative postictal exclusion
POSTICTAL_SEC = 10 * 60

METADATA_PATH = "data/metadata/recordings_metadata.csv"
FORECASTING_PATH = "data/metadata/forecasting_events_primary.csv"
EDF_PATHS_PATH = "data/metadata/eligible_edfs.csv"

OUTPUT_PATH = "data/metadata/window_manifest.csv"


# ============================================================
# Helpers
# ============================================================

def recording_key(row):
    """
    Create a unique recording identifier.

    Split + patient + recording should uniquely identify
    a TUSZ recording.
    """

    return (
        str(row["split"]),
        str(row["patient"]),
        str(row["recording"]),
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("CREATING ACTUAL TUSZ WINDOW MANIFEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)
    forecasting = pd.read_csv(FORECASTING_PATH)
    edf_paths = pd.read_csv(EDF_PATHS_PATH)

    print("\nLoaded:")
    print(f"All seizure events      : {len(metadata)}")
    print(f"Eligible seizure events : {len(forecasting)}")
    print(f"Eligible recordings     : {len(edf_paths)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_metadata = {
        "split",
        "patient",
        "recording",
        "duration_sec",
        "seizure_start_sec",
        "seizure_end_sec",
    }

    required_forecasting = {
        "split",
        "patient",
        "recording",
        "seizure_start_sec",
    }

    missing_metadata = required_metadata - set(metadata.columns)
    missing_forecasting = required_forecasting - set(forecasting.columns)

    if missing_metadata:
        raise ValueError(
            f"Missing columns in recordings_metadata.csv: "
            f"{sorted(missing_metadata)}"
        )

    if missing_forecasting:
        raise ValueError(
            f"Missing columns in forecasting_events_primary.csv: "
            f"{sorted(missing_forecasting)}"
        )

    # --------------------------------------------------------
    # Create recording keys
    # --------------------------------------------------------

    metadata["recording_key"] = metadata.apply(
        recording_key,
        axis=1,
    )

    forecasting["recording_key"] = forecasting.apply(
        recording_key,
        axis=1,
    )

    edf_paths["recording_key"] = edf_paths.apply(
        recording_key,
        axis=1,
    )

    # --------------------------------------------------------
    # Restrict metadata to our 88 eligible recordings
    # --------------------------------------------------------

    eligible_keys = set(edf_paths["recording_key"])

    metadata = metadata[
        metadata["recording_key"].isin(eligible_keys)
    ].copy()

    forecasting = forecasting[
        forecasting["recording_key"].isin(eligible_keys)
    ].copy()

    print("\nAfter restricting to eligible recordings:")
    print(f"Seizure events : {len(metadata)}")
    print(f"Forecast events: {len(forecasting)}")

    # --------------------------------------------------------
    # Group all seizures by recording
    # --------------------------------------------------------

    seizures_by_recording = {}

    for key, group in metadata.groupby("recording_key"):

        seizures = []

        for _, row in group.iterrows():

            seizures.append(
                {
                    "start": float(row["seizure_start_sec"]),
                    "end": float(row["seizure_end_sec"]),
                }
            )

        seizures_by_recording[key] = seizures

    # --------------------------------------------------------
    # Identify eligible seizure events
    # --------------------------------------------------------

    positive_events = {}

    for key, group in forecasting.groupby("recording_key"):

        positive_events[key] = [
            float(x)
            for x in group["seizure_start_sec"]
        ]

    # --------------------------------------------------------
    # Generate windows
    # --------------------------------------------------------

    rows = []

    positive_count = 0
    negative_count = 0

    recordings_processed = 0

    print("\nGenerating windows...")

    for key in sorted(eligible_keys):

        split, patient, recording = key

        # ----------------------------------------------------
        # Get recording duration
        # ----------------------------------------------------

        recording_metadata = metadata[
            metadata["recording_key"] == key
        ]

        if recording_metadata.empty:
            print(
                f"WARNING: No metadata found for {key}"
            )
            continue

        duration_sec = float(
            recording_metadata["duration_sec"].iloc[0]
        )

        # ----------------------------------------------------
        # All seizure exclusion intervals
        # ----------------------------------------------------

        exclusion_intervals = []

        for seizure in seizures_by_recording.get(key, []):

            interval = get_exclusion_interval(
                seizure_start=seizure["start"],
                seizure_end=seizure["end"],
                postictal_sec=POSTICTAL_SEC,
                sph_sec=SPH_SEC,
                sop_sec=SOP_SEC,
            )

            # Clip to recording boundaries
            interval.start = max(
                0.0,
                interval.start,
            )

            interval.end = min(
                duration_sec,
                interval.end,
            )

            exclusion_intervals.append(interval)

        # ----------------------------------------------------
        # Generate POSITIVE windows
        #
        # These are anchored directly to each forecast
        # interval, rather than the recording-wide 5-sec grid.
        # ----------------------------------------------------

        recording_positive_windows = []

        for seizure_start in positive_events.get(key, []):

            positive_interval = get_positive_interval(
                seizure_start=seizure_start,
                sph_sec=SPH_SEC,
                sop_sec=SOP_SEC,
            )

            if positive_interval.start < 0:
                raise ValueError(
                    f"Positive interval begins before recording: "
                    f"{key}, seizure={seizure_start}"
                )

            positive_windows = generate_positive_windows(
                seizure_start_sec=seizure_start,
                duration_sec=duration_sec,
                window_size_sec=WINDOW_SIZE_SEC,
                sph_sec=SPH_SEC,
                sop_sec=SOP_SEC,
            )

            # Every eligible event must generate exactly 60 windows
            expected_windows = SPH_SEC // WINDOW_SIZE_SEC

            if len(positive_windows) != expected_windows:
                raise RuntimeError(
                    f"Positive-window generation error for "
                    f"{key}, seizure={seizure_start}: "
                    f"expected {expected_windows}, "
                    f"got {len(positive_windows)}"
                )

            for window in positive_windows:

                rows.append(
                    {
                        "split": split,
                        "patient": patient,
                        "recording": recording,
                        "start_sec": float(window["start_sec"]),
                        "end_sec": float(window["end_sec"]),
                        "duration_sec": duration_sec,
                        "label": 1,
                        "label_type": "preictal",
                        "target_seizure_start_sec": seizure_start,
                    }
                )

                recording_positive_windows.append(
                    (
                        float(window["start_sec"]),
                        float(window["end_sec"]),
                    )
                )

                positive_count += 1

        # ----------------------------------------------------
        # Generate recording-wide windows
        #
        # These are candidates for negative/interictal samples.
        # ----------------------------------------------------

        windows = generate_windows(
            duration_sec=duration_sec,
            window_size_sec=WINDOW_SIZE_SEC,
        )

        for window in windows:

            start = float(window["start_sec"])
            end = float(window["end_sec"])

            # ------------------------------------------------
            # Never use a window that overlaps a positive
            # forecasting interval.
            # ------------------------------------------------

            overlaps_positive = False

            for positive_start, positive_end in recording_positive_windows:

                if (
                    start < positive_end
                    and end > positive_start
                ):
                    overlaps_positive = True
                    break

            if overlaps_positive:
                continue

            # ------------------------------------------------
            # Check whether this is a valid negative window.
            # ------------------------------------------------

            if is_valid_negative(
                start,
                end,
                exclusion_intervals,
            ):

                rows.append(
                    {
                        "split": split,
                        "patient": patient,
                        "recording": recording,
                        "start_sec": start,
                        "end_sec": end,
                        "duration_sec": duration_sec,
                        "label": 0,
                        "label_type": "interictal",
                        "target_seizure_start_sec": None,
                    }
                )

                negative_count += 1

        recordings_processed += 1

        if recordings_processed % 10 == 0:
            print(
                f"Processed {recordings_processed}/"
                f"{len(eligible_keys)} recordings..."
            )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    manifest = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Sanity check: every eligible forecasting event should
    # produce exactly 60 positive windows.
    # --------------------------------------------------------

    expected_positive_windows = len(forecasting) * (
        SPH_SEC // WINDOW_SIZE_SEC
    )

    actual_positive_windows = (
        manifest["label"] == 1
    ).sum()

    if actual_positive_windows != expected_positive_windows:
        raise RuntimeError(
            f"Positive-window count mismatch: "
            f"expected {expected_positive_windows}, "
            f"got {actual_positive_windows}"
        )

    print(
        f"\n✓ Positive-window count verified: "
        f"{actual_positive_windows}"
    )

    # --------------------------------------------------------
    # Verify each forecasting event has exactly 60 windows
    # --------------------------------------------------------

    positive_manifest = manifest[
        manifest["label"] == 1
    ]

    windows_per_event = (
        positive_manifest
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

    incorrect_events = windows_per_event[
        windows_per_event != (SPH_SEC // WINDOW_SIZE_SEC)
    ]

    if not incorrect_events.empty:

        print("\nIncorrect positive-window counts:")
        print(incorrect_events)

        raise RuntimeError(
            "Some forecasting events do not have exactly "
            "60 positive windows."
        )

    print(
        "✓ Every forecasting event has exactly "
        f"{SPH_SEC // WINDOW_SIZE_SEC} positive windows"
    )

    # --------------------------------------------------------
    # Add EDF paths
    # --------------------------------------------------------

    path_columns = [
        "split",
        "patient",
        "recording",
    ]

    path_table = edf_paths[
        path_columns + ["edf_local"]
    ].copy()

    manifest = manifest.merge(
        path_table,
        on=path_columns,
        how="left",
    )

    # --------------------------------------------------------
    # Verify EDF paths
    # --------------------------------------------------------

    if "edf_local" not in manifest.columns:
        raise RuntimeError(
            "edf_local column was not added to the manifest."
        )

    missing_paths = manifest["edf_local"].isna().sum()

    if missing_paths > 0:
        raise RuntimeError(
            f"{missing_paths} manifest rows have no EDF local path."
        )

    print(
        "✓ Every manifest row has an EDF local path."
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    manifest = manifest.sort_values(
        [
            "split",
            "patient",
            "recording",
            "start_sec",
            "label",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("WINDOW MANIFEST CREATED")
    print("=" * 80)

    print(f"\nOutput:")
    print(f"  {OUTPUT_PATH}")

    print("\nTotal windows:")
    print(f"  {len(manifest):,}")

    print("\nPositive windows:")
    print(f"  {positive_count:,}")

    print("\nNegative windows:")
    print(f"  {negative_count:,}")

    print("\nLabels:")

    counts = manifest["label"].value_counts().sort_index()

    for label, count in counts.items():

        percentage = (
            count / len(manifest) * 100
        )

        print(
            f"  {label}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nLabel types:")

    print(
        manifest["label_type"]
        .value_counts()
        .to_string()
    )

    print("\nWindows by split:")

    split_table = pd.crosstab(
        manifest["split"],
        manifest["label"],
    )

    print(split_table)

    print("\nRecordings by split:")

    print(
        manifest.groupby("split")["recording"]
        .nunique()
        .to_string()
    )

    print("\nPatients by split:")

    print(
        manifest.groupby("split")["patient"]
        .nunique()
        .to_string()
    )

    print("\nPositive windows per forecasting event:")

    print(
        windows_per_event.describe()
    )

    print("\nWindows per recording:")

    recording_counts = (
        manifest
        .groupby(
            ["split", "recording", "label"]
        )
        .size()
        .unstack(fill_value=0)
    )

    print(
        recording_counts.describe()
    )

    print("\n" + "=" * 80)
    print("DONE ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()