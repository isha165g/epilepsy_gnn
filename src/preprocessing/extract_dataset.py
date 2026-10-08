from pathlib import Path
import sys

# pyrefly: ignore [missing-import]
import mne
import numpy as np
import pandas as pd


# Allow imports from src/preprocessing/
sys.path.append(str(Path(__file__).resolve().parent))

from montage import create_bipolar_montage, resample_data
from filtering import filter_eeg


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path(
    "data/metadata/window_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

TARGET_SFREQ = 256
WINDOW_SIZE_SEC = 5
CONTEXT_SEC = 30

EXPECTED_CHANNELS = 20
EXPECTED_SAMPLES = TARGET_SFREQ * WINDOW_SIZE_SEC

ZERO_FRACTION_THRESHOLD = 0.95
FLAT_STD_THRESHOLD = 1e-10


# ============================================================
# WINDOW EXTRACTION
# ============================================================

def extract_window(
    raw,
    start_sec,
    end_sec,
):
    """
    Extract one EEG window without using data after end_sec.

    Steps:
        EDF
        -> bipolar montage
        -> historical context
        -> resampling
        -> filtering
        -> final 5-second window
    """

    original_sfreq = float(
        raw.info["sfreq"]
    )

    # --------------------------------------------------------
    # Create 20-channel bipolar montage
    # --------------------------------------------------------

    data, channel_names = create_bipolar_montage(
        raw
    )

    # --------------------------------------------------------
    # Historical context only
    # --------------------------------------------------------

    context_start_sec = max(
        0.0,
        start_sec - CONTEXT_SEC,
    )

    native_start = int(
        round(context_start_sec * original_sfreq)
    )

    native_end = int(
        round(end_sec * original_sfreq)
    )

    segment = data[
        :,
        native_start:native_end
    ]

    if segment.shape[1] == 0:
        raise ValueError(
            "Empty EEG segment."
        )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    segment = resample_data(
        segment,
        original_sfreq=original_sfreq,
        target_sfreq=TARGET_SFREQ,
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    segment = filter_eeg(
        segment,
        sfreq=TARGET_SFREQ,
    )

    # --------------------------------------------------------
    # Extract final 5 seconds
    # --------------------------------------------------------

    if segment.shape[1] < EXPECTED_SAMPLES:
        raise ValueError(
            f"Insufficient samples after preprocessing: "
            f"{segment.shape[1]} < {EXPECTED_SAMPLES}"
        )

    window = segment[
        :,
        -EXPECTED_SAMPLES:
    ]

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    if window.shape != (
        EXPECTED_CHANNELS,
        EXPECTED_SAMPLES,
    ):
        raise ValueError(
            f"Unexpected window shape: "
            f"{window.shape}"
        )

    return window, channel_names


# ============================================================
# QUALITY CONTROL
# ============================================================

def calculate_qc(window):
    """
    Calculate window-level signal quality metrics.
    """

    nan_count = int(
        np.isnan(window).sum()
    )

    inf_count = int(
        np.isinf(window).sum()
    )

    finite = np.isfinite(window)

    if not finite.any():
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "peak_to_peak": np.nan,
            "zero_fraction": 1.0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "flat": True,
            "usable": False,
            "qc_reason": "no_finite_values",
        }

    values = window[finite]

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    peak_to_peak = float(
        maximum - minimum
    )

    zero_fraction = float(
        np.mean(
            np.isclose(
                window,
                0.0,
                atol=1e-12,
            )
        )
    )

    flat = (
        std < FLAT_STD_THRESHOLD
    )

    # --------------------------------------------------------
    # Determine usability
    # --------------------------------------------------------

    reasons = []

    if nan_count > 0:
        reasons.append("nan")

    if inf_count > 0:
        reasons.append("inf")

    if zero_fraction >= ZERO_FRACTION_THRESHOLD:
        reasons.append("mostly_zero")

    if flat:
        reasons.append("flat")

    usable = len(reasons) == 0

    qc_reason = (
        "ok"
        if usable
        else ";".join(reasons)
    )

    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "peak_to_peak": peak_to_peak,
        "zero_fraction": zero_fraction,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "flat": flat,
        "usable": usable,
        "qc_reason": qc_reason,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 80)
    print("FULL EEG WINDOW EXTRACTION PIPELINE")
    print("=" * 80)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print()
    print(
        f"Manifest windows : "
        f"{len(manifest):,}"
    )

    print(
        f"Recordings       : "
        f"{manifest['recording'].nunique():,}"
    )

    print(
        f"Patients          : "
        f"{manifest['patient'].nunique():,}"
    )

    print()

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    for split in ["train", "dev", "eval"]:
        (
            OUTPUT_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    qc_results = []

    # --------------------------------------------------------
    # Cache currently loaded EDF
    # --------------------------------------------------------

    current_edf = None
    current_raw = None

    total = len(manifest)

    for index, row in manifest.iterrows():

        split = row["split"]
        patient = row["patient"]
        recording = row["recording"]

        edf_path = Path(
            row["edf_local"]
        )

        start_sec = float(
            row["start_sec"]
        )

        end_sec = float(
            row["end_sec"]
        )

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # Load EDF only when recording changes
        # ----------------------------------------------------

        if (
            current_edf is None
            or edf_path != current_edf
        ):

            if current_raw is not None:
                current_raw.close()

            print()
            print(
                f"Loading EDF: "
                f"{recording}"
            )

            current_raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                verbose="ERROR",
            )

            current_edf = edf_path

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        try:

            window, channel_names = extract_window(
                current_raw,
                start_sec,
                end_sec,
            )

            qc = calculate_qc(
                window
            )

            qc_row = {
                "manifest_index": index,
                "split": split,
                "patient": patient,
                "recording": recording,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
                **qc,
            }

            qc_results.append(
                qc_row
            )

            # ------------------------------------------------
            # Save only usable windows
            # ------------------------------------------------

            if qc["usable"]:

                filename = (
                    f"{index:06d}.npy"
                )

                output_path = (
                    OUTPUT_DIR
                    / split
                    / filename
                )

                np.save(
                    output_path,
                    window.astype(
                        np.float32
                    ),
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (index + 1) % 100 == 0
                or index == 0
                or index == total - 1
            ):

                usable_count = sum(
                    r["usable"]
                    for r in qc_results
                )

                print(
                    f"[{index + 1:5d}/{total}] "
                    f"usable={usable_count:5d} "
                    f"current_label={label}"
                )

        except Exception as e:

            print(
                f"\nERROR at manifest index "
                f"{index}: {e}"
            )

            qc_results.append({
                "manifest_index": index,
                "split": split,
                "patient": patient,
                "recording": recording,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": label,
                "usable": False,
                "qc_reason": "extraction_error",
                "error": str(e),
            })

    # --------------------------------------------------------
    # Close EDF
    # --------------------------------------------------------

    if current_raw is not None:
        current_raw.close()

    # --------------------------------------------------------
    # Save QC results
    # --------------------------------------------------------

    qc_df = pd.DataFrame(
        qc_results
    )

    qc_path = (
        Path("data/metadata")
        / "window_qc_full.csv"
    )

    qc_df.to_csv(
        qc_path,
        index=False,
    )

    # --------------------------------------------------------
    # Create processed manifest
    # --------------------------------------------------------

    processed_manifest = (
        manifest
        .copy()
    )

    processed_manifest[
        "usable"
    ] = False

    processed_manifest[
        "qc_reason"
    ] = "unknown"

    for result in qc_results:

        idx = result[
            "manifest_index"
        ]

        processed_manifest.loc[
            idx,
            "usable"
        ] = result["usable"]

        processed_manifest.loc[
            idx,
            "qc_reason"
        ] = result[
            "qc_reason"
        ]

    processed_manifest[
        "processed_path"
    ] = processed_manifest.apply(
        lambda row: (
            str(
                OUTPUT_DIR
                / row["split"]
                / f"{int(row.name):06d}.npy"
            )
            if row["usable"]
            else ""
        ),
        axis=1,
    )

    processed_manifest_path = (
        Path("data/metadata")
        / "processed_window_manifest.csv"
    )

    processed_manifest.to_csv(
        processed_manifest_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    total_windows = len(qc_df)

    usable_windows = int(
        qc_df["usable"].sum()
    )

    rejected_windows = (
        total_windows
        - usable_windows
    )

    print(
        f"Total windows    : "
        f"{total_windows:,}"
    )

    print(
        f"Usable windows   : "
        f"{usable_windows:,}"
    )

    print(
        f"Rejected windows : "
        f"{rejected_windows:,}"
    )

    print()
    print("By split:")

    summary = (
        qc_df
        .groupby("split")
        .agg(
            total=("usable", "size"),
            usable=("usable", "sum"),
        )
    )

    summary["rejected"] = (
        summary["total"]
        - summary["usable"]
    )

    print(summary)

    print()
    print("By label:")

    label_summary = (
        qc_df
        .groupby("label")
        .agg(
            total=("usable", "size"),
            usable=("usable", "sum"),
        )
    )

    label_summary["rejected"] = (
        label_summary["total"]
        - label_summary["usable"]
    )

    print(label_summary)

    print()
    print("QC reasons:")

    print(
        qc_df[
            "qc_reason"
        ].value_counts()
    )

    print()
    print(
        f"QC results saved to:\n"
        f"  {qc_path}"
    )

    print(
        f"\nProcessed manifest saved to:\n"
        f"  {processed_manifest_path}"
    )

    print()
    print(
        f"Processed EEG saved under:\n"
        f"  {OUTPUT_DIR}/"
    )

    print()
    print("=" * 80)
    print("FULL EXTRACTION COMPLETE ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()