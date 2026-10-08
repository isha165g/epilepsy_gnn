from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow imports from src/preprocessing/
sys.path.append(str(Path(__file__).resolve().parent))

from montage import create_bipolar_montage, resample_data
from filtering import filter_eeg

# pyrefly: ignore [missing-import]
import mne


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path("data/metadata/window_manifest.csv")

TARGET_SFREQ = 256
WINDOW_SIZE_SEC = 5
CONTEXT_SEC = 30

# Windows with a zero fraction above this are flagged.
ZERO_FRACTION_THRESHOLD = 0.95

# Very small standard deviation → potentially flat signal.
FLAT_STD_THRESHOLD = 1e-10


# ============================================================
# EEG WINDOW EXTRACTION
# ============================================================

def extract_window(
    edf_path,
    start_sec,
    end_sec,
    target_sfreq=TARGET_SFREQ,
    context_sec=CONTEXT_SEC,
):
    """
    Leakage-safe EEG extraction.

    Only data ending at end_sec is loaded.
    A historical context is used for filtering.
    The final 5 seconds are returned.
    """

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose="ERROR",
    )

    original_sfreq = float(raw.info["sfreq"])

    # Create standardized 20-channel bipolar montage
    data, channel_names = create_bipolar_montage(raw)

    # Historical context only
    context_start_sec = max(0.0, start_sec - context_sec)

    native_start = int(round(context_start_sec * original_sfreq))
    native_end = int(round(end_sec * original_sfreq))

    segment = data[:, native_start:native_end]

    if segment.shape[1] == 0:
        raise ValueError("Empty EEG segment.")

    # Resample to 256 Hz
    segment = resample_data(
        segment,
        original_sfreq=original_sfreq,
        target_sfreq=target_sfreq,
    )

    # Filter only historical data
    segment = filter_eeg(
        segment,
        sfreq=target_sfreq,
    )

    expected_samples = int(WINDOW_SIZE_SEC * target_sfreq)

    if segment.shape[1] < expected_samples:
        raise ValueError(
            f"Not enough samples: "
            f"{segment.shape[1]} < {expected_samples}"
        )

    # Keep only requested 5-second window
    window = segment[:, -expected_samples:]

    if window.shape != (20, 1280):
        raise ValueError(
            f"Unexpected shape: {window.shape}"
        )

    return window, channel_names


# ============================================================
# QUALITY METRICS
# ============================================================

def calculate_qc_metrics(window):
    """
    Calculate quality metrics for one EEG window.
    """

    finite_mask = np.isfinite(window)

    nan_count = int(np.isnan(window).sum())
    inf_count = int(np.isinf(window).sum())

    finite_values = window[finite_mask]

    if len(finite_values) == 0:
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
        }

    abs_values = np.abs(finite_values)

    zero_fraction = float(
        np.mean(np.isclose(window, 0.0, atol=1e-12))
    )

    std = float(np.std(finite_values))

    flat = std < FLAT_STD_THRESHOLD

    usable = (
        nan_count == 0
        and inf_count == 0
        and zero_fraction < ZERO_FRACTION_THRESHOLD
        and not flat
    )

    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": std,
        "peak_to_peak": float(
            np.max(finite_values) - np.min(finite_values)
        ),
        "zero_fraction": zero_fraction,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "flat": flat,
        "usable": usable,
    }


# ============================================================
# SELECT REPRESENTATIVE WINDOWS
# ============================================================

def select_test_windows(manifest):
    """
    Select representative windows instead of processing
    the entire dataset.
    """

    selected = []

    # Positive windows
    positive = manifest[manifest["label"] == 1]

    # Negative windows
    negative = manifest[manifest["label"] == 0]

    # Take windows from different positions
    for df in [positive, negative]:

        if len(df) == 0:
            continue

        # Beginning
        selected.append(df.iloc[0])

        # Around 25%
        selected.append(df.iloc[len(df) // 4])

        # Middle
        selected.append(df.iloc[len(df) // 2])

        # Around 75%
        selected.append(df.iloc[(3 * len(df)) // 4])

        # End
        selected.append(df.iloc[-1])

    # Remove duplicate manifest rows
    result = pd.DataFrame(selected).drop_duplicates()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EEG WINDOW QUALITY CONTROL")
    print("=" * 80)

    manifest = pd.read_csv(MANIFEST_PATH)

    print()
    print(f"Manifest rows: {len(manifest):,}")
    print()

    test_windows = select_test_windows(manifest)

    print(f"Selected QC windows: {len(test_windows)}")
    print()

    results = []

    for i, row in test_windows.iterrows():

        print("-" * 80)

        print(
            f"[{len(results) + 1}/{len(test_windows)}] "
            f"{row['split']} | "
            f"{row['patient']} | "
            f"{row['recording']}"
        )

        print(
            f"Window: "
            f"{row['start_sec']:.2f} - "
            f"{row['end_sec']:.2f} sec | "
            f"Label: {row['label']}"
        )

        try:

            window, channels = extract_window(
                row["edf_local"],
                row["start_sec"],
                row["end_sec"],
            )

            metrics = calculate_qc_metrics(window)

            result = {
                "split": row["split"],
                "patient": row["patient"],
                "recording": row["recording"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "label": row["label"],
                **metrics,
            }

            results.append(result)

            print(
                f"Shape        : {window.shape}"
            )
            print(
                f"Min          : {metrics['min']:.6e}"
            )
            print(
                f"Max          : {metrics['max']:.6e}"
            )
            print(
                f"Mean         : {metrics['mean']:.6e}"
            )
            print(
                f"Std          : {metrics['std']:.6e}"
            )
            print(
                f"Peak-to-peak  : {metrics['peak_to_peak']:.6e}"
            )
            print(
                f"Zero fraction : {metrics['zero_fraction']:.4f}"
            )
            print(
                f"NaN           : {metrics['nan_count']}"
            )
            print(
                f"Inf           : {metrics['inf_count']}"
            )
            print(
                f"Flat          : {metrics['flat']}"
            )
            print(
                f"Usable        : {metrics['usable']}"
            )

        except Exception as e:

            print(f"ERROR: {e}")

            results.append({
                "split": row["split"],
                "patient": row["patient"],
                "recording": row["recording"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "label": row["label"],
                "usable": False,
                "error": str(e),
            })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    output_path = Path(
        "data/metadata/window_qc_sample.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("QC SUMMARY")
    print("=" * 80)

    print(
        f"Windows tested : {len(results_df)}"
    )

    if "usable" in results_df:

        print(
            f"Usable         : "
            f"{results_df['usable'].sum()}"
        )

        print(
            f"Flagged        : "
            f"{(~results_df['usable']).sum()}"
        )

    if "label" in results_df:

        print()
        print("By label:")

        print(
            results_df.groupby("label")["usable"]
            .agg(["count", "sum"])
        )

    print()
    print(
        f"Saved results to:\n"
        f"  {output_path}"
    )

    print()
    print("=" * 80)
    print("QC TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
