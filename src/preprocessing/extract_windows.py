from pathlib import Path

# pyrefly: ignore [missing-import]
import mne
import numpy as np
import pandas as pd

from montage import create_bipolar_montage, resample_data
from filtering import filter_eeg


MANIFEST_PATH = Path("data/metadata/window_manifest.csv")

TARGET_SFREQ = 256
WINDOW_SIZE_SEC = 5

# Give zero-phase filtering some historical context,
# but never use data after the prediction time.
CONTEXT_SEC = 30


def extract_window(
    edf_path,
    start_sec,
    end_sec,
    target_sfreq=TARGET_SFREQ,
    context_sec=CONTEXT_SEC,
):
    """
    Extract one leakage-safe 5-second EEG window.

    Only EEG occurring at or before end_sec is used.

    Returns
    -------
    window : np.ndarray
        Shape: (20, 1280)

    channel_names : list[str]

    sfreq : float
        Target sampling frequency.
    """

    # ----------------------------------------------------
    # 1. Load EDF without temporal preprocessing
    # ----------------------------------------------------

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose="ERROR",
    )

    original_sfreq = float(raw.info["sfreq"])

    # ----------------------------------------------------
    # 2. Create bipolar montage at native sampling rate
    # ----------------------------------------------------

    data, channel_names = create_bipolar_montage(raw)

    # ----------------------------------------------------
    # 3. Select historical context
    #
    # Never use anything after end_sec.
    # ----------------------------------------------------

    context_start_sec = max(
        0.0,
        start_sec - context_sec,
    )

    native_start = int(
        round(context_start_sec * original_sfreq)
    )

    native_end = int(
        round(end_sec * original_sfreq)
    )

    segment = data[:, native_start:native_end]

    if segment.shape[1] == 0:
        raise ValueError(
            f"Empty EEG segment for "
            f"{start_sec}-{end_sec}s"
        )

    # ----------------------------------------------------
    # 4. Resample only the historical segment
    # ----------------------------------------------------

    segment = resample_data(
        segment,
        original_sfreq=original_sfreq,
        target_sfreq=target_sfreq,
    )

    # ----------------------------------------------------
    # 5. Filter only data available by prediction time
    # ----------------------------------------------------

    segment = filter_eeg(
        segment,
        sfreq=target_sfreq,
    )

    # ----------------------------------------------------
    # 6. Keep final 5 seconds
    # ----------------------------------------------------

    expected_samples = int(
        WINDOW_SIZE_SEC * target_sfreq
    )

    if segment.shape[1] < expected_samples:
        raise ValueError(
            f"Not enough samples after preprocessing. "
            f"Got {segment.shape[1]}, "
            f"need {expected_samples}."
        )

    window = segment[:, -expected_samples:]

    # ----------------------------------------------------
    # 7. Validation
    # ----------------------------------------------------

    expected_shape = (
        20,
        expected_samples,
    )

    if window.shape != expected_shape:
        raise ValueError(
            f"Unexpected window shape: "
            f"{window.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(window).all():
        raise ValueError(
            "Window contains NaN or Inf."
        )

    return window, channel_names, target_sfreq


def main():

    manifest = pd.read_csv(MANIFEST_PATH)

    print("=" * 80)
    print("LEAKAGE-SAFE EEG WINDOW EXTRACTION TEST")
    print("=" * 80)

    row = manifest[(manifest["label"] == 1) & (manifest["start_sec"] >= 30)].iloc[0]

    print("\nSelected window:")
    print(f"  Split     : {row['split']}")
    print(f"  Patient   : {row['patient']}")
    print(f"  Recording : {row['recording']}")
    print(f"  Start     : {row['start_sec']} sec")
    print(f"  End       : {row['end_sec']} sec")
    print(f"  Label     : {row['label']}")

    window, channel_names, sfreq = extract_window(
        row["edf_local"],
        row["start_sec"],
        row["end_sec"],
    )

    print("\nExtracted EEG:")
    print(f"  Shape       : {window.shape}")
    print(f"  Sampling Hz : {sfreq}")
    print(f"  Channels    : {len(channel_names)}")
    print(f"  Min         : {window.min():.8f}")
    print(f"  Max         : {window.max():.8f}")
    print(f"  Mean        : {window.mean():.8f}")
    print(f"  Std         : {window.std():.8f}")
    print(f"  NaN         : {np.isnan(window).sum()}")
    print(f"  Inf         : {np.isinf(window).sum()}")

    print("\nChannels:")

    for i, channel in enumerate(channel_names):
        print(
            f"  {i:02d}: {channel}"
        )

    print("\n" + "=" * 80)
    print("EXTRACTION TEST PASSED ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()