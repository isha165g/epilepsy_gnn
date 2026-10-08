import os
import sys
import numpy as np

# Allow imports from src/
sys.path.append(os.path.abspath("src"))

from preprocessing.montage import load_and_create_montage
# pyrefly: ignore [missing-import]
from preprocessing.filtering import filter_eeg


def test_filtering(edf_path):
    print("\n" + "=" * 70)
    print(f"Testing: {os.path.basename(edf_path)}")
    print("=" * 70)

    # Load bipolar montage and resample to 256 Hz
    data, channel_names, sfreq = load_and_create_montage(
        edf_path,
        target_sfreq=256,
    )

    print(f"Input shape       : {data.shape}")
    print(f"Sampling rate     : {sfreq} Hz")

    print("\nBefore filtering:")
    print(f"Min               : {data.min():.6f}")
    print(f"Max               : {data.max():.6f}")
    print(f"Mean              : {data.mean():.6f}")
    print(f"Std               : {data.std():.6f}")

    # Apply filtering
    filtered = filter_eeg(
        data,
        sfreq=sfreq,
        low_freq=0.5,
        high_freq=40.0,
        notch_freq=50.0,
    )

    print("\nAfter filtering:")
    print(f"Min               : {filtered.min():.6f}")
    print(f"Max               : {filtered.max():.6f}")
    print(f"Mean              : {filtered.mean():.6f}")
    print(f"Std               : {filtered.std():.6f}")

    # Validation
    assert filtered.shape == data.shape
    print("\n✓ Shape preserved")

    assert sfreq == 256
    print("✓ Sampling rate = 256 Hz")

    assert np.isfinite(filtered).all()
    print("✓ No NaN/Inf")

    assert np.isrealobj(filtered)
    print("✓ Real-valued data")

    print("\nFiltering test PASSED ✓")


# ---------------------------------------------------------
# Test recordings
# ---------------------------------------------------------

edf_250 = (
    "data/raw/tusz_edf/"
    "dev/aaaaaewf/s001_2006/02_tcp_le/"
    "aaaaaewf_s001_t001.edf"
)

edf_256 = (
    "data/raw/tusz_edf/"
    "eval/aaaaasip/s004_2015/01_tcp_ar/"
    "aaaaasip_s004_t000.edf"
)


if __name__ == "__main__":
    print("\nTesting 250 Hz → 256 Hz → filtering...")
    test_filtering(edf_250)

    print("\nTesting 256 Hz → filtering...")
    test_filtering(edf_256)