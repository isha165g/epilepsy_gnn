import os
import sys

import numpy as np
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from scipy.signal import welch

sys.path.append(os.path.abspath("src"))

from preprocessing.montage import load_and_create_montage
# pyrefly: ignore [missing-import]
from preprocessing.filtering import filter_eeg


EDF_PATH = (
    "data/raw/tusz_edf/"
    "dev/aaaaaewf/s001_2006/02_tcp_le/"
    "aaaaaewf_s001_t001.edf"
)

CHANNEL = "FP1-F7"

# Start 10 minutes into the recording.
# Plot only 10 seconds so the EEG morphology is easy to inspect.
START_SEC = 600
DURATION_SEC = 10


def main():

    print("Loading EEG...")

    data, channel_names, sfreq = load_and_create_montage(
        EDF_PATH,
        target_sfreq=256,
    )

    filtered = filter_eeg(
        data,
        sfreq=sfreq,
        low_freq=0.5,
        high_freq=40.0,
        notch_freq=50.0,
    )

    channel_idx = channel_names.index(CHANNEL)

    start_sample = int(START_SEC * sfreq)
    end_sample = int((START_SEC + DURATION_SEC) * sfreq)

    raw_segment = data[channel_idx, start_sample:end_sample]
    filtered_segment = filtered[channel_idx, start_sample:end_sample]

    time = np.arange(len(raw_segment)) / sfreq

    # ---------------------------------------------------------
    # Plot 1: Raw vs filtered waveform
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        time,
        raw_segment * 1e6,
        label="Before filtering",
    )

    plt.plot(
        time,
        filtered_segment * 1e6,
        label="After filtering",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (µV)")
    plt.title(f"{CHANNEL}: Before vs After Filtering")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("results/preprocessing", exist_ok=True)

    waveform_path = (
        "results/preprocessing/filtering_waveform_comparison.png"
    )

    plt.savefig(waveform_path, dpi=150)
    plt.show()

    print(f"\n✓ Saved waveform plot:")
    print(f"  {waveform_path}")

    # ---------------------------------------------------------
    # Plot 2: Power spectral density
    # ---------------------------------------------------------

    raw_freq, raw_psd = welch(
        raw_segment,
        fs=sfreq,
        nperseg=min(2048, len(raw_segment)),
    )

    filtered_freq, filtered_psd = welch(
        filtered_segment,
        fs=sfreq,
        nperseg=min(2048, len(filtered_segment)),
    )

    plt.figure(figsize=(14, 6))

    plt.semilogy(
        raw_freq,
        raw_psd,
        label="Before filtering",
    )

    plt.semilogy(
        filtered_freq,
        filtered_psd,
        label="After filtering",
    )

    plt.xlim(0, 100)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"{CHANNEL}: Frequency Spectrum")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    spectrum_path = (
        "results/preprocessing/filtering_spectrum_comparison.png"
    )

    plt.savefig(spectrum_path, dpi=150)
    plt.show()

    print(f"\n✓ Saved spectrum plot:")
    print(f"  {spectrum_path}")

    print("\nVisualization complete ✓")


if __name__ == "__main__":
    main()