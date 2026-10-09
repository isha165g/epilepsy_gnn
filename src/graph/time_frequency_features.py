
import numpy as np
from scipy.signal import stft


SFREQ = 256

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def extract_time_frequency_features(
    window,
    sfreq=SFREQ,
):
    """
    Extract compact STFT-based features per EEG channel.

    Parameters
    ----------
    window : ndarray
        Shape (20, 1280), representing 5 seconds of EEG.

    Returns
    -------
    features : ndarray
        Shape (20, 10), float32.

        For each frequency band:
        1. Mean log-power across STFT frames.
        2. Standard deviation of log-power across STFT frames.
    """

    window = np.asarray(window, dtype=np.float64)

    if window.ndim != 2:
        raise ValueError(
            f"Expected a 2D EEG window, got {window.shape}"
        )

    if window.shape != (20, 1280):
        raise ValueError(
            f"Expected shape (20, 1280), got {window.shape}"
        )

    if not np.all(np.isfinite(window)):
        raise ValueError("EEG window contains NaN or Inf.")

    feature_rows = []

    for signal in window:

        frequencies, times, spectrum = stft(
            signal,
            fs=sfreq,
            window="hann",
            nperseg=256,
            noverlap=192,
            nfft=256,
            boundary=None,
            padded=False,
        )

        power = np.abs(spectrum) ** 2

        row = []

        for band_name, (low, high) in FREQUENCY_BANDS.items():

            mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

            if not np.any(mask):
                band_power_per_frame = np.zeros(
                    power.shape[1],
                    dtype=np.float64,
                )
            else:
                band_power_per_frame = np.mean(
                    power[mask, :],
                    axis=0,
                )

            log_power = np.log10(
                band_power_per_frame + 1e-12
            )

            # Average band power over time.
            row.append(np.mean(log_power))

            # Temporal variability of band power.
            row.append(np.std(log_power))

        feature_rows.append(row)

    features = np.asarray(
        feature_rows,
        dtype=np.float32,
    )

    if features.shape != (20, 10):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Time-frequency features contain NaN or Inf."
        )

    return features


if __name__ == "__main__":

    test_window = np.random.default_rng(42).normal(
        size=(20, 1280)
    ).astype(np.float32)

    features = extract_time_frequency_features(
        test_window
    )

    print("=" * 60)
    print("TIME-FREQUENCY FEATURE SMOKE TEST")
    print("=" * 60)
    print("Input shape :", test_window.shape)
    print("Output shape:", features.shape)
    print("Finite      :", np.all(np.isfinite(features)))
    print("Feature count per channel:", features.shape[1])
