import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from scipy.signal import welch


# ============================================================
# Configuration
# ============================================================

SFREQ = 256

FREQUENCY_BANDS = {
    "delta_power": (0.5, 4.0),
    "theta_power": (4.0, 8.0),
    "alpha_power": (8.0, 13.0),
    "beta_power": (13.0, 30.0),
    "gamma_power": (30.0, 40.0),
}


# ============================================================
# Time-domain features
# ============================================================

def mean_feature(signal):
    return np.mean(signal)


def std_feature(signal):
    return np.std(signal)


def rms_feature(signal):
    return np.sqrt(np.mean(signal ** 2))


def peak_to_peak_feature(signal):
    return np.ptp(signal)


def line_length_feature(signal):
    return np.sum(np.abs(np.diff(signal)))


# ============================================================
# Frequency-domain features
# ============================================================

def compute_psd(signal, sfreq=SFREQ):
    """
    Compute power spectral density using Welch's method.
    """

    frequencies, power = welch(
        signal,
        fs=sfreq,
        nperseg=min(512, len(signal)),
        noverlap=None,
    )

    return frequencies, power

def spectral_entropy(signal, sfreq=SFREQ):
    """
    Calculate normalized spectral entropy.

    Spectral entropy measures how distributed the signal's
    power is across frequencies.

    Lower entropy:
        Power concentrated in fewer frequencies.

    Higher entropy:
        Power distributed across more frequencies.
    """

    frequencies, power = compute_psd(
        signal,
        sfreq=sfreq
    )

    # Restrict to our EEG frequency range
    mask = (
        (frequencies >= 0.5)
        & (frequencies < 40.0)
    )

    power = power[mask]

    total_power = np.sum(power)

    if total_power <= 0:
        return 0.0

    # Convert PSD into a probability distribution
    probabilities = power / total_power

    # Avoid log(0)
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    # Normalize to [0, 1]
    max_entropy = np.log2(len(power))

    if max_entropy > 0:
        entropy /= max_entropy

    return entropy

def band_power(signal, sfreq, low_freq, high_freq):
    """
    Calculate absolute power within a frequency band.
    """

    frequencies, power = compute_psd(
        signal,
        sfreq=sfreq
    )

    mask = (
        (frequencies >= low_freq)
        & (frequencies < high_freq)
    )

    if not np.any(mask):
        return 0.0

    return np.trapezoid(
        power[mask],
        frequencies[mask]
    )


def extract_frequency_features(signal, sfreq=SFREQ):
    """
    Calculate absolute power for all frequency bands.
    """

    features = {}

    for band_name, (low, high) in FREQUENCY_BANDS.items():

        features[band_name] = band_power(
            signal,
            sfreq,
            low,
            high
        )

    return features


def extract_relative_frequency_features(
    absolute_features
):
    """
    Convert absolute band powers into relative powers.
    """

    total_power = sum(
        absolute_features.values()
    )

    relative_features = {}

    for name, value in absolute_features.items():

        relative_name = name.replace(
            "_power",
            "_relative_power"
        )

        if total_power > 0:
            relative_features[relative_name] = (
                value / total_power
            )
        else:
            relative_features[relative_name] = 0.0

    return relative_features


# ============================================================
# Node feature extraction
# ============================================================

def extract_node_features(
    window,
    sfreq=SFREQ
):
    """
    Convert one EEG window into a node feature matrix.

    Parameters
    ----------
    window : np.ndarray
        Shape: (20, samples)

    sfreq : float
        Sampling frequency.

    Returns
    -------
    features : np.ndarray
        Shape: (20, 15)

    feature_names : list
        Names of the 15 features.
    """

    if window.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {window.ndim}D"
        )

    num_channels = window.shape[0]

    feature_rows = []

    for channel_idx in range(num_channels):

        signal = window[channel_idx].astype(
            np.float64
        )

        # ------------------------------
        # Time-domain
        # ------------------------------

        row = {
            "mean": mean_feature(signal),
            "std": std_feature(signal),
            "rms": rms_feature(signal),
            "peak_to_peak": peak_to_peak_feature(signal),
            "line_length": line_length_feature(signal),
            "spectral_entropy": spectral_entropy(
                signal,
                sfreq
            ),
        }

        # ------------------------------
        # Frequency-domain
        # ------------------------------

        absolute_features = (
            extract_frequency_features(
                signal,
                sfreq
            )
        )

        row.update(absolute_features)

        # ------------------------------
        # Relative frequency-domain
        # ------------------------------

        relative_features = (
            extract_relative_frequency_features(
                absolute_features
            )
        )

        row.update(relative_features)

        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)

    return (
        feature_df.to_numpy(dtype=np.float32),
        list(feature_df.columns)
    )

# ============================================================
# Advanced Morphological Features (Hjorth, ZCR, Energy)
# ============================================================

def zero_crossing_rate(signal):
    """Compute normalized zero-crossing rate."""
    zero_crosses = np.nonzero(np.diff(signal > 0))[0]
    return float(len(zero_crosses) / max(1, len(signal) - 1))


def energy_feature(signal):
    """Compute total signal energy."""
    return float(np.sum(signal ** 2))


def hjorth_parameters(signal):
    """
    Compute Hjorth Activity, Mobility, and Complexity.

    Activity   = var(signal)
    Mobility   = sqrt(var(diff(signal)) / var(signal))
    Complexity = Mobility(diff(signal)) / Mobility(signal)
    """
    variance = float(np.var(signal))
    if variance < 1e-12:
        return 0.0, 0.0, 0.0

    d1 = np.diff(signal)
    var_d1 = float(np.var(d1))
    if var_d1 < 1e-12:
        return variance, 0.0, 0.0

    mobility = np.sqrt(var_d1 / variance)

    d2 = np.diff(d1)
    var_d2 = float(np.var(d2))
    if var_d2 < 1e-12:
        return variance, float(mobility), 0.0

    mobility_d1 = np.sqrt(var_d2 / var_d1)
    complexity = float(mobility_d1 / max(1e-12, mobility))

    return variance, float(mobility), float(complexity)


def extract_multimodal_streams(signal, sfreq=SFREQ):
    """
    Extract strictly separated Time-Domain Morphological Stream (x_time)
    and Frequency-Domain Spectral Stream (x_freq).

    Returns
    -------
    x_time : np.ndarray
        Shape (8,): [mean, std, rms, peak_to_peak, line_length, hjorth_activity, hjorth_mobility, hjorth_complexity]
    x_freq : np.ndarray
        Shape (11,): [spectral_entropy, 5 abs band powers, 5 rel band powers]
    feature_names : tuple of (list[str], list[str])
    """
    var, mob, comp = hjorth_parameters(signal)
    time_feats = [
        mean_feature(signal),
        std_feature(signal),
        rms_feature(signal),
        peak_to_peak_feature(signal),
        line_length_feature(signal),
        var,
        mob,
        comp,
    ]
    time_names = [
        "mean", "std", "rms", "peak_to_peak", "line_length",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity"
    ]

    abs_f = extract_frequency_features(signal, sfreq)
    rel_f = extract_relative_frequency_features(abs_f)
    freq_feats = [
        spectral_entropy(signal, sfreq),
        abs_f["delta_power"], abs_f["theta_power"], abs_f["alpha_power"],
        abs_f["beta_power"], abs_f["gamma_power"],
        rel_f["delta_relative_power"], rel_f["theta_relative_power"], rel_f["alpha_relative_power"],
        rel_f["beta_relative_power"], rel_f["gamma_relative_power"],
    ]
    freq_names = [
        "spectral_entropy",
        "abs_delta", "abs_theta", "abs_alpha", "abs_beta", "abs_gamma",
        "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma"
    ]

    return (
        np.asarray(time_feats, dtype=np.float32),
        np.asarray(freq_feats, dtype=np.float32),
        (time_names, freq_names),
    )
