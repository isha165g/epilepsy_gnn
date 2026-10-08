import numpy as np
# pyrefly: ignore [missing-import]
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).
    sfreq : float
        Sampling frequency in Hz.
    low_freq : float
        Lower cutoff frequency.
    high_freq : float
        Upper cutoff frequency.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered EEG data with the same shape as input.
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape (channels, samples), got {data.shape}"
        )

    nyquist = sfreq / 2

    if not 0 < low_freq < high_freq < nyquist:
        raise ValueError(
            f"Invalid frequency range: {low_freq}-{high_freq} Hz "
            f"for sampling rate {sfreq} Hz"
        )

    sos = butter(
        order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )

    filtered = sosfiltfilt(sos, data, axis=1)

    return filtered


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Apply a zero-phase notch filter.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).
    sfreq : float
        Sampling frequency in Hz.
    notch_freq : float
        Frequency to suppress, typically 50 Hz for Indian recordings.
    quality_factor : float
        Quality factor of the notch filter.

    Returns
    -------
    np.ndarray
        Filtered EEG data with the same shape as input.
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected data with shape (channels, samples), got {data.shape}"
        )

    if not 0 < notch_freq < sfreq / 2:
        raise ValueError(
            f"Notch frequency {notch_freq} Hz must be below "
            f"Nyquist frequency {sfreq / 2} Hz"
        )

    b, a = iirnotch(
        w0=notch_freq,
        Q=quality_factor,
        fs=sfreq,
    )

    filtered = filtfilt(b, a, data, axis=1)

    return filtered


def filter_eeg(
    data: np.ndarray,
    sfreq: float,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """
    Apply the complete EEG filtering pipeline.

    Pipeline:
        1. 0.5-40 Hz band-pass
        2. 50 Hz notch filter
    """

    filtered = bandpass_filter(
        data,
        sfreq=sfreq,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    filtered = notch_filter(
        filtered,
        sfreq=sfreq,
        notch_freq=notch_freq,
    )

    return filtered