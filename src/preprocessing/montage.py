from pathlib import Path

# pyrefly: ignore [missing-import]
import mne
import numpy as np


# Standard 20-channel bipolar montage
BIPOLAR_PAIRS = [
    ("FP1-F7", "FP1", "F7"),
    ("F7-T3", "F7", "T3"),
    ("T3-T5", "T3", "T5"),
    ("T5-O1", "T5", "O1"),

    ("FP2-F8", "FP2", "F8"),
    ("F8-T4", "F8", "T4"),
    ("T4-T6", "T4", "T6"),
    ("T6-O2", "T6", "O2"),

    ("T3-C3", "T3", "C3"),
    ("C3-CZ", "C3", "CZ"),
    ("CZ-C4", "CZ", "C4"),
    ("C4-T4", "C4", "T4"),

    ("FP1-F3", "FP1", "F3"),
    ("F3-C3", "F3", "C3"),
    ("C3-P3", "C3", "P3"),
    ("P3-O1", "P3", "O1"),

    ("FP2-F4", "FP2", "F4"),
    ("F4-C4", "F4", "C4"),
    ("C4-P4", "C4", "P4"),
    ("P4-O2", "P4", "O2"),
]


def normalize_channel_name(channel_name):
    """
    Convert an EDF channel name into a standardized electrode name.

    Examples:
        'EEG FP1-REF' -> 'FP1'
        'EEG FP1-LE'  -> 'FP1'
        'FP1-REF'     -> 'FP1'
    """
    name = channel_name.strip().upper()

    # Remove common prefixes
    name = name.replace("EEG ", "")
    
    # Remove reference suffixes
    for suffix in ["-REF", "-LE"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name


def get_electrode_channels(raw):
    """
    Build a mapping from standardized electrode names to
    their actual MNE channel names.
    """
    mapping = {}

    for channel in raw.ch_names:
        electrode = normalize_channel_name(channel)

        # Keep the first occurrence if duplicate names exist
        if electrode not in mapping:
            mapping[electrode] = channel

    return mapping


def create_bipolar_montage(raw):
    """
    Create the standardized 20-channel bipolar montage.

    Returns
    -------
    data : np.ndarray
        Shape: (20, n_samples)

    channel_names : list[str]
        Standardized bipolar channel names.
    """

    electrode_channels = get_electrode_channels(raw)

    # Check that all electrodes required for the montage exist
    required_electrodes = set()

    for _, first, second in BIPOLAR_PAIRS:
        required_electrodes.add(first)
        required_electrodes.add(second)

    missing = sorted(
        electrode
        for electrode in required_electrodes
        if electrode not in electrode_channels
    )

    if missing:
        raise ValueError(
            f"Missing electrodes: {', '.join(missing)}"
        )

    # Load only the required channels
    actual_channels = [
        electrode_channels[electrode]
        for electrode in sorted(required_electrodes)
    ]

    data = raw.get_data(
        picks=actual_channels
    )

    # Map electrode name -> row in data
    data_by_electrode = {
        electrode: data[i]
        for i, electrode in enumerate(sorted(required_electrodes))
    }

    bipolar_data = []
    channel_names = []

    for name, first, second in BIPOLAR_PAIRS:
        signal = data_by_electrode[first] - data_by_electrode[second]

        bipolar_data.append(signal)
        channel_names.append(name)

    bipolar_data = np.asarray(bipolar_data, dtype=np.float64)

    # Basic numerical validation
    if not np.isfinite(bipolar_data).all():
        raise ValueError("Bipolar montage contains NaN or Inf values.")

    return bipolar_data, channel_names


def load_and_create_montage(edf_path, target_sfreq=256):
    """
    Load an EDF, create the standardized 20-channel
    bipolar montage, and resample to target_sfreq.
    """

    edf_path = Path(edf_path)

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose="ERROR",
    )

    data, channel_names = create_bipolar_montage(raw)

    original_sfreq = raw.info["sfreq"]

    data = resample_data(
        data,
        original_sfreq,
        target_sfreq,
    )

    return data, channel_names, target_sfreq


def resample_data(data, original_sfreq, target_sfreq=256):
    """
    Resample bipolar EEG data to a target sampling frequency.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).

    original_sfreq : float
        Original sampling frequency.

    target_sfreq : float
        Desired sampling frequency.

    Returns
    -------
    resampled_data : np.ndarray
        Resampled EEG data.
    """

    if original_sfreq == target_sfreq:
        return data

    # pyrefly: ignore [missing-import]
    from scipy.signal import resample_poly

    # Find integer ratio for polyphase resampling
    from math import gcd

    original = int(round(original_sfreq))
    target = int(round(target_sfreq))

    divisor = gcd(original, target)

    up = target // divisor
    down = original // divisor

    resampled_data = resample_poly(
        data,
        up=up,
        down=down,
        axis=1,
    )

    return resampled_data


# ============================================================
# Standard 10-20 19-Channel Unipolar Montage (Tang et al. ICLR 2022)
# ============================================================

STANDARD_1020_UNIPOLAR = [
    "FP1", "FP2", "F3", "F4", "C3", "C4",
    "P3", "P4", "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
]

UNIPOLAR_CHANNEL_INDICES = {ch: i for i, ch in enumerate(STANDARD_1020_UNIPOLAR)}


def get_unipolar_to_bipolar_matrix():
    """
    Compute (20, 19) linear projection matrix M mapping standard 19-channel
    unipolar 10-20 EEG to the standardized 20-channel bipolar montage:
        X_bipolar = M @ X_unipolar
    """
    M = np.zeros((len(BIPOLAR_PAIRS), len(STANDARD_1020_UNIPOLAR)), dtype=np.float64)
    for k, (name, first, second) in enumerate(BIPOLAR_PAIRS):
        if first not in UNIPOLAR_CHANNEL_INDICES or second not in UNIPOLAR_CHANNEL_INDICES:
            raise KeyError(f"Electrode {first} or {second} not in STANDARD_1020_UNIPOLAR")
        M[k, UNIPOLAR_CHANNEL_INDICES[first]] = 1.0
        M[k, UNIPOLAR_CHANNEL_INDICES[second]] = -1.0
    return M


def convert_unipolar_to_bipolar(unipolar_data):
    """
    Convert 19-channel unipolar EEG into 20-channel bipolar EEG using linear projection.

    Parameters
    ----------
    unipolar_data : np.ndarray
        Shape (19, n_samples)

    Returns
    -------
    bipolar_data : np.ndarray
        Shape (20, n_samples)
    channel_names : list[str]
        20 standardized bipolar channel names.
    """
    unipolar_data = np.asarray(unipolar_data, dtype=np.float64)
    if unipolar_data.ndim != 2 or unipolar_data.shape[0] != len(STANDARD_1020_UNIPOLAR):
        raise ValueError(
            f"Expected shape (19, n_samples), got {unipolar_data.shape}"
        )
    M = get_unipolar_to_bipolar_matrix()
    bipolar_data = M @ unipolar_data
    channel_names = [name for name, _, _ in BIPOLAR_PAIRS]
    return bipolar_data, channel_names


def create_unipolar_1020_montage(raw, reference="average"):
    """
    Create standardized 19-channel unipolar 10-20 montage (Tang et al. ICLR 2022).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MNE EDF object.
    reference : str
        'average' (common average reference across the 19 electrodes) or 'raw'.

    Returns
    -------
    data : np.ndarray
        Shape: (19, n_samples)
    channel_names : list[str]
        19 unipolar channel names in standard 10-20 order.
    """
    electrode_channels = get_electrode_channels(raw)
    missing = [el for el in STANDARD_1020_UNIPOLAR if el not in electrode_channels]
    if missing:
        raise ValueError(f"Missing required 10-20 electrodes: {', '.join(missing)}")

    actual_channels = [electrode_channels[el] for el in STANDARD_1020_UNIPOLAR]
    data = raw.get_data(picks=actual_channels).astype(np.float64)

    if reference == "average":
        avg_ref = np.mean(data, axis=0, keepdims=True)
        data = data - avg_ref

    if not np.isfinite(data).all():
        raise ValueError("Unipolar montage contains NaN or Inf values.")

    return data, list(STANDARD_1020_UNIPOLAR)


def load_and_create_unipolar_montage(edf_path, target_sfreq=256, reference="average"):
    """
    Load an EDF, create standardized 19-channel unipolar montage, and resample to target_sfreq.
    """
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    data, channel_names = create_unipolar_1020_montage(raw, reference=reference)
    original_sfreq = raw.info["sfreq"]
    data = resample_data(data, original_sfreq, target_sfreq)
    return data, channel_names, target_sfreq

if __name__ == "__main__":
    print("Montage module loaded successfully.")