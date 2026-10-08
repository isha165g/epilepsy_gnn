from typing import List, Dict


def generate_windows(
    duration_sec: float,
    window_size_sec: int = 5,
) -> List[Dict]:
    """
    Generate non-overlapping windows across a recording.

    Returns dictionaries containing:

        start_sec
        end_sec
    """

    windows = []

    start = 0.0

    while start + window_size_sec <= duration_sec:

        end = start + window_size_sec

        windows.append(
            {
                "start_sec": start,
                "end_sec": end,
            }
        )

        start += window_size_sec

    return windows


def generate_positive_windows(
    seizure_start_sec: float,
    duration_sec: float,
    window_size_sec: int = 5,
    sph_sec: int = 300,
    sop_sec: int = 300,
):
    """
    Generate positive/pre-ictal windows for an eligible seizure.

    Positive interval:

        [T-SPH-SOP, T-SOP]

    With SPH=5 min and SOP=5 min:

        [T-10 min, T-5 min]

    The interval is exactly 300 seconds, producing
    exactly 60 non-overlapping 5-second windows.
    """

    positive_start = seizure_start_sec - sph_sec - sop_sec
    positive_end = seizure_start_sec - sop_sec

    if positive_start < 0:
        return []

    interval_duration = positive_end - positive_start

    # Number of complete windows
    num_windows = int(
        round(interval_duration / window_size_sec)
    )

    windows = []

    for i in range(num_windows):

        start = positive_start + i * window_size_sec
        end = start + window_size_sec

        windows.append(
            {
                "start_sec": start,
                "end_sec": end,
                "label": 1,
            }
        )

    return windows