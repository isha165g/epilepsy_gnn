from dataclasses import dataclass
from typing import List


@dataclass
class Interval:
    start: float
    end: float


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """
    Merge overlapping or touching intervals.
    """

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x.start)

    merged = [intervals[0]]

    for current in intervals[1:]:
        previous = merged[-1]

        if current.start <= previous.end:
            previous.end = max(previous.end, current.end)
        else:
            merged.append(current)

    return merged


def get_positive_interval(
    seizure_start: float,
    sph_sec: int = 300,
    sop_sec: int = 300,
) -> Interval:
    """
    Return the pre-ictal forecasting interval.

    For seizure onset T:

        [T - SPH - SOP, T - SOP]

    With SPH=5 min and SOP=5 min:

        [T-10 min, T-5 min]
    """

    end = seizure_start - sop_sec
    start = end - sph_sec

    return Interval(start=start, end=end)


def get_exclusion_interval(
    seizure_start: float,
    seizure_end: float,
    postictal_sec: int = 600,
    sph_sec: int = 300,
    sop_sec: int = 300,
) -> Interval:
    """
    Return the region that must NOT be used as a negative window.

    Excluded region:

        [T-SPH-SOP, seizure_end + postictal]

    With SPH=5 min, SOP=5 min and postictal=10 min:

        [T-10 min, seizure_end+10 min]
    """

    start = seizure_start - sph_sec - sop_sec
    end = seizure_end + postictal_sec

    return Interval(start=start, end=end)


def is_inside_interval(
    start: float,
    end: float,
    interval: Interval,
) -> bool:
    """
    Check whether a window overlaps an interval.
    """

    return start < interval.end and end > interval.start


def is_valid_negative(
    window_start: float,
    window_end: float,
    exclusion_intervals: List[Interval],
) -> bool:
    """
    A negative window is valid only if it does not overlap
    any seizure-related exclusion interval.
    """

    for interval in exclusion_intervals:
        if is_inside_interval(
            window_start,
            window_end,
            interval,
        ):
            return False

    return True