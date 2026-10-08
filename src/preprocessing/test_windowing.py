import os
import sys

sys.path.append(os.path.abspath("src"))

# pyrefly: ignore [missing-import]
from preprocessing.windowing import generate_positive_windows
# pyrefly: ignore [missing-import]
from preprocessing.labels import (
    get_positive_interval,
    get_exclusion_interval,
    is_valid_negative,
)


def main():

    # Example seizure at 900 seconds = 15 minutes
    seizure_start = 900
    seizure_end = 930

    print("=" * 70)
    print("WINDOW / LABEL TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Positive interval
    # ---------------------------------------------------------

    positive_interval = get_positive_interval(
        seizure_start=seizure_start,
        sph_sec=300,
        sop_sec=300,
    )

    print("\nPositive interval:")
    print(f"Start : {positive_interval.start} sec")
    print(f"End   : {positive_interval.end} sec")
    print(
        f"Duration : "
        f"{positive_interval.end - positive_interval.start} sec"
    )

    assert positive_interval.start == 300
    assert positive_interval.end == 600

    print("✓ Correct [T-10min, T-5min] interval")

    # ---------------------------------------------------------
    # Positive windows
    # ---------------------------------------------------------

    windows = generate_positive_windows(
        seizure_start_sec=seizure_start,
        duration_sec=1200,
        window_size_sec=5,
        sph_sec=300,
        sop_sec=300,
    )

    print("\nPositive windows:")
    print(f"Number of windows : {len(windows)}")
    print(f"First window      : {windows[0]}")
    print(f"Last window       : {windows[-1]}")

    assert len(windows) == 60
    assert windows[0]["start_sec"] == 300
    assert windows[-1]["end_sec"] == 600

    print("✓ 60 positive windows generated")
    print("✓ Each window is 5 seconds")

    for i, window in enumerate(windows):

        expected_start = 300 + i * 5
        expected_end = expected_start + 5

        assert abs(window["start_sec"] - expected_start) < 1e-9
        assert abs(window["end_sec"] - expected_end) < 1e-9

    print("✓ Windows have correct 5-second boundaries")

    # ---------------------------------------------------------
    # Exclusion interval
    # ---------------------------------------------------------

    exclusion = get_exclusion_interval(
        seizure_start=seizure_start,
        seizure_end=seizure_end,
        postictal_sec=600,
        sph_sec=300,
        sop_sec=300,
    )

    print("\nNegative exclusion interval:")
    print(f"Start : {exclusion.start} sec")
    print(f"End   : {exclusion.end} sec")

    assert exclusion.start == 300
    assert exclusion.end == 1530

    print("✓ Correct exclusion interval")

    # ---------------------------------------------------------
    # Negative tests
    # ---------------------------------------------------------

    assert not is_valid_negative(
        400,
        405,
        [exclusion],
    )

    print("✓ Pre-ictal window rejected as negative")

    assert not is_valid_negative(
        1000,
        1005,
        [exclusion],
    )

    print("✓ Postictal window rejected as negative")

    assert is_valid_negative(
        100,
        105,
        [exclusion],
    )

    print("✓ Distant interictal window accepted")

    print("\n" + "=" * 70)
    print("ALL WINDOW/LABEL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()