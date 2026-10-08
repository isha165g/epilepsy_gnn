import os
import sys

import pandas as pd

sys.path.append(os.path.abspath("src"))

# pyrefly: ignore [missing-import]
from preprocessing.windowing import generate_positive_windows
# pyrefly: ignore [missing-import]
from preprocessing.labels import get_positive_interval


FORECASTING_PATH = "data/metadata/forecasting_events_primary.csv"

SPH_SEC = 300
SOP_SEC = 300
WINDOW_SIZE_SEC = 5

EPSILON = 1e-6


def main():

    forecasting = pd.read_csv(FORECASTING_PATH)

    print("=" * 80)
    print("DIAGNOSING POSITIVE WINDOW GENERATION")
    print("=" * 80)

    # Take the first eligible event
    row = forecasting.iloc[0]

    seizure_start = float(row["seizure_start_sec"])

    print("\nRecording:")
    print(f"  split     : {row['split']}")
    print(f"  patient   : {row['patient']}")
    print(f"  recording : {row['recording']}")

    print("\nActual seizure start:")
    print(f"  {seizure_start!r}")

    # ---------------------------------------------------------
    # Positive interval
    # ---------------------------------------------------------

    interval = get_positive_interval(
        seizure_start=seizure_start,
        sph_sec=SPH_SEC,
        sop_sec=SOP_SEC,
    )

    print("\nPositive interval:")
    print(f"  start : {interval.start!r}")
    print(f"  end   : {interval.end!r}")
    print(
        f"  duration : "
        f"{interval.end - interval.start!r}"
    )

    # ---------------------------------------------------------
    # Generate windows
    # ---------------------------------------------------------

    windows = generate_positive_windows(
        seizure_start_sec=seizure_start,
        duration_sec=99999,
        window_size_sec=WINDOW_SIZE_SEC,
        sph_sec=SPH_SEC,
        sop_sec=SOP_SEC,
    )

    print("\nGenerated windows:")
    print(f"  count : {len(windows)}")

    # ---------------------------------------------------------
    # Test each window
    # ---------------------------------------------------------

    print("\nWindow boundary checks:")

    failed = []

    for i, window in enumerate(windows):

        start = float(window["start_sec"])
        end = float(window["end_sec"])

        valid = (
            start >= interval.start - EPSILON
            and end <= interval.end + EPSILON
        )

        if not valid:
            failed.append(i)

        if i < 3 or i >= len(windows) - 3:

            print(
                f"  {i:02d}: "
                f"{start!r} → {end!r} "
                f"valid={valid}"
            )

    print("\nFailed windows:")
    print(f"  {failed}")

    if failed:

        for i in failed:

            window = windows[i]

            print(f"\nFAILED WINDOW {i}")

            print(
                f"  start        = {window['start_sec']!r}"
            )

            print(
                f"  end          = {window['end_sec']!r}"
            )

            print(
                f"  interval.end = {interval.end!r}"
            )

            print(
                f"  end - interval.end = "
                f"{window['end_sec'] - interval.end!r}"
            )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()