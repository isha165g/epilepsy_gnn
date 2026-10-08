import pandas as pd

# ============================================================
# Configuration
# ============================================================

SPH_MIN = 5
SOP_MIN = 5

SPH_SEC = SPH_MIN * 60
SOP_SEC = SOP_MIN * 60

TOTAL_PREICTAL_SEC = SPH_SEC + SOP_SEC


# ============================================================
# Load seizure metadata
# ============================================================

df = pd.read_csv(
    "data/metadata/recordings_metadata.csv"
)

seizures = df[
    df["seizure_index"].notna()
].copy()

seizures["seizure_index"] = (
    seizures["seizure_index"].astype(int)
)

seizures = seizures.sort_values(
    [
        "split",
        "patient",
        "recording",
        "seizure_start_sec",
    ]
).reset_index(drop=True)


# ============================================================
# Forecasting window
#
# Positive window:
#
# [ seizure_start - SPH - SOP,
#   seizure_start - SOP ]
# ============================================================

seizures["forecast_start_sec"] = (
    seizures["seizure_start_sec"]
    - TOTAL_PREICTAL_SEC
)

seizures["forecast_end_sec"] = (
    seizures["seizure_start_sec"]
    - SOP_SEC
)


# ============================================================
# Previous seizure
# ============================================================

seizures["previous_seizure_end_sec"] = (
    seizures
    .groupby(["split", "patient", "recording"])
    ["seizure_end_sec"]
    .shift(1)
)


# ============================================================
# Eligibility
#
# Requirement 1:
# Enough recording exists before the forecast window.
#
# Requirement 2:
# If a previous seizure exists, at least
# SPH + SOP seconds separate its end from
# the current seizure start.
# ============================================================

enough_recording = (
    seizures["forecast_start_sec"] >= 0
)

enough_separation = (
    seizures["previous_seizure_end_sec"].isna()
    |
    (
        seizures["seizure_start_sec"]
        - seizures["previous_seizure_end_sec"]
        >= TOTAL_PREICTAL_SEC
    )
)

seizures["eligible"] = (
    enough_recording
    & enough_separation
)


# ============================================================
# Save event-level manifest
# ============================================================

event_columns = [
    "split",
    "patient",
    "recording",
    "duration_sec",
    "seizure_index",
    "seizure_start_sec",
    "seizure_end_sec",
    "forecast_start_sec",
    "forecast_end_sec",
    "eligible",
]

events = seizures[event_columns].copy()

events = events[
    events["eligible"]
].reset_index(drop=True)

events.to_csv(
    "data/metadata/forecasting_events_primary.csv",
    index=False
)


# ============================================================
# Create unique recording manifest
# ============================================================

recordings = (
    events[
        [
            "split",
            "patient",
            "recording",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["split", "patient", "recording"]
    )
    .reset_index(drop=True)
)

recordings.to_csv(
    "data/metadata/eligible_recordings.csv",
    index=False
)


# ============================================================
# Report
# ============================================================

print("\n========================================")
print("PRIMARY FORECASTING MANIFEST")
print("========================================")

print(f"SPH: {SPH_MIN} min")
print(f"SOP: {SOP_MIN} min")

print("\nEligible seizure events:")
print(len(events))

print("\nEligible recordings:")
print(len(recordings))

print("\nEvents by split:")
print(
    events.groupby("split").size()
)

print("\nPatients by split:")
print(
    events.groupby("split")["patient"]
    .nunique()
)

print("\nRecordings by split:")
print(
    recordings.groupby("split").size()
)

print("\nFiles created:")
print(
    "  data/metadata/forecasting_events_primary.csv"
)
print(
    "  data/metadata/eligible_recordings.csv"
)