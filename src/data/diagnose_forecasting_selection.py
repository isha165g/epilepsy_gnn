import pandas as pd

SPH_MIN = 5
SOP_MIN = 5

SPH_SEC = SPH_MIN * 60
SOP_SEC = SOP_MIN * 60

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

# Forecasting window:
# [seizure_start - 10 min, seizure_start - 5 min]

seizures["forecast_start_sec"] = (
    seizures["seizure_start_sec"]
    - (SPH_SEC + SOP_SEC)
)

seizures["forecast_end_sec"] = (
    seizures["seizure_start_sec"]
    - SOP_SEC
)

# Previous seizure in same recording
seizures["previous_seizure_end_sec"] = (
    seizures
    .groupby(["split", "patient", "recording"])
    ["seizure_end_sec"]
    .shift(1)
)

seizures["enough_recording"] = (
    seizures["forecast_start_sec"] >= 0
)

seizures["overlaps_previous"] = (
    seizures["previous_seizure_end_sec"].notna()
    &
    (
        seizures["forecast_start_sec"]
        <= seizures["previous_seizure_end_sec"]
    )
)

seizures["eligible"] = (
    seizures["enough_recording"]
    &
    ~seizures["overlaps_previous"]
)

# --------------------------------------------------
# Overall diagnostics
# --------------------------------------------------

print("\n========================================")
print("FORECASTING SELECTION DIAGNOSTICS")
print("========================================")

print(f"SPH = {SPH_MIN} min")
print(f"SOP = {SOP_MIN} min")

print("\nTotal seizure events:")
print(len(seizures))

print("\nEnough recording:")
print(seizures["enough_recording"].sum())

print("\nRejected because insufficient recording:")
print((~seizures["enough_recording"]).sum())

print("\nOverlaps previous seizure:")
print(seizures["overlaps_previous"].sum())

print("\nFinal eligible:")
print(seizures["eligible"].sum())


# --------------------------------------------------
# Split-wise
# --------------------------------------------------

print("\n=== SPLIT-WISE ===")

summary = seizures.groupby("split").agg(
    total_events=("seizure_index", "count"),
    enough_recording=("enough_recording", "sum"),
    overlaps_previous=("overlaps_previous", "sum"),
    eligible=("eligible", "sum"),
)

print(summary)


# --------------------------------------------------
# Look at rejected events
# --------------------------------------------------

print("\n=== EXAMPLES OF EVENTS REJECTED BY PREVIOUS SEIZURE ===")

rejected = seizures[
    seizures["enough_recording"]
    & seizures["overlaps_previous"]
].copy()

rejected["gap_after_previous_sec"] = (
    rejected["seizure_start_sec"]
    - rejected["previous_seizure_end_sec"]
)

print(
    rejected[
        [
            "split",
            "patient",
            "recording",
            "seizure_index",
            "previous_seizure_end_sec",
            "seizure_start_sec",
            "gap_after_previous_sec",
            "forecast_start_sec",
            "forecast_end_sec",
        ]
    ]
    .sort_values("gap_after_previous_sec")
    .head(30)
    .to_string(index=False)
)


# --------------------------------------------------
# Gap distribution
# --------------------------------------------------

print("\n=== GAP BETWEEN PREVIOUS SEIZURE AND CURRENT SEIZURE ===")

print(
    rejected["gap_after_previous_sec"]
    .describe()
    .round(2)
)


# --------------------------------------------------
# Save diagnostic data
# --------------------------------------------------

seizures.to_csv(
    "data/metadata/forecasting_selection_diagnostics.csv",
    index=False
)

print(
    "\nSaved:"
    " data/metadata/forecasting_selection_diagnostics.csv"
)