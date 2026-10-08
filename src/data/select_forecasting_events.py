import pandas as pd

# --------------------------------------------------
# Forecasting configuration
# --------------------------------------------------

SPH_MIN = 5
SOP_MIN = 5

SPH_SEC = SPH_MIN * 60
SOP_SEC = SOP_MIN * 60

PREICTAL_START_OFFSET = SPH_SEC + SOP_SEC
PREICTAL_END_OFFSET = SOP_SEC


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(
    "data/metadata/recordings_metadata.csv"
)

seizures = df[
    df["seizure_index"].notna()
].copy()

seizures["seizure_index"] = (
    seizures["seizure_index"].astype(int)
)


# --------------------------------------------------
# Sort events chronologically
# --------------------------------------------------

seizures = seizures.sort_values(
    [
        "split",
        "patient",
        "recording",
        "seizure_start_sec",
    ]
).reset_index(drop=True)


# --------------------------------------------------
# Determine forecasting windows
# --------------------------------------------------

seizures["forecast_start_sec"] = (
    seizures["seizure_start_sec"]
    - PREICTAL_START_OFFSET
)

seizures["forecast_end_sec"] = (
    seizures["seizure_start_sec"]
    - PREICTAL_END_OFFSET
)


# --------------------------------------------------
# Basic eligibility
# --------------------------------------------------

seizures["enough_recording"] = (
    seizures["forecast_start_sec"] >= 0
)


# --------------------------------------------------
# Check overlap with previous seizure
#
# We conservatively require the forecasting
# window to begin after the previous seizure ended.
# --------------------------------------------------

seizures["previous_seizure_end_sec"] = (
    seizures
    .groupby(["split", "patient", "recording"])
    ["seizure_end_sec"]
    .shift(1)
)

seizures["clear_of_previous_seizure"] = (
    seizures["previous_seizure_end_sec"].isna()
    |
    (
        seizures["forecast_start_sec"]
        > seizures["previous_seizure_end_sec"]
    )
)


# --------------------------------------------------
# Final eligibility
# --------------------------------------------------

seizures["eligible"] = (
    seizures["enough_recording"]
    &
    seizures["clear_of_previous_seizure"]
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

output = seizures[
    [
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
]

output.to_csv(
    "data/metadata/forecasting_events.csv",
    index=False
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("\n========================================")
print("FORECASTING EVENT SELECTION")
print("========================================")

print(f"SPH: {SPH_MIN} minutes")
print(f"SOP: {SOP_MIN} minutes")

print("\nTotal seizure events:")
print(len(output))

print("\nEligible events:")
print(output["eligible"].sum())

print(
    f"Eligibility rate: "
    f"{output['eligible'].mean() * 100:.2f}%"
)

print("\nEligibility by split:")

print(
    output.groupby("split")["eligible"]
    .agg(["count", "sum"])
)

eligible = output[
    output["eligible"]
]

print("\nEligible patients:")

print(
    eligible.groupby("split")["patient"]
    .nunique()
)

print("\nEligible recordings:")

print(
    eligible.groupby("split")["recording"]
    .nunique()
)

print("\nFirst 20 eligible events:")

print(
    eligible.head(20).to_string(index=False)
)