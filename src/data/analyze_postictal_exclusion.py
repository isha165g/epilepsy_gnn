import pandas as pd

SPH_MIN = 5
SOP_MIN = 5

TOTAL_PREICTAL_SEC = (SPH_MIN + SOP_MIN) * 60

EXCLUSION_MINUTES = [0, 5, 10, 15, 30, 60]


# --------------------------------------------------
# Load seizure events
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

seizures = seizures.sort_values(
    [
        "split",
        "patient",
        "recording",
        "seizure_start_sec",
    ]
).reset_index(drop=True)


# --------------------------------------------------
# Forecasting window
# --------------------------------------------------

seizures["forecast_start_sec"] = (
    seizures["seizure_start_sec"]
    - TOTAL_PREICTAL_SEC
)

seizures["forecast_end_sec"] = (
    seizures["seizure_start_sec"]
    - SOP_MIN * 60
)


# --------------------------------------------------
# Previous seizure
# --------------------------------------------------

seizures["previous_seizure_end_sec"] = (
    seizures
    .groupby(["split", "patient", "recording"])
    ["seizure_end_sec"]
    .shift(1)
)

seizures["enough_recording"] = (
    seizures["forecast_start_sec"] >= 0
)


# --------------------------------------------------
# Evaluate different exclusion periods
# --------------------------------------------------

print("\n========================================")
print("POSTICTAL EXCLUSION ANALYSIS")
print("========================================")

print(f"SPH = {SPH_MIN} min")
print(f"SOP = {SOP_MIN} min")
print(f"Total preictal window = {SPH_MIN + SOP_MIN} min")

print("\nTotal seizure events:", len(seizures))
print(
    "Events with enough recording:",
    seizures["enough_recording"].sum()
)

print("\n========================================")
print("RETENTION BY POSTICTAL EXCLUSION")
print("========================================")

results = []

for exclusion_min in EXCLUSION_MINUTES:

    exclusion_sec = exclusion_min * 60

    # If there is no previous seizure,
    # the event passes the postictal test.
    clear_of_postictal = (
        seizures["previous_seizure_end_sec"].isna()
        |
        (
            seizures["forecast_start_sec"]
            >=
            seizures["previous_seizure_end_sec"]
            + exclusion_sec
        )
    )

    eligible = (
        seizures["enough_recording"]
        &
        clear_of_postictal
    )

    eligible_df = seizures[eligible]

    results.append(
        {
            "postictal_exclusion_min": exclusion_min,
            "eligible_events": len(eligible_df),
            "eligible_percent": (
                len(eligible_df)
                / len(seizures)
                * 100
            ),
            "eligible_patients": (
                eligible_df
                .groupby("split")["patient"]
                .nunique()
                .sum()
            ),
            "train_events": (
                (eligible_df["split"] == "train").sum()
            ),
            "dev_events": (
                (eligible_df["split"] == "dev").sum()
            ),
            "eval_events": (
                (eligible_df["split"] == "eval").sum()
            ),
        }
    )

results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

results_df.to_csv(
    "data/metadata/postictal_exclusion_analysis.csv",
    index=False
)

print(
    "\nSaved:"
    " data/metadata/postictal_exclusion_analysis.csv"
)