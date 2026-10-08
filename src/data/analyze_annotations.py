from pathlib import Path
import pandas as pd
import re


ANNOTATION_ROOT = Path("data/metadata/tusz_annotations")
OUTPUT_FILE = Path("data/metadata/recordings_metadata.csv")

SEIZURE_LABELS = {
    "spsz",
    "cpsz",
    "fnsz",
    "gnsz",
    "absz",
    "tnsz",
    "tcsz",
    "seiz",
}


def get_split(path: Path) -> str:
    for split in ("train", "dev", "eval"):
        if split in path.parts:
            return split
    return "unknown"


def get_patient(path: Path) -> str:
    for split in ("train", "dev", "eval"):
        if split in path.parts:
            idx = path.parts.index(split)
            return path.parts[idx + 1]
    return "unknown"


def read_duration(path: Path):
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("# duration"):
                match = re.search(r"=\s*([0-9.]+)", line)
                if match:
                    return float(match.group(1))
    return None


def merge_events(rows, tolerance=1.0):
    """
    Merge overlapping/nearby channel annotations into seizure events.

    Individual CSV rows are channel-level annotations. Multiple channels
    may therefore refer to the same seizure.
    """

    if not rows:
        return []

    intervals = sorted(
        [
            (float(start), float(stop), str(label))
            for start, stop, label in rows
        ],
        key=lambda x: x[0],
    )

    events = []

    current_start = intervals[0][0]
    current_end = intervals[0][1]
    current_labels = {intervals[0][2]}

    for start, end, label in intervals[1:]:

        if start <= current_end + tolerance:
            current_end = max(current_end, end)
            current_labels.add(label)

        else:
            events.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "labels": sorted(current_labels),
                }
            )

            current_start = start
            current_end = end
            current_labels = {label}

    events.append(
        {
            "start": current_start,
            "end": current_end,
            "labels": sorted(current_labels),
        }
    )

    return events


def analyze_recording(csv_path: Path):

    split = get_split(csv_path)
    patient = get_patient(csv_path)
    recording = csv_path.stem
    duration = read_duration(csv_path)

    try:
        df = pd.read_csv(csv_path, comment="#")
    except Exception as exc:
        print(f"ERROR reading {csv_path}: {exc}")
        return []

    if df.empty:
        return [{
            "split": split,
            "patient": patient,
            "recording": recording,
            "duration_sec": duration,
            "num_seizures": 0,
            "seizure_index": None,
            "seizure_start_sec": None,
            "seizure_end_sec": None,
            "seizure_duration_sec": None,
            "seizure_labels": None,
        }]

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    seizure_df = df[df["label"].isin(SEIZURE_LABELS)]

    rows = [
        (row["start_time"], row["stop_time"], row["label"])
        for _, row in seizure_df.iterrows()
    ]

    events = merge_events(rows)

    if not events:
        return [{
            "split": split,
            "patient": patient,
            "recording": recording,
            "duration_sec": duration,
            "num_seizures": 0,
            "seizure_index": None,
            "seizure_start_sec": None,
            "seizure_end_sec": None,
            "seizure_duration_sec": None,
            "seizure_labels": None,
        }]

    output = []

    for index, event in enumerate(events, start=1):

        start = event["start"]
        end = event["end"]

        output.append({
            "split": split,
            "patient": patient,
            "recording": recording,
            "duration_sec": duration,
            "num_seizures": len(events),
            "seizure_index": index,
            "seizure_start_sec": start,
            "seizure_end_sec": end,
            "seizure_duration_sec": end - start,
            "seizure_labels": "|".join(event["labels"]),
        })

    return output


def main():

    csv_files = sorted(
        ANNOTATION_ROOT.rglob("*.csv")
    )

    print(f"Found {len(csv_files)} annotation files.")

    rows = []

    for index, csv_path in enumerate(csv_files, start=1):

        if index % 500 == 0:
            print(
                f"Processed {index}/{len(csv_files)}"
            )

        rows.extend(
            analyze_recording(csv_path)
        )

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nAnalysis complete.")
    print(f"Rows: {len(df)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nRecordings by split:")
    print(
        df.groupby("split")["recording"]
        .nunique()
    )

    print("\nSeizure events by split:")
    print(
        df[df["seizure_index"].notna()]
        .groupby("split")
        .size()
    )

    print("\nPatients with seizures:")
    print(
        df[df["seizure_index"].notna()]
        .groupby("split")["patient"]
        .nunique()
    )


if __name__ == "__main__":
    main()