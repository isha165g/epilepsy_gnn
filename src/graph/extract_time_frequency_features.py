
from pathlib import Path

import numpy as np
import pandas as pd

from time_frequency_features import (
    extract_time_frequency_features,
)


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/time_frequency")

INPUT_MANIFEST = Path(
    "data/metadata/processed_window_manifest.csv"
)

OUTPUT_MANIFEST = Path(
    "data/metadata/time_frequency_manifest.csv"
)

SFREQ = 256


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(INPUT_MANIFEST)

    manifest = manifest[
        manifest["usable"].astype(str).str.lower().isin(
            ["true", "1"]
        )
    ].copy()

    manifest = manifest.reset_index(drop=True)

    print("=" * 70)
    print("BATCH TIME-FREQUENCY FEATURE EXTRACTION")
    print("=" * 70)
    print("Usable windows:", len(manifest))

    records = []

    for idx, row in manifest.iterrows():

        processed_path = Path(row["processed_path"])

        if not processed_path.exists():
            raise FileNotFoundError(
                f"Missing processed EEG window: {processed_path}"
            )

        window = np.load(processed_path)

        features = extract_time_frequency_features(
            window,
            sfreq=SFREQ,
        )

        if features.shape != (20, 10):
            raise ValueError(
                f"Unexpected feature shape: {features.shape}"
            )

        split = str(row["split"])

        output_dir = OUTPUT_DIR / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{idx:06d}.npy"

        np.save(output_path, features)

        record = row.to_dict()
        record["time_frequency_path"] = str(output_path)

        records.append(record)

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1:,}/{len(manifest):,}")

    output_manifest = pd.DataFrame(records)
    output_manifest.to_csv(OUTPUT_MANIFEST, index=False)

    print("\n" + "=" * 70)
    print("TIME-FREQUENCY EXTRACTION COMPLETE")
    print("=" * 70)
    print("Feature files:", len(output_manifest))
    print("Feature shape: (20, 10)")
    print("Manifest:", OUTPUT_MANIFEST)

    print("\nFiles by split:")
    print(output_manifest["split"].value_counts().sort_index())


if __name__ == "__main__":
    main()
