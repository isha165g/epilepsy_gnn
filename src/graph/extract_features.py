from pathlib import Path

import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
from node_features import extract_node_features


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR = Path("data/features")

MANIFEST_PATH = Path(
    "data/metadata/processed_window_manifest.csv"
)

SFREQ = 256


# ============================================================
# Main
# ============================================================

def main():

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # Only feature-extract windows that passed preprocessing QC.
    manifest = manifest[
        manifest["usable"] == True
    ].copy()

    manifest = manifest.reset_index(drop=True)

    print("=" * 70)
    print("BATCH NODE FEATURE EXTRACTION")
    print("=" * 70)

    print(f"Manifest rows : {len(manifest):,}")

    # --------------------------------------------------------
    # Process every usable window
    # --------------------------------------------------------

    feature_records = []

    for idx, row in manifest.iterrows():

        split = row["split"]

        # The processed window filename
        processed_path = Path(
            row["processed_path"]
        )

        if not processed_path.exists():
            raise FileNotFoundError(
                f"Missing processed window:\n"
                f"{processed_path}"
            )

        # ----------------------------------------------------
        # Load EEG window
        # ----------------------------------------------------

        window = np.load(
            processed_path
        )

        if window.shape != (20, 1280):
            raise ValueError(
                f"Unexpected shape for "
                f"{processed_path}: "
                f"{window.shape}"
            )

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features, feature_names = (
            extract_node_features(
                window,
                sfreq=SFREQ
            )
        )

        if features.shape != (20, 16):
            raise ValueError(
                f"Unexpected feature shape "
                f"for {processed_path}: "
                f"{features.shape}"
            )

        if not np.all(
            np.isfinite(features)
        ):
            raise ValueError(
                f"NaN/Inf detected in "
                f"{processed_path}"
            )

        # ----------------------------------------------------
        # Create output path
        # ----------------------------------------------------

        output_dir = (
            FEATURE_DIR / split
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / f"{idx:06d}.npy"
        )

        np.save(
            output_path,
            features.astype(np.float32)
        )

        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        record = row.to_dict()

        record["feature_path"] = str(
            output_path
        )

        feature_records.append(record)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (idx + 1) % 500 == 0:
            print(
                f"Processed "
                f"{idx + 1:,} / "
                f"{len(manifest):,}"
            )

    # --------------------------------------------------------
    # Save feature manifest
    # --------------------------------------------------------

    feature_manifest = pd.DataFrame(
        feature_records
    )

    feature_manifest_path = Path(
        "data/metadata/feature_manifest.csv"
    )

    feature_manifest.to_csv(
        feature_manifest_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Total feature files : "
        f"{len(feature_manifest):,}"
    )

    print("\nBy split:")
    print(
        feature_manifest["split"]
        .value_counts()
        .sort_index()
    )

    print("\nBy label:")
    print(
        feature_manifest["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nFeature shape : "
        f"(20, 16)"
    )

    print(
        f"Feature manifest:\n"
        f"{feature_manifest_path}"
    )


if __name__ == "__main__":
    main()