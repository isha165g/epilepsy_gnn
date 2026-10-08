from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DIR = Path("data/features")
MANIFEST_PATH = Path(
    "data/metadata/feature_manifest.csv"
)

NORMALIZATION_PATH = Path(
    "data/metadata/feature_normalization.npz"
)

SFREQ = 256


def calculate_training_statistics(
    manifest: pd.DataFrame
):
    """
    Calculate feature-wise mean and standard deviation
    using TRAINING WINDOWS ONLY.

    Statistics are calculated across:
        all training windows × all 20 nodes
    """

    train_manifest = manifest[
        manifest["split"] == "train"
    ].reset_index(drop=True)

    print(
        f"Training feature files: "
        f"{len(train_manifest):,}"
    )

    feature_sum = None
    feature_sum_sq = None
    count = 0

    for idx, row in train_manifest.iterrows():

        path = Path(row["feature_path"])

        X = np.load(path).astype(
            np.float64
        )

        if X.shape != (20, 16):
            raise ValueError(
                f"Unexpected shape: "
                f"{path}: {X.shape}"
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                f"NaN/Inf found in {path}"
            )

        # X shape = (20, 16)
        if feature_sum is None:
            feature_sum = np.zeros(
                16,
                dtype=np.float64
            )
            feature_sum_sq = np.zeros(
                16,
                dtype=np.float64
            )

        feature_sum += X.sum(axis=0)
        feature_sum_sq += (
            X ** 2
        ).sum(axis=0)

        count += X.shape[0]

        if (idx + 1) % 500 == 0:
            print(
                f"Statistics: "
                f"{idx + 1:,} / "
                f"{len(train_manifest):,}"
            )

    mean = feature_sum / count

    variance = (
        feature_sum_sq / count
        - mean ** 2
    )

    # Numerical precision can occasionally produce
    # tiny negative values.
    variance = np.maximum(
        variance,
        0.0
    )

    std = np.sqrt(
        variance
    )

    # Prevent division by zero.
    std[std < 1e-12] = 1.0

    return mean, std


def normalize_split(
    manifest: pd.DataFrame,
    split: str,
    mean: np.ndarray,
    std: np.ndarray
):
    """
    Normalize one dataset split using
    TRAINING statistics.
    """

    split_manifest = manifest[
        manifest["split"] == split
    ].reset_index(drop=True)

    normalized_dir = (
        FEATURE_DIR
        / f"{split}_normalized"
    )

    normalized_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    normalized_paths = []

    for idx, row in split_manifest.iterrows():

        path = Path(
            row["feature_path"]
        )

        X = np.load(path).astype(
            np.float32
        )

        X_normalized = (
            X - mean
        ) / std

        X_normalized = (
            X_normalized
            .astype(np.float32)
        )

        if not np.all(
            np.isfinite(
                X_normalized
            )
        ):
            raise ValueError(
                f"NaN/Inf after normalization: "
                f"{path}"
            )

        output_path = (
            normalized_dir
            / f"{idx:06d}.npy"
        )

        np.save(
            output_path,
            X_normalized
        )

        normalized_paths.append(
            str(output_path)
        )

    split_manifest[
        "normalized_feature_path"
    ] = normalized_paths

    return split_manifest


def main():

    print("=" * 70)
    print("TRAIN-ONLY FEATURE NORMALIZATION")
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Total feature files: "
        f"{len(manifest):,}"
    )

    # --------------------------------------------------------
    # Calculate statistics from TRAIN ONLY
    # --------------------------------------------------------

    print("\nCalculating training statistics...")

    mean, std = (
        calculate_training_statistics(
            manifest
        )
    )

    print("\nTraining statistics:")
    print("-" * 70)

    feature_names = [
        "mean",
        "std",
        "rms",
        "peak_to_peak",
        "line_length",
        "spectral_entropy",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "delta_relative_power",
        "theta_relative_power",
        "alpha_relative_power",
        "beta_relative_power",
        "gamma_relative_power",
    ]

    for name, m, s in zip(
        feature_names,
        mean,
        std
    ):
        print(
            f"{name:25s} "
            f"mean={m:.6e} "
            f"std={s:.6e}"
        )

    # --------------------------------------------------------
    # Save normalization parameters
    # --------------------------------------------------------

    NORMALIZATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        NORMALIZATION_PATH,
        mean=mean,
        std=std,
        feature_names=np.array(
            feature_names
        )
    )

    print(
        f"\nSaved normalization statistics:"
        f"\n{NORMALIZATION_PATH}"
    )

    # --------------------------------------------------------
    # Normalize all splits
    # --------------------------------------------------------

    normalized_manifests = []

    for split in [
        "train",
        "dev",
        "eval"
    ]:

        print(
            f"\nNormalizing {split}..."
        )

        split_manifest = normalize_split(
            manifest,
            split,
            mean,
            std
        )

        normalized_manifests.append(
            split_manifest
        )

        print(
            f"✓ {split}: "
            f"{len(split_manifest):,} files"
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    normalized_manifest = pd.concat(
        normalized_manifests,
        ignore_index=True
    )

    output_manifest = Path(
        "data/metadata/"
        "normalized_feature_manifest.csv"
    )

    normalized_manifest.to_csv(
        output_manifest,
        index=False
    )

    print("\n" + "=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Normalized files: "
        f"{len(normalized_manifest):,}"
    )

    print(
        f"Manifest:\n"
        f"{output_manifest}"
    )


if __name__ == "__main__":
    main()