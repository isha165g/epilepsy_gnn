
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
MANIFEST_PATH = ROOT / "data/metadata/time_frequency_manifest.csv"
OUTPUT_MANIFEST = (
    ROOT / "data/metadata/time_frequency_normalized_manifest.csv"
)
STATS_PATH = ROOT / "data/metadata/time_frequency_normalization.npz"

df = pd.read_csv(MANIFEST_PATH)

# Keep only successfully extracted, usable windows.
df = df[
    df["time_frequency_path"].notna()
].copy().reset_index(drop=True)

train_df = df[df["split"] == "train"]

if train_df.empty:
    raise ValueError("No training features found.")

# Fit normalization statistics using TRAIN only.
train_features = np.stack([
    np.load(path)
    for path in train_df["time_frequency_path"]
])

mean = train_features.mean(axis=(0, 1), keepdims=True)
std = train_features.std(axis=(0, 1), keepdims=True)

# Avoid division by zero for constant features.
std = np.where(std < 1e-8, 1.0, std)

print("=" * 65)
print("TIME-FREQUENCY NORMALIZATION")
print("=" * 65)
print("Training feature tensor:", train_features.shape)
print("Mean shape:", mean.shape)
print("Std shape:", std.shape)

# Save train-derived statistics for reproducibility.
STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez(STATS_PATH, mean=mean, std=std)

# Save normalized arrays separately; raw arrays remain untouched.
normalized_paths = []

for i, row in df.iterrows():
    split = row["split"]
    source_path = Path(row["time_frequency_path"])

    features = np.load(source_path)
    normalized = ((features - mean.reshape(10)) /
                  std.reshape(10)).astype(np.float32)

    output_dir = ROOT / "data/time_frequency_normalized" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / source_path.name
    np.save(output_path, normalized)

    normalized_paths.append(str(output_path))

    if (i + 1) % 1000 == 0:
        print(f"Normalized {i + 1:,}/{len(df):,}")

df["normalized_time_frequency_path"] = normalized_paths
df.to_csv(OUTPUT_MANIFEST, index=False)

print("\nNormalization complete.")
print("Normalized files:", len(df))
print("Manifest:", OUTPUT_MANIFEST)
print("Statistics:", STATS_PATH)

# Verify the training distribution after normalization.
check = np.load(
    df.loc[df["split"] == "train",
           "normalized_time_frequency_path"].iloc[0]
)
assert check.shape == (20, 10)
assert np.isfinite(check).all()

print("Smoke test passed: shape (20, 10), all values finite.")
