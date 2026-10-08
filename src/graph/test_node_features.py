from pathlib import Path

import numpy as np


# pyrefly: ignore [missing-import]
from node_features import extract_node_features


# Pick the first processed EEG window
files = sorted(
    Path("data/processed").rglob("*.npy")
)

if not files:
    raise RuntimeError(
        "No processed .npy files found."
    )

path = files[0]

print("=" * 60)
print("NODE FEATURE TEST")
print("=" * 60)

print(f"Input file: {path}")

# ------------------------------------------------------------
# Load EEG window
# ------------------------------------------------------------

window = np.load(path)

print(f"\nInput shape: {window.shape}")
print(f"Input dtype: {window.dtype}")

# ------------------------------------------------------------
# Extract features
# ------------------------------------------------------------

features, feature_names = extract_node_features(
    window,
    sfreq=256
)

print(f"\nFeature matrix shape: {features.shape}")

print("\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"{i:2d}: {name}")

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

assert window.shape == (20, 1280)

assert features.shape == (20, 16)

assert features.dtype == np.float32

assert np.all(np.isfinite(features))

print("\nFeature statistics:")
print(f"Min  : {features.min():.6e}")
print(f"Max  : {features.max():.6e}")
print(f"Mean : {features.mean():.6e}")
print(f"Std  : {features.std():.6e}")

print("\nFirst node:")
for name, value in zip(
    feature_names,
    features[0]
):
    print(
        f"{name:25s}: {value:.6e}"
    )

print("\n" + "=" * 60)
print("NODE FEATURE TEST PASSED")
print("=" * 60)