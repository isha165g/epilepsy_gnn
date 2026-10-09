
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
GRAPH_MANIFEST = ROOT / "data/metadata/graph_manifest.csv"
TF_MANIFEST = (
    ROOT / "data/metadata/time_frequency_normalized_manifest.csv"
)
OUTPUT_MANIFEST = ROOT / "data/metadata/multimodal_graph_manifest.csv"

KEYS = [
    "split",
    "patient",
    "recording",
    "start_sec",
    "end_sec",
    "processed_path",
]

graph_df = pd.read_csv(GRAPH_MANIFEST)
tf_df = pd.read_csv(TF_MANIFEST)

print("=" * 65)
print("MULTIMODAL GRAPH DATASET BUILD")
print("=" * 65)
print("Graph rows:", len(graph_df))
print("Time-frequency rows:", len(tf_df))

# Keep usable rows and ensure each window has a unique match.
graph_df = graph_df[
    graph_df["usable"].astype(str).str.lower().isin(["true", "1"])
].copy()

tf_df = tf_df[
    tf_df["usable"].astype(str).str.lower().isin(["true", "1"])
].copy()

if graph_df.duplicated(KEYS).any():
    raise ValueError("Duplicate window keys in graph manifest.")

if tf_df.duplicated(KEYS).any():
    raise ValueError("Duplicate window keys in time-frequency manifest.")

# Include label in validation, but use stable recording/window identifiers
# for matching.
merged = graph_df.merge(
    tf_df[KEYS + ["label", "normalized_time_frequency_path"]],
    on=KEYS,
    how="outer",
    suffixes=("_graph", "_tf"),
    indicator=True,
    validate="one_to_one",
)

unmatched = merged[merged["_merge"] != "both"]
if not unmatched.empty:
    print("\nUnmatched rows by merge status:")
    print(unmatched["_merge"].value_counts())
    raise ValueError(
        f"{len(unmatched)} windows could not be matched exactly."
    )

if not np.array_equal(
    merged["label_graph"].to_numpy(),
    merged["label_tf"].to_numpy(),
):
    raise ValueError("Labels differ between the two manifests.")

if not (merged["split"].notna()).all():
    raise ValueError("Missing split values.")

print("Exact window matches:", len(merged))
print("Labels and splits validated.")

# Build separate multimodal files without modifying baseline graphs.
output_paths = []

for i, row in merged.iterrows():
    graph_path = Path(row["graph_path"])
    tf_path = Path(row["normalized_time_frequency_path"])

    with np.load(graph_path, allow_pickle=False) as graph:
        graph_data = {key: graph[key] for key in graph.files}

    graph_x = graph_data["x"]
    tf_x = np.load(tf_path)

    if graph_x.shape != (20, 16):
        raise ValueError(
            f"Unexpected graph feature shape {graph_x.shape}: {graph_path}"
        )

    if tf_x.shape != (20, 10):
        raise ValueError(
            f"Unexpected time-frequency shape {tf_x.shape}: {tf_path}"
        )

    fused_x = np.concatenate([graph_x, tf_x], axis=1)

    if fused_x.shape != (20, 26):
        raise ValueError(f"Unexpected fused shape: {fused_x.shape}")

    if not np.isfinite(fused_x).all():
        raise ValueError(f"Non-finite fused features: {graph_path}")

    # Verify graph label matches manifest.
    if "y" in graph_data:
        graph_label = int(np.asarray(graph_data["y"]).reshape(-1)[0])
        if graph_label != int(row["label_graph"]):
            raise ValueError(f"Graph label mismatch: {graph_path}")

    graph_data["x"] = fused_x.astype(np.float32)

    output_dir = ROOT / "data/graph_multimodal" / row["split"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / graph_path.name
    np.savez_compressed(output_path, **graph_data)
    output_paths.append(str(output_path))

    if (i + 1) % 1000 == 0:
        print(f"Fused {i + 1:,}/{len(merged):,}")

merged["multimodal_graph_path"] = output_paths
merged.drop(columns=["_merge"], inplace=True)
merged.to_csv(OUTPUT_MANIFEST, index=False)

print("\n" + "=" * 65)
print("MULTIMODAL FUSION COMPLETE")
print("=" * 65)
print("Fused graphs:", len(merged))
print("Output manifest:", OUTPUT_MANIFEST)
print("\nGraphs by split:")
print(merged["split"].value_counts().to_string())
print("\nFeature shape: (20, 26)")
print("Original graph files were not modified.")
