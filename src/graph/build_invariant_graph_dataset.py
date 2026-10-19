from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
MANIFEST_PATH = ROOT / "data/metadata/multimodal_graph_manifest.csv"
OUTPUT_MANIFEST = ROOT / "data/metadata/invariant_graph_manifest.csv"
OUTPUT_DIR = ROOT / "data/graph_invariant"

print("=" * 65)
print("BUILDING ARTIFACT-INVARIANT MULTIMODAL GRAPH DATASET")
print("=" * 65)

df = pd.read_csv(MANIFEST_PATH)
print(f"Total windows: {len(df)}")
print(f"Unique recordings: {df['recording'].nunique()}")
print(f"Unique patients:   {df['patient'].nunique()}")

recording_handcrafted = {}
recording_tf = {}

print("Step 1/2: Loading raw arrays and computing per-recording normalization statistics...")

for rec_id, group in df.groupby("recording"):
    hc_paths = group["feature_path"].tolist()
    tf_paths = [
        p.replace("time_frequency_normalized", "time_frequency")
        for p in group["normalized_time_frequency_path"]
    ]
    
    hc_arrs = np.stack([np.load(p).astype(np.float32) for p in hc_paths], axis=0)
    tf_arrs = np.stack([np.load(p).astype(np.float32) for p in tf_paths], axis=0)
    
    dyn_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    hc_dyn = hc_arrs[:, :, dyn_idx]
    hc_mean = hc_dyn.mean(axis=(0, 1), keepdims=True)
    hc_std = hc_dyn.std(axis=(0, 1), keepdims=True)
    hc_std = np.where(hc_std < 1e-8, 1.0, hc_std)
    hc_normed_dyn = (hc_dyn - hc_mean) / hc_std
    
    rel_idx = [5, 11, 12, 13, 14, 15]
    hc_rel = hc_arrs[:, :, rel_idx]
    hc_invariant = np.concatenate([hc_rel, hc_normed_dyn], axis=-1)
    
    tf_mean = tf_arrs.mean(axis=(0, 1), keepdims=True)
    tf_std = tf_arrs.std(axis=(0, 1), keepdims=True)
    tf_std = np.where(tf_std < 1e-8, 1.0, tf_std)
    tf_normed = (tf_arrs - tf_mean) / tf_std
    
    for p, arr in zip(hc_paths, hc_invariant):
        recording_handcrafted[p] = arr
    for p, arr in zip(group["normalized_time_frequency_path"], tf_normed):
        recording_tf[p] = arr

print("Step 2/2: Fusing into invariant graph archives...")
output_paths = []

for i, row in df.iterrows():
    split = row["split"]
    graph_path = Path(row["graph_path"])
    
    with np.load(graph_path, allow_pickle=False) as g:
        adjacency = g["adjacency"]
        y = g["y"]
        
    hc_x = recording_handcrafted[row["feature_path"]]
    tf_x = recording_tf[row["normalized_time_frequency_path"]]
    
    fused_x = np.concatenate([hc_x, tf_x], axis=1).astype(np.float32)
    
    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = split_dir / graph_path.name
    np.savez_compressed(
        out_file,
        x=fused_x,
        adjacency=adjacency,
        y=y,
    )
    output_paths.append(str(out_file))

df["invariant_graph_path"] = output_paths
df.to_csv(OUTPUT_MANIFEST, index=False)
print("INVARIANT GRAPH DATASET CREATION COMPLETE")
