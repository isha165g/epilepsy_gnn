from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset


ROOT = Path(".")
MANIFEST_PATH = ROOT / "data/metadata/multimodal_graph_manifest.csv"

SEQUENCE_LENGTH = 6
WINDOW_DURATION = 5.0

FEATURE_DIMS = {
    "handcrafted": 16,
    "time_frequency": 10,
    "multimodal": 26,
}


class TemporalAblationDataset(Dataset):
    """
    Controlled temporal dataset for feature ablation.

    All configurations use identical windows, labels, and topology.
    Feature modes:
      - handcrafted: first 16 features from original graph archives
      - time_frequency: normalized 10-feature STFT arrays
      - multimodal: fused 26-feature graph archives
    """

    def __init__(self, split, feature_mode, sequence_length=SEQUENCE_LENGTH):
        if split not in {"train", "dev", "eval"}:
            raise ValueError(f"Invalid split: {split}")

        if feature_mode not in FEATURE_DIMS:
            raise ValueError(
                f"Invalid feature_mode: {feature_mode}. "
                f"Choose from {list(FEATURE_DIMS)}"
            )

        self.split = split
        self.feature_mode = feature_mode
        self.sequence_length = sequence_length

        df = pd.read_csv(MANIFEST_PATH)
        df = df[df["split"] == split].copy()
        df = df.sort_values(
            ["patient", "recording", "start_sec"]
        ).reset_index(drop=True)

        self.sequences = []

        for _, group in df.groupby(
            ["patient", "recording"], sort=False
        ):
            group = group.reset_index(drop=True)
            starts = group["start_sec"].to_numpy(dtype=float)

            for i in range(len(group) - sequence_length + 1):
                seq = group.iloc[i:i + sequence_length]
                actual_starts = starts[i:i + sequence_length]

                # Use the exact continuity rule from the original baseline.
                expected_starts = (
                    actual_starts[0]
                    + np.arange(sequence_length) * WINDOW_DURATION
                )

                if not np.allclose(
                    actual_starts, expected_starts, atol=1e-3
                ):
                    continue

                # Also require adjacent window boundaries to match.
                ends = seq["end_sec"].to_numpy(dtype=float)
                if not np.allclose(
                    ends[:-1], actual_starts[1:], atol=1e-3
                ):
                    continue

                self.sequences.append(seq.copy())

        if not self.sequences:
            raise RuntimeError(f"No sequences created for {split}")

        labels = np.array([
            int(seq.iloc[-1]["label_graph"])
            for seq in self.sequences
        ])

        print(
            f"{split.upper()} | {feature_mode}: "
            f"{len(self.sequences)} sequences | "
            f"negative={(labels == 0).sum()} | "
            f"positive={(labels == 1).sum()}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        node_features = []
        edge_index = None

        for _, row in seq.iterrows():
            if self.feature_mode == "handcrafted":
                path = row["graph_path"]
                with np.load(path, allow_pickle=False) as graph:
                    x = graph["x"].astype(np.float32)
                    adjacency = graph["adjacency"]

            elif self.feature_mode == "time_frequency":
                path = row["normalized_time_frequency_path"]
                x = np.load(path, allow_pickle=False).astype(np.float32)
                with np.load(
                    row["multimodal_graph_path"], allow_pickle=False
                ) as graph:
                    adjacency = graph["adjacency"]

            else:
                path = row["multimodal_graph_path"]
                with np.load(path, allow_pickle=False) as graph:
                    x = graph["x"].astype(np.float32)
                    adjacency = graph["adjacency"]

            expected_dim = FEATURE_DIMS[self.feature_mode]
            if x.shape != (20, expected_dim):
                raise ValueError(
                    f"{self.feature_mode}: expected (20, {expected_dim}), "
                    f"got {x.shape} from {path}"
                )

            if not np.allclose(adjacency, adjacency.T):
                raise ValueError(f"Asymmetric adjacency: {path}")

            if edge_index is None:
                edge_index = np.vstack(
                    np.where(adjacency > 0)
                ).astype(np.int64)

            node_features.append(x)

        x = np.stack(node_features, axis=0)
        y = int(seq.iloc[-1]["label_graph"])

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


if __name__ == "__main__":
    for split in ["train", "dev"]:
        for mode in FEATURE_DIMS:
            dataset = TemporalAblationDataset(split, mode)
            x, edge_index, y = dataset[0]
            print(
                f"  sample shape={tuple(x.shape)}, "
                f"edges={tuple(edge_index.shape)}, label={y.item()}"
            )
