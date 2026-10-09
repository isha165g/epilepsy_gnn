
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset

ROOT = Path(".")
MANIFEST_PATH = (
    ROOT / "data/metadata/multimodal_graph_manifest.csv"
)


class TemporalMultimodalDataset(Dataset):
    def __init__(self, split, sequence_length=6):
        if split not in {"train", "dev", "eval"}:
            raise ValueError(f"Invalid split: {split}")

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

            for start in range(
                len(group) - sequence_length + 1
            ):
                seq = group.iloc[start:start + sequence_length]

                starts = seq["start_sec"].to_numpy(dtype=float)
                ends = seq["end_sec"].to_numpy(dtype=float)

                # Require consecutive 5-second windows.
                if not np.allclose(
                    np.diff(starts), 5.0, atol=0.05
                ):
                    continue

                if not np.allclose(
                    ends[:-1], starts[1:], atol=0.05
                ):
                    continue

                self.sequences.append(seq.copy())

        print(
            f"{split.upper()} multimodal sequences: "
            f"{len(self.sequences)}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        node_features = []
        edge_index = None

        for _, row in seq.iterrows():
            with np.load(
                row["multimodal_graph_path"],
                allow_pickle=False,
            ) as graph:
                x = graph["x"].astype(np.float32)
                adjacency = graph["adjacency"]

                if x.shape != (20, 26):
                    raise ValueError(
                        f"Expected (20, 26), got {x.shape}"
                    )

                if adjacency.shape != (20, 20):
                    raise ValueError(
                        f"Unexpected adjacency shape: "
                        f"{adjacency.shape}"
                    )

                if not np.allclose(adjacency, adjacency.T):
                    raise ValueError("Adjacency matrix is not symmetric.")

                if edge_index is None:
                    # Convert adjacency matrix to PyTorch edge format.
                    edge_index = np.vstack(
                        np.where(adjacency > 0)
                    ).astype(np.int64)

                node_features.append(x)

        x = np.stack(node_features)  # (6, 20, 26)

        # Target is the label of the final window.
        y = int(seq.iloc[-1]["label_graph"])

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
