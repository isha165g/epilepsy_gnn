from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset


# ============================================================
# CONFIG
# ============================================================

GRAPH_ROOT = Path("data/graph")
GRAPH_MANIFEST = Path(
    "data/metadata/graph_manifest.csv"
)

SEQUENCE_LENGTH = 6
WINDOW_DURATION = 5.0


# ============================================================
# DATASET
# ============================================================

class TemporalGraphDataset(Dataset):
    """
    Temporal sequences of consecutive EEG graph windows.

    Each sample contains:

        x:
            (T, 20, 16)

        y:
            label of the LAST window

    T = SEQUENCE_LENGTH

    Only windows from the same recording are grouped.
    Windows must be temporally consecutive.
    """

    def __init__(
        self,
        split,
        sequence_length=SEQUENCE_LENGTH
    ):

        if split not in {"train", "dev", "eval"}:
            raise ValueError(
                f"Invalid split: {split}"
            )

        self.split = split
        self.sequence_length = sequence_length

        # ----------------------------------------------------
        # Load graph manifest
        # ----------------------------------------------------

        manifest = pd.read_csv(
            GRAPH_MANIFEST
        )

        manifest = manifest[
            manifest["split"] == split
        ].copy()

        manifest = manifest.sort_values(
            [
                "patient",
                "recording",
                "start_sec"
            ]
        ).reset_index(drop=True)

        self.manifest = manifest

        # ----------------------------------------------------
        # Build sequences
        # ----------------------------------------------------

        self.sequences = []

        for (_, _), group in manifest.groupby(
            ["patient", "recording"],
            sort=False
        ):

            group = group.sort_values(
                "start_sec"
            ).reset_index(drop=True)

            starts = group["start_sec"].to_numpy()

            for i in range(
                len(group) - sequence_length + 1
            ):

                window_group = group.iloc[
                    i:i + sequence_length
                ]

                # ------------------------------------------------
                # Ensure consecutive 5-second windows.
                # This prevents gaps in the temporal sequence.
                # ------------------------------------------------

                expected_starts = (
                    starts[i]
                    + np.arange(sequence_length)
                    * WINDOW_DURATION
                )

                actual_starts = (
                    window_group["start_sec"]
                    .to_numpy()
                )

                if not np.allclose(
                    actual_starts,
                    expected_starts,
                    atol=1e-3
                ):
                    continue

                graph_paths = (
                    window_group["graph_path"]
                    .tolist()
                )

                target_label = int(
                    window_group.iloc[-1]["label"]
                )

                self.sequences.append(
                    {
                        "graph_paths": graph_paths,
                        "label": target_label,
                        "patient": window_group.iloc[-1]["patient"],
                        "recording": window_group.iloc[-1]["recording"],
                        "target_start_sec": float(
                            window_group.iloc[-1]["start_sec"]
                        )
                    }
                )

        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No temporal sequences created for {split}"
            )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        labels = np.array(
            [s["label"] for s in self.sequences]
        )

        print(
            f"{split.upper()} temporal sequences: "
            f"{len(self.sequences)}"
        )

        print(
            f"  Negative: {(labels == 0).sum()}"
        )

        print(
            f"  Positive: {(labels == 1).sum()}"
        )

    # ========================================================
    # Dataset API
    # ========================================================

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        item = self.sequences[idx]

        graphs = []

        for graph_path in item["graph_paths"]:

            graph = np.load(
                graph_path
            )

            x = graph["x"].astype(
                np.float32
            )

            if x.shape != (20, 16):
                raise ValueError(
                    f"Unexpected graph shape: {x.shape}"
                )

            graphs.append(x)

        x = np.stack(
            graphs,
            axis=0
        )

        # Shape:
        # (sequence_length, 20, 16)

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            torch.tensor(
                item["label"],
                dtype=torch.long
            )
        )


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    for split in ["train", "dev"]:

        dataset = TemporalGraphDataset(
            split=split
        )

        x, y = dataset[0]

        print(
            f"\n{split} first sequence:"
        )

        print(
            "X shape:",
            tuple(x.shape)
        )

        print(
            "y:",
            y.item()
        )