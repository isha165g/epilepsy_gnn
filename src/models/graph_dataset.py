from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch_geometric.data import Dataset, Data


class EEGGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for the EEG seizure-forecasting graphs.

    Each .npz file contains:
        x          : (20, 16) node features
        y          : scalar binary label
        adjacency  : (20, 20) graph adjacency matrix
    """

    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split

        if split not in {"train", "dev", "eval"}:
            raise ValueError(
                f"Invalid split '{split}'. "
                "Expected 'train', 'dev', or 'eval'."
            )

        self.graph_dir = self.root / split

        if not self.graph_dir.exists():
            raise FileNotFoundError(
                f"Graph directory not found: {self.graph_dir}"
            )

        self.files = sorted(self.graph_dir.glob("*.npz"))

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz graph files found in {self.graph_dir}"
            )

        super().__init__(str(self.root))

    def len(self):
        return len(self.files)

    def get(self, idx):
        path = self.files[idx]

        graph = np.load(path)

        x = torch.tensor(
            graph["x"],
            dtype=torch.float32
        )

        y = torch.tensor(
            graph["y"],
            dtype=torch.long
        )

        adjacency = graph["adjacency"]

        # Convert adjacency matrix -> edge_index
        # np.where returns:
        #   row = source nodes
        #   col = destination nodes
        row, col = np.where(adjacency > 0)

        edge_index = torch.tensor(
            np.stack([row, col]),
            dtype=torch.long
        )

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        return data


if __name__ == "__main__":
    dataset = EEGGraphDataset(
        root="data/graph",
        split="train"
    )

    print("Number of graphs:", len(dataset))

    sample = dataset[0]

    print("\nSample graph:")
    print(sample)

    print("\nNode features:", sample.x.shape)
    print("Edge index:", sample.edge_index.shape)
    print("Label:", sample.y.item())
    print("Number of edges:", sample.edge_index.shape[1])