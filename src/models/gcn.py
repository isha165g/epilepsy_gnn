# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from torch_geometric.nn import GCNConv, global_mean_pool


class EEGGCN(nn.Module):
    """
    Baseline GCN for EEG seizure forecasting.

    Input:
        x          : [num_nodes, 16]
        edge_index : [2, num_edges]
        batch      : [num_nodes]

    Output:
        logits     : [num_graphs, 2]
    """

    def __init__(
        self,
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels
        )

        self.classifier = nn.Linear(
            out_channels,
            num_classes
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        # First graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Convert node representations
        # into one representation per EEG window
        x = global_mean_pool(x, batch)

        # Classification
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    from graph_dataset import EEGGraphDataset
    # pyrefly: ignore [missing-import]
    from torch_geometric.loader import DataLoader

    dataset = EEGGraphDataset(
        root="data/graph",
        split="train"
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    batch = next(iter(loader))

    model = EEGGCN()

    logits = model(
        batch.x,
        batch.edge_index,
        batch.batch
    )

    print("Input:")
    print("  Nodes:", batch.x.shape)
    print("  Edges:", batch.edge_index.shape)

    print("\nOutput:")
    print("  Logits:", logits.shape)
    print("  Values:")
    print(logits)

    print("\nLabels:")
    print(" ", batch.y.shape)