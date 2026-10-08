# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from torch_geometric.nn import GATConv, global_mean_pool


class EEGGAT(nn.Module):
    """
    Graph Attention Network baseline for EEG seizure forecasting.

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
        dropout=0.3,
        heads=1
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )

        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=heads,
            concat=False,
            dropout=dropout
        )

        self.classifier = nn.Linear(
            out_channels,
            num_classes
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch):

        # First graph attention layer
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        # Second graph attention layer
        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        # One representation per EEG window
        x = global_mean_pool(
            x,
            batch
        )

        # Classification
        logits = self.classifier(x)

        return logits