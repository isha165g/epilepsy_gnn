import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATGRUAblation(nn.Module):
    def __init__(
        self,
        input_dim,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        dropout=0.3,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.gat1 = GATConv(
            input_dim,
            gat_hidden,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            gat_hidden,
            embedding_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden,
            batch_first=True,
        )

        self.classifier = nn.Linear(gru_hidden, 2)

    def forward(self, x, edge_index):
        """
        x:          (batch, time, nodes, features)
        edge_index: (2, edges), shared graph topology

        Returns:
            logits: (batch, 2)
        """
        batch_size, seq_len, num_nodes, num_features = x.shape

        if num_features != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"got {num_features}"
            )

        offsets = (
            torch.arange(batch_size, device=x.device)
            .view(batch_size, 1, 1) * num_nodes
        )

        batched_edges = (
            edge_index.unsqueeze(0) + offsets
        ).permute(1, 0, 2).reshape(2, -1)

        temporal_embeddings = []

        for t in range(seq_len):
            node_x = x[:, t, :, :].reshape(
                batch_size * num_nodes, num_features
            )

            node_x = self.gat1(node_x, batched_edges)
            node_x = self.relu(node_x)
            node_x = self.dropout(node_x)

            node_x = self.gat2(node_x, batched_edges)
            node_x = self.relu(node_x)

            graph_x = node_x.reshape(
                batch_size, num_nodes, -1
            ).mean(dim=1)

            temporal_embeddings.append(graph_x)

        temporal_embeddings = torch.stack(
            temporal_embeddings, dim=1
        )

        _, hidden = self.gru(temporal_embeddings)
        return self.classifier(hidden[-1])
