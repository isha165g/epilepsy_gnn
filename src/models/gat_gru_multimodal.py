
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch_geometric.nn import GATConv


class MultimodalGATGRU(nn.Module):
    def __init__(
        self,
        input_dim=26,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        dropout=0.3,
    ):
        super().__init__()

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
                    (B, 6, 20, 26)
        edge_index: (2, edges), shared graph topology

        Returns:
            logits: (batch, 2)
        """
        batch_size, seq_len, num_nodes, num_features = x.shape

        if num_features != 26:
            raise ValueError(
                f"Expected 26 input features, got {num_features}"
            )

        # Create disconnected copies of the same graph,
        # allowing all samples to pass through GAT together.
        offsets = (
            torch.arange(batch_size, device=x.device)
            .view(batch_size, 1, 1) * num_nodes
        )

        batched_edges = (
            edge_index.unsqueeze(0) + offsets
        ).permute(1, 0, 2).reshape(2, -1)

        temporal_embeddings = []

        for t in range(seq_len):
            # Combine nodes from all graphs in this batch.
            node_x = x[:, t, :, :].reshape(
                batch_size * num_nodes, num_features
            )

            node_x = self.gat1(node_x, batched_edges)
            node_x = self.relu(node_x)
            node_x = self.dropout(node_x)

            node_x = self.gat2(node_x, batched_edges)
            node_x = self.relu(node_x)

            # Mean-pool nodes independently for each graph.
            graph_x = node_x.reshape(
                batch_size, num_nodes, -1
            ).mean(dim=1)

            temporal_embeddings.append(graph_x)

        temporal_embeddings = torch.stack(
            temporal_embeddings, dim=1
        )  # (B, 6, 64)

        _, hidden = self.gru(temporal_embeddings)

        logits = self.classifier(hidden[-1])
        return logits
