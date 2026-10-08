# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

# pyrefly: ignore [missing-import]
from torch_geometric.nn import GATConv


class EEGGATGRU(nn.Module):
    """
    Spatiotemporal EEG model.

    Input:
        x          : (batch, sequence_length, 20, 16)
        edge_index : graph topology for the 20 EEG channels

    Processing:
        GAT -> spatial representation
        GRU -> temporal representation
        Linear -> classification
    """

    def __init__(
        self,
        input_dim=16,
        gat_hidden=32,
        gat_output=64,
        gru_hidden=64,
        num_classes=2,
        dropout=0.3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Spatial encoder
        # ----------------------------------------------------

        self.gat1 = GATConv(
            input_dim,
            gat_hidden,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.gat2 = GATConv(
            gat_hidden,
            gat_output,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ----------------------------------------------------
        # Temporal encoder
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=gat_output,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            gru_hidden,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index
    ):
        """
        x shape:
            (batch, T, 20, 16)

        Returns:
            logits: (batch, 2)
        """

        batch_size, seq_len, num_nodes, num_features = x.shape

        spatial_embeddings = []

        # ----------------------------------------------------
        # Process every temporal window independently
        # through the same GAT.
        # ----------------------------------------------------

        for t in range(seq_len):

            xt = x[:, t, :, :]

            # Process each graph independently.
            graph_embeddings = []

            for b in range(batch_size):

                node_features = xt[b]

                h = self.gat1(
                    node_features,
                    edge_index
                )

                h = self.relu(h)
                h = self.dropout(h)

                h = self.gat2(
                    h,
                    edge_index
                )

                h = self.relu(h)

                # Global mean pooling over 20 EEG nodes
                graph_embedding = h.mean(
                    dim=0
                )

                graph_embeddings.append(
                    graph_embedding
                )

            graph_embeddings = torch.stack(
                graph_embeddings,
                dim=0
            )

            spatial_embeddings.append(
                graph_embeddings
            )

        # ----------------------------------------------------
        # (T, batch, features)
        # ->
        # (batch, T, features)
        # ----------------------------------------------------

        temporal_input = torch.stack(
            spatial_embeddings,
            dim=1
        )

        # ----------------------------------------------------
        # GRU
        # ----------------------------------------------------

        gru_output, _ = self.gru(
            temporal_input
        )

        # ----------------------------------------------------
        # Use representation of LAST window.
        #
        # This corresponds to the prediction time.
        # ----------------------------------------------------

        final_representation = gru_output[:, -1, :]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            final_representation
        )

        return logits


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    model = EEGGATGRU()

    batch_size = 4
    sequence_length = 6

    x = torch.randn(
        batch_size,
        sequence_length,
        20,
        16
    )

    # Same fixed 20-node topology used by the project.
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0]
        ],
        dtype=torch.long
    ).t()

    output = model(
        x,
        edge_index
    )

    print("=" * 60)
    print("GAT + GRU SMOKE TEST")
    print("=" * 60)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)