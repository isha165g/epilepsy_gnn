import torch
import torch.nn as nn
from torch.autograd import Function
from torch_geometric.nn import GATConv

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, alpha=None):
        a = alpha if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, a)

class InvariantGATGRU(nn.Module):
    def __init__(
        self,
        input_dim=26,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_patients=30,
        dropout=0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.gru_hidden = gru_hidden
        self.num_patients = num_patients

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

        self.seizure_classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        self.grl = GradientReversal()
        self.patient_discriminator = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_patients),
        )

    def extract_embedding(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.shape
        if num_features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {num_features}")

        offsets = (
            torch.arange(batch_size, device=x.device)
            .view(batch_size, 1, 1) * num_nodes
        )
        batched_edges = (
            edge_index.unsqueeze(0) + offsets
        ).permute(1, 0, 2).reshape(2, -1)

        temporal_embeddings = []
        for t in range(seq_len):
            node_x = x[:, t, :, :].reshape(batch_size * num_nodes, num_features)
            node_x = self.gat1(node_x, batched_edges)
            node_x = self.relu(node_x)
            node_x = self.dropout(node_x)

            node_x = self.gat2(node_x, batched_edges)
            node_x = self.relu(node_x)

            graph_x = node_x.reshape(batch_size, num_nodes, -1).mean(dim=1)
            temporal_embeddings.append(graph_x)

        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        _, hidden = self.gru(temporal_embeddings)
        h = hidden[-1]
        return h

    def forward(self, x, edge_index, alpha=1.0):
        h = self.extract_embedding(x, edge_index)
        seizure_logits = self.seizure_classifier(h)
        h_reversed = self.grl(h, alpha=alpha)
        patient_logits = self.patient_discriminator(h_reversed)
        return seizure_logits, patient_logits, h
