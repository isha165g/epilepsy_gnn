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


from src.models.multimodal_fusion import GatedMultimodalFusion


class InvariantGatedMultimodalGATGRU(nn.Module):
    """
    Patient-Independent Spatiotemporal GNN with Gated Multimodal Unit (GMU).

    Input:
      Multimodal node features x (..., 26) decomposed into:
        x_time (features 0:16) - Time-Domain Morphological Stream
        x_freq (features 16:26) - Frequency-Domain Spectral Stream
      GMU fuses x_time and x_freq into fused_dim (e.g. 32) at each node.
      Spatial GAT layers propagate across electrode topology.
      Temporal GRU tracks dynamic sequence context.
      Dual-head: Task Classification + Adversarial Patient Discriminator (GRL).
    """
    def __init__(
        self,
        time_dim=16,
        freq_dim=10,
        fused_dim=32,
        gat_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_patients=30,
        dropout=0.3,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.fused_dim = fused_dim
        self.embedding_dim = embedding_dim
        self.gru_hidden = gru_hidden
        self.num_patients = num_patients

        # 1. Gated Multimodal Fusion
        self.fusion = GatedMultimodalFusion(
            time_dim=time_dim,
            freq_dim=freq_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # 2. Spatial Graph Attention Encoder
        self.gat1 = GATConv(
            fused_dim,
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

        # 3. Temporal Recurrent Encoder
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden,
            batch_first=True,
        )

        # 4. Task Classifier
        self.seizure_classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        # 5. Adversarial Patient Discriminator
        self.grl = GradientReversal()
        self.patient_discriminator = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_patients),
        )

    def extract_embedding(self, x, edge_index, return_attention=False):
        batch_size, seq_len, num_nodes, num_features = x.shape
        if num_features != (self.time_dim + self.freq_dim):
            raise ValueError(
                f"Expected {self.time_dim + self.freq_dim} features, got {num_features}"
            )

        offsets = (
            torch.arange(batch_size, device=x.device)
            .view(batch_size, 1, 1) * num_nodes
        )
        batched_edges = (
            edge_index.unsqueeze(0) + offsets
        ).permute(1, 0, 2).reshape(2, -1)

        temporal_embeddings = []
        temporal_attentions = []
        last_gate = None

        for t in range(seq_len):
            step_x = x[:, t, :, :]  # (B, N, F)
            x_time = step_x[..., :self.time_dim]
            x_freq = step_x[..., self.time_dim:]

            fused_x, gate = self.fusion(x_time, x_freq)
            last_gate = gate

            node_x = fused_x.reshape(batch_size * num_nodes, self.fused_dim)
            if return_attention:
                node_x, (b_edges, alpha) = self.gat1(node_x, batched_edges, return_attention_weights=True)
                node_att = torch.zeros(batch_size * num_nodes, device=x.device, dtype=torch.float32)
                node_att.scatter_add_(0, b_edges[1], alpha.squeeze(-1))
                temporal_attentions.append(node_att.reshape(batch_size, num_nodes))
            else:
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

        if return_attention:
            att_map = torch.stack(temporal_attentions, dim=-1)  # (B, num_nodes, seq_len)
            return h, last_gate, att_map
        return h, last_gate

    def forward(self, x, edge_index, alpha=1.0, return_attention=False):
        if return_attention:
            h, gate, att_map = self.extract_embedding(x, edge_index, return_attention=True)
            seizure_logits = self.seizure_classifier(h)
            h_reversed = self.grl(h, alpha=alpha)
            patient_logits = self.patient_discriminator(h_reversed)
            return seizure_logits, patient_logits, h, gate, att_map
        else:
            h, gate = self.extract_embedding(x, edge_index, return_attention=False)
            seizure_logits = self.seizure_classifier(h)
            h_reversed = self.grl(h, alpha=alpha)
            patient_logits = self.patient_discriminator(h_reversed)
            return seizure_logits, patient_logits, h, gate
