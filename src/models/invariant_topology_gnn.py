#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invariant Spatiotemporal GNN supporting Multiple Dynamic and Adaptive Graph Topologies:
1. Static Distance Graph (Anatomical 10-20 adjacency)
2. Tang et al. Dynamic Correlation Graph (Normalized cross-correlation, top-tau)
3. Dynamic Phase-Locking Value (PLV) Coherence Graph (Hilbert phase synchrony, top-tau)
4. Learnable Adaptive Graph Topology (A_adapt = Softmax(ReLU(E_1 * E_2^T)))
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.models.multimodal_fusion import GatedMultimodalFusion


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


class InvariantTopologyGNN(nn.Module):
    """
    Spatiotemporal GNN with Gated Multimodal Fusion, Dynamic/Adaptive Topologies,
    Temporal GRU, and Domain-Adversarial Invariance.
    """
    def __init__(
        self,
        time_dim=16,
        freq_dim=10,
        fused_dim=32,
        gnn_hidden=32,
        embedding_dim=64,
        gru_hidden=64,
        num_nodes=20,
        num_patients=30,
        topology_mode='static',  # 'static', 'correlation', 'plv', 'adaptive'
        adapt_embed_dim=16,
        dropout=0.3,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.fused_dim = fused_dim
        self.gnn_hidden = gnn_hidden
        self.embedding_dim = embedding_dim
        self.gru_hidden = gru_hidden
        self.num_nodes = num_nodes
        self.num_patients = num_patients
        self.topology_mode = topology_mode

        # 1. Gated Multimodal Unit (GMU) Node Feature Fusion
        self.fusion = GatedMultimodalFusion(
            time_dim=time_dim,
            freq_dim=freq_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # 2. Learnable Adaptive Graph Topology Parameters
        self.node_embed1 = nn.Parameter(torch.randn(num_nodes, adapt_embed_dim))
        self.node_embed2 = nn.Parameter(torch.randn(num_nodes, adapt_embed_dim))
        nn.init.kaiming_uniform_(self.node_embed1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.node_embed2, a=math.sqrt(5))

        # 3. Spatial Graph Convolution / Message Passing Layers
        self.spatial_fc1 = nn.Linear(fused_dim, gnn_hidden, bias=False)
        self.spatial_fc2 = nn.Linear(gnn_hidden, embedding_dim, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(gnn_hidden)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # 4. Temporal Recurrent Encoder (GRU)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden,
            batch_first=True,
        )

        # 5. Task Classification Head (Seizure Detection)
        self.seizure_classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

        # 6. Adversarial Patient Discriminator (GRL)
        self.grl = GradientReversal()
        self.patient_discriminator = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_patients),
        )

    def get_adaptive_adjacency(self):
        """
        Compute A_adaptive = Softmax(ReLU(E_1 * E_2^T))
        """
        score = torch.mm(self.node_embed1, self.node_embed2.t())
        return F.softmax(F.relu(score), dim=-1)

    def extract_embedding(self, x, adj_batch=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, num_nodes, num_features)
        adj_batch : torch.Tensor, optional
            Shape (batch_size, seq_len, num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
        """
        batch_size, seq_len, num_nodes, num_features = x.shape

        temporal_embeddings = []
        last_gate = None
        last_adj_used = None

        if self.topology_mode == 'adaptive':
            # Compute learnable adaptive adjacency matrix
            A_matrix = self.get_adaptive_adjacency().unsqueeze(0)  # (1, N, N)
        else:
            if adj_batch is None:
                raise ValueError(f"adj_batch must be provided for topology mode '{self.topology_mode}'")
            if adj_batch.dim() == 3:
                # Shape (B, N, N) -> add seq dimension
                A_matrix = adj_batch.unsqueeze(1)
            else:
                A_matrix = adj_batch  # (B, T, N, N)

        for t in range(seq_len):
            step_x = x[:, t, :, :]  # (B, N, F)
            x_time = step_x[..., :self.time_dim]
            x_freq = step_x[..., self.time_dim:]

            # 1. Gated Multimodal Fusion
            fused_x, gate = self.fusion(x_time, x_freq)  # (B, N, fused_dim)
            last_gate = gate

            # 2. Determine adjacency for this time step
            if self.topology_mode == 'adaptive':
                A_t = A_matrix  # (1, N, N) broadcastable to (B, N, N)
            else:
                if A_matrix.dim() == 4:
                    A_t = A_matrix[:, t, :, :]  # (B, N, N)
                else:
                    A_t = A_matrix[:, :, :]
            last_adj_used = A_t

            # Normalize row sums of adjacency if needed
            if self.topology_mode != 'adaptive':
                deg = torch.clamp(A_t.sum(dim=-1, keepdim=True), min=1e-8)
                A_norm = A_t / deg
            else:
                A_norm = A_t

            # 3. Spatial Graph Message Passing (Layer 1)
            # Z1 = A * (X * W1)
            proj1 = self.spatial_fc1(fused_x)  # (B, N, gnn_hidden)
            h1 = torch.matmul(A_norm, proj1)  # (B, N, gnn_hidden)
            h1 = self.layer_norm1(self.dropout(self.relu(h1)))

            # 4. Spatial Graph Message Passing (Layer 2)
            proj2 = self.spatial_fc2(h1)  # (B, N, embedding_dim)
            h2 = torch.matmul(A_norm, proj2)  # (B, N, embedding_dim)
            h2 = self.layer_norm2(self.dropout(self.relu(h2)))

            # Readout across nodes
            graph_rep = h2.mean(dim=1)  # (B, embedding_dim)
            temporal_embeddings.append(graph_rep)

        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)  # (B, seq_len, embedding_dim)
        _, hidden = self.gru(temporal_embeddings)
        h = hidden[-1]  # (B, gru_hidden)

        return h, last_gate, last_adj_used

    def forward(self, x, adj_batch=None, alpha=1.0):
        h, gate, adj_used = self.extract_embedding(x, adj_batch)
        seizure_logits = self.seizure_classifier(h)
        h_reversed = self.grl(h, alpha=alpha)
        patient_logits = self.patient_discriminator(h_reversed)
        return seizure_logits, patient_logits, h, gate, adj_used
