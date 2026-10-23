#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learnable Adaptive Graph Topology:
Implements end-to-end differentiable graph structure learning:
    A_adaptive = Softmax(ReLU(E_1 * E_2^T))
where E_1, E_2 in R^(N x d) are learnable node embedding matrices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphTopology(nn.Module):
    """
    End-to-end Learnable Adaptive Graph Adjacency.
    Learns dynamic inter-channel coupling directly from data without fixed anatomical assumptions.
    """
    def __init__(self, num_nodes=20, embed_dim=16, directed=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.directed = directed

        # Node embeddings E_1 and E_2
        self.node_embed1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        if directed:
            self.node_embed2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        else:
            self.node_embed2 = self.node_embed1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.node_embed1, a=math.sqrt(5))
        if self.directed:
            nn.init.kaiming_uniform_(self.node_embed2, a=math.sqrt(5))

    def forward(self):
        """
        Compute A_adaptive = Softmax(ReLU(E_1 * E_2^T))

        Returns
        -------
        adj : torch.Tensor
            Shape (num_nodes, num_nodes), row-stochastic non-negative adjacency.
        """
        if self.directed:
            score = torch.mm(self.node_embed1, self.node_embed2.t())
        else:
            score = torch.mm(self.node_embed1, self.node_embed1.t())

        adj = F.softmax(F.relu(score), dim=-1)
        return adj


class AdaptiveGraphGATConv(nn.Module):
    """
    Graph Convolution supporting both Discrete Topologies (Distance, Correlation, PLV)
    and Continuous Learnable Adaptive Topologies.
    """
    def __init__(self, in_features, out_features, num_nodes=20, embed_dim=16, dropout=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.adaptive_topo = AdaptiveGraphTopology(num_nodes=num_nodes, embed_dim=embed_dim)

        self.gamma = nn.Parameter(torch.tensor([0.5]))  # Learnable balance between prior and adaptive
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, prior_adj=None, mode='adaptive'):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_nodes, in_features)
        prior_adj : torch.Tensor, optional
            Shape (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        mode : str
            'static' (use prior_adj), 'correlation' (use prior_adj),
            'plv' (use prior_adj), 'adaptive' (use learned A_adaptive),
            'hybrid' (combine prior and adaptive).

        Returns
        -------
        out : torch.Tensor
            Shape (batch_size, num_nodes, out_features)
        adj_used : torch.Tensor
            Shape (batch_size, num_nodes, num_nodes) or (num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = x.shape
        proj_x = self.linear(x)  # (B, N, D_out)

        A_adapt = self.adaptive_topo()  # (N, N)

        if mode == 'adaptive':
            # Pure adaptive learned topology
            out = torch.matmul(A_adapt.unsqueeze(0), proj_x)
            adj_used = A_adapt
        elif mode == 'hybrid' and prior_adj is not None:
            # Gated combination of prior graph and learned graph
            g = torch.sigmoid(self.gamma)
            if prior_adj.dim() == 2:
                A_comb = (1.0 - g) * prior_adj + g * A_adapt
                out = torch.matmul(A_comb.unsqueeze(0), proj_x)
                adj_used = A_comb
            else:
                A_comb = (1.0 - g) * prior_adj + g * A_adapt.unsqueeze(0)
                out = torch.matmul(A_comb, proj_x)
                adj_used = A_comb
        else:
            # Use discrete prior graph (Distance, Correlation, or PLV)
            if prior_adj is None:
                raise ValueError("prior_adj must be provided when mode is not 'adaptive'")
            if prior_adj.dim() == 2:
                out = torch.matmul(prior_adj.unsqueeze(0), proj_x)
                adj_used = prior_adj
            else:
                out = torch.matmul(prior_adj, proj_x)
                adj_used = prior_adj

        out = self.norm(self.dropout(self.relu(out)))
        return out, adj_used


if __name__ == '__main__':
    B, N, D_in, D_out = 4, 20, 32, 64
    x = torch.randn(B, N, D_in)
    conv = AdaptiveGraphGATConv(in_features=D_in, out_features=D_out, num_nodes=N)

    out_adapt, A_a = conv(x, mode='adaptive')
    print("Adaptive graph forward pass:")
    print("  Output shape:", out_adapt.shape)
    print("  A_adapt shape:", A_a.shape)
    print("  Row sum:", A_a[0].sum().item())
