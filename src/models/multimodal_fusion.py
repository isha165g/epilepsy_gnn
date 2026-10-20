#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gated Multimodal Unit (GMU) and Cross-Modal Attention Fusion for EEG Graph Nodes.

Mathematical Formulation:
    h_time = tanh(W_t x_time + b_t)
    h_freq = tanh(W_f x_freq + b_f)
    z      = sigmoid(W_g [x_time || x_freq] + b_g)
    x_fused = z * h_time + (1 - z) * h_freq
"""

import torch
import torch.nn as nn


class GatedMultimodalFusion(nn.Module):
    """
    Gated Multimodal Unit (GMU) that dynamically weights morphological time-domain
    dynamics against time-frequency spectral rhythms.
    """
    def __init__(self, time_dim, freq_dim, fused_dim=32, dropout=0.1):
        super().__init__()
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.fused_dim = fused_dim

        # Stream feature transformation matrices
        self.fc_time = nn.Linear(time_dim, fused_dim)
        self.fc_freq = nn.Linear(freq_dim, fused_dim)

        # Multimodal gating mechanism
        self.fc_gate = nn.Linear(time_dim + freq_dim, fused_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(fused_dim)

    def forward(self, x_time, x_freq):
        """
        Parameters
        ----------
        x_time : torch.Tensor
            Shape (..., time_dim)
        x_freq : torch.Tensor
            Shape (..., freq_dim)

        Returns
        -------
        x_fused : torch.Tensor
            Shape (..., fused_dim)
        gate : torch.Tensor
            Shape (..., fused_dim) - Gate values in [0, 1] for interpretability
        """
        h_time = self.tanh(self.fc_time(x_time))
        h_freq = self.tanh(self.fc_freq(x_freq))

        # Dynamic gating coefficient
        concat = torch.cat([x_time, x_freq], dim=-1)
        gate = self.sigmoid(self.fc_gate(concat))

        # Fused multimodal convex combination
        fused = gate * h_time + (1.0 - gate) * h_freq
        fused = self.layer_norm(self.dropout(fused))

        return fused, gate


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion: allows the frequency spectral stream to attend
    over morphological time-domain features and vice versa.
    """
    def __init__(self, time_dim, freq_dim, fused_dim=32, num_heads=2, dropout=0.1):
        super().__init__()
        self.fused_dim = fused_dim
        self.proj_time = nn.Linear(time_dim, fused_dim)
        self.proj_freq = nn.Linear(freq_dim, fused_dim)

        self.mha_t_to_f = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_f_to_t = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.out_proj = nn.Linear(fused_dim * 2, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_time, x_freq):
        orig_shape = x_time.shape[:-1]
        t = self.proj_time(x_time).reshape(-1, 1, self.fused_dim)  # (N_total, 1, D)
        f = self.proj_freq(x_freq).reshape(-1, 1, self.fused_dim)  # (N_total, 1, D)

        att_t, _ = self.mha_t_to_f(t, f, f)
        att_f, _ = self.mha_f_to_t(f, t, t)

        fused = torch.cat([att_t.squeeze(1), att_f.squeeze(1)], dim=-1)
        fused = self.norm(self.dropout(self.out_proj(fused)))
        return fused.view(*orig_shape, self.fused_dim), None


if __name__ == "__main__":
    b_size, num_nodes = 4, 20
    x_t = torch.randn(b_size, num_nodes, 16)
    x_f = torch.randn(b_size, num_nodes, 10)

    gmu = GatedMultimodalFusion(time_dim=16, freq_dim=10, fused_dim=32)
    out, gate = gmu(x_t, x_f)
    print("GMU Smoke Test:")
    print("  Input time shape :", x_t.shape)
    print("  Input freq shape :", x_f.shape)
    print("  Fused out shape  :", out.shape)
    print("  Gate mean value  :", gate.mean().item())
