#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Brain Connectivity & Graph Topology Generation:
1. Tang et al. (ICLR 2022) Dynamic Correlation Graph (normalized cross-correlation, top-tau).
2. Phase-Locking Value (PLV) Functional Connectivity Graph (Hilbert transform phase synchrony).
3. Static Distance Anatomical Graph (10-20 system geometry).
"""

from pathlib import Path
import numpy as np
import scipy.signal
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
STATIC_ADJACENCY_PATH = REPO_DIR / 'data/graph/adjacency.npy'


def compute_static_distance_adjacency(path=STATIC_ADJACENCY_PATH):
    """
    Load the static anatomical 10-20 distance adjacency matrix (20x20).
    """
    if Path(path).exists():
        adj = np.load(path).astype(np.float32)
    else:
        # Fallback to identity + standard connections if file missing
        adj = np.eye(20, dtype=np.float32)
    return adj


def compute_correlation_adjacency(signals, top_tau=3):
    """
    Compute Tang et al. (ICLR 2022) dynamic normalized cross-correlation graph.

    W_ij = |X_i * X_j| for top-tau neighbors of each node, plus self-edges.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N_nodes, T_samples), e.g. (20, 1280)
    top_tau : int
        Number of strongest neighbor edges to retain per node (default: 3).

    Returns
    -------
    adj : np.ndarray
        Shape (N_nodes, N_nodes), directed weighted adjacency matrix with self-loops.
    """
    signals = np.asarray(signals, dtype=np.float64)
    num_nodes = signals.shape[0]

    # Compute pairwise Pearson correlation
    corr = np.corrcoef(signals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)

    # Zero out diagonal to select only inter-channel neighbors
    np.fill_diagonal(abs_corr, 0.0)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        row = abs_corr[i]
        if np.all(row == 0.0):
            continue
        top_k = min(top_tau, num_nodes - 1)
        top_indices = np.argsort(row)[-top_k:]
        adj[i, top_indices] = abs_corr[i, top_indices]

    # Add self-loops (weight = 1.0)
    np.fill_diagonal(adj, 1.0)
    return adj


def compute_plv_adjacency(signals, sfreq=256, band=(4.0, 30.0), top_tau=3):
    """
    Compute Phase-Locking Value (PLV) functional connectivity graph.

    Measures instantaneous phase synchrony via Hilbert transform across
    the ictal EEG frequency band (e.g. theta to beta: 4-30 Hz).
    PLV_ij = |1/T * sum(exp(j * (phi_i(t) - phi_j(t))))|

    Parameters
    ----------
    signals : np.ndarray
        Shape (N_nodes, T_samples), e.g. (20, 1280)
    sfreq : float
        Sampling rate in Hz.
    band : tuple(float, float)
        Passband frequencies for phase synchrony.
    top_tau : int
        Number of strongest neighbor edges to retain per node (default: 3).

    Returns
    -------
    adj : np.ndarray
        Shape (N_nodes, N_nodes), directed weighted adjacency matrix with self-loops.
    """
    signals = np.asarray(signals, dtype=np.float64)
    num_nodes = signals.shape[0]

    # Bandpass filter into synchronization band
    sos = scipy.signal.butter(4, list(band), btype='bandpass', fs=sfreq, output='sos')
    filtered = scipy.signal.sosfiltfilt(sos, signals, axis=-1)

    # Analytic signal and instantaneous phase
    analytic = scipy.signal.hilbert(filtered, axis=-1)
    phase = np.angle(analytic)  # (N_nodes, T_samples)

    # Pairwise phase difference matrix
    diff = phase[:, None, :] - phase[None, :, :]  # (N, N, T)
    plv = np.abs(np.mean(np.exp(1j * diff), axis=-1))  # (N, N)
    plv = np.nan_to_num(plv, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero out diagonal to select non-self neighbors
    np.fill_diagonal(plv, 0.0)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        row = plv[i]
        if np.all(row == 0.0):
            continue
        top_k = min(top_tau, num_nodes - 1)
        top_indices = np.argsort(row)[-top_k:]
        adj[i, top_indices] = plv[i, top_indices]

    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    return adj


def adjacency_to_edge_index(adj):
    """
    Convert an (N, N) adjacency matrix into PyTorch edge_index (2, E) and edge_weight (E,).
    """
    src, dst = np.where(adj > 0)
    weights = adj[src, dst]

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


if __name__ == '__main__':
    # Smoke test
    N, T = 20, 1280
    test_sig = np.random.randn(N, T)
    adj_c = compute_correlation_adjacency(test_sig, top_tau=3)
    adj_p = compute_plv_adjacency(test_sig, top_tau=3)
    print("Smoke test:")
    print("  Correlation adj shape:", adj_c.shape, "non-zeros:", np.count_nonzero(adj_c))
    print("  PLV adj shape        :", adj_p.shape, "non-zeros:", np.count_nonzero(adj_p))
