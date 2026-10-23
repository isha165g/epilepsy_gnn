#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset for Patient-Independent Seizure Detection Graphs with Topology Support.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPO_DIR = Path('/home/isha/epilepsy_gnn')
MANIFEST_PATH = REPO_DIR / 'data/metadata/detection_graph_manifest.csv'
SEQUENCE_LENGTH = 1


class DetectionGraphDataset(Dataset):
    def __init__(self, split='train', sequence_length=SEQUENCE_LENGTH, topology_mode='static'):
        if split not in {'train', 'dev', 'eval'}:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.sequence_length = sequence_length
        self.topology_mode = topology_mode

        df = pd.read_csv(MANIFEST_PATH)
        df_split = df[df['split'] == split].copy()

        train_patients = sorted(df[df['split'] == 'train']['patient'].unique())
        self.patient_to_idx = {p: i for i, p in enumerate(train_patients)}
        self.num_train_patients = len(train_patients)

        df_split = df_split.sort_values(['patient', 'recording', 'start_sec']).reset_index(drop=True)

        self.samples = []
        for (pt, rec), group in df_split.groupby(['patient', 'recording'], sort=False):
            group = group.reset_index(drop=True)
            n_rows = len(group)
            starts = group['start_sec'].to_numpy(dtype=float)

            if sequence_length <= 1:
                for i in range(n_rows):
                    self.samples.append(group.iloc[i : i + 1])
            else:
                for i in range(n_rows - sequence_length + 1):
                    w_starts = starts[i : i + sequence_length]
                    diffs = np.diff(w_starts)
                    if np.all(np.abs(diffs - 5.0) < 1.0):
                        labels = group['label'].iloc[i : i + sequence_length].values
                        if len(set(labels)) == 1:
                            self.samples.append(group.iloc[i : i + sequence_length])

        if not self.samples:
            for i in range(len(df_split)):
                self.samples.append(df_split.iloc[i : i + 1])
            self.sequence_length = 1

        labels = np.array([int(s.iloc[-1]['label']) for s in self.samples])
        print(f"[{split.upper()}] Detection Dataset ({topology_mode}): {len(self.samples)} sequences | "
              f"Ictal={np.sum(labels == 1)} | Interictal={np.sum(labels == 0)} | "
              f"Patients={df_split['patient'].nunique()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_df = self.samples[idx]
        pt = seq_df.iloc[-1]['patient']
        pt_idx = self.patient_to_idx.get(pt, 0)
        label = int(seq_df.iloc[-1]['label'])

        x_list = []
        adj_list = []

        for _, row in seq_df.iterrows():
            g_path = REPO_DIR / row['graph_path']
            data = np.load(g_path)
            x_list.append(data['x'])

            if self.topology_mode == 'correlation':
                adj = data.get('adj_corr', data.get('adjacency'))
            elif self.topology_mode == 'plv':
                adj = data.get('adj_plv', data.get('adjacency'))
            elif self.topology_mode == 'adaptive':
                adj = np.eye(20, dtype=np.float32)
            else:  # 'static'
                adj = data.get('adj_static', data.get('adjacency'))

            adj_list.append(adj.astype(np.float32))

        x_tensor = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float32)  # (seq_len, 20, 26)
        adj_tensor = torch.tensor(np.stack(adj_list, axis=0), dtype=torch.float32)  # (seq_len, 20, 20)
        if self.sequence_length == 1:
            adj_tensor = adj_tensor.squeeze(0)  # (20, 20)

        y_tensor = torch.tensor(label, dtype=torch.long)
        pt_tensor = torch.tensor(pt_idx, dtype=torch.long)

        return x_tensor, adj_tensor, y_tensor, pt_tensor
