#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset for Patient-Independent Seizure Detection Graphs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPO_DIR = Path('/home/isha/epilepsy_gnn')
MANIFEST_PATH = REPO_DIR / 'data/metadata/detection_graph_manifest.csv'
SEQUENCE_LENGTH = 3  # 3 consecutive 5-sec windows = 15s spatiotemporal context


class DetectionGraphDataset(Dataset):
    def __init__(self, split='train', sequence_length=SEQUENCE_LENGTH):
        if split not in {'train', 'dev', 'eval'}:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.sequence_length = sequence_length

        df = pd.read_csv(MANIFEST_PATH)
        df_split = df[df['split'] == split].copy()

        # Map train patients to indices for adversarial discriminator
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
                    # Check temporal continuity
                    w_starts = starts[i : i + sequence_length]
                    diffs = np.diff(w_starts)
                    # Windows are consecutive 5s windows
                    if np.all(np.abs(diffs - 5.0) < 1.0):
                        # Consistent label within sequence
                        labels = group['label'].iloc[i : i + sequence_length].values
                        if len(set(labels)) == 1:
                            self.samples.append(group.iloc[i : i + sequence_length])

        if not self.samples:
            # Fallback to single windows if not enough consecutive blocks
            for i in range(len(df_split)):
                self.samples.append(df_split.iloc[i : i + 1])
            self.sequence_length = 1

        labels = np.array([int(s.iloc[-1]['label']) for s in self.samples])
        print(f"[{split.upper()}] Detection Dataset: {len(self.samples)} sequences | "
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
        for _, row in seq_df.iterrows():
            g_path = REPO_DIR / row['graph_path']
            data = np.load(g_path)
            x_list.append(data['x'])  # (20, 26)

        x_tensor = torch.tensor(np.stack(x_list, axis=0), dtype=torch.float32)  # (seq_len, 20, 26)
        y_tensor = torch.tensor(label, dtype=torch.long)
        pt_tensor = torch.tensor(pt_idx, dtype=torch.long)

        return x_tensor, y_tensor, pt_tensor
