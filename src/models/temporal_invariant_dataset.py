from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path('.')
MANIFEST_PATH = ROOT / 'data/metadata/invariant_graph_manifest.csv'
SEQUENCE_LENGTH = 6
WINDOW_DURATION = 5.0

class TemporalInvariantDataset(Dataset):
    def __init__(self, split='train', sequence_length=SEQUENCE_LENGTH, balanced_patients_only=False):
        if split not in {'train', 'dev', 'eval'}:
            raise ValueError(f'Invalid split: {split}')
        self.split = split
        self.sequence_length = sequence_length
        self.balanced_patients_only = balanced_patients_only

        df = pd.read_csv(MANIFEST_PATH)
        train_patients = sorted(df[df['split'] == 'train']['patient'].unique())
        self.patient_to_idx = {p: i for i, p in enumerate(train_patients)}
        self.num_train_patients = len(train_patients)

        sub_df = df[df['split'] == split].copy()
        sub_df = sub_df.sort_values(['patient', 'recording', 'start_sec']).reset_index(drop=True)

        counts = sub_df.groupby(['patient', 'label_graph']).size().unstack(fill_value=0)
        has_both = set(counts[(counts.get(0, 0) > 0) & (counts.get(1, 0) > 0)].index)

        self.sequences = []
        for _, group in sub_df.groupby(['patient', 'recording'], sort=False):
            group = group.reset_index(drop=True)
            patient = group['patient'].iloc[0]
            if self.balanced_patients_only and patient not in has_both:
                continue

            starts = group['start_sec'].to_numpy(dtype=float)
            ends = group['end_sec'].to_numpy(dtype=float)

            for i in range(len(group) - sequence_length + 1):
                actual_starts = starts[i : i + sequence_length]
                expected_starts = actual_starts[0] + np.arange(sequence_length) * WINDOW_DURATION

                if not np.allclose(actual_starts, expected_starts, atol=1e-3):
                    continue
                if not np.allclose(ends[i : i + sequence_length - 1], actual_starts[1:], atol=1e-3):
                    continue

                seq = group.iloc[i : i + sequence_length].copy()
                self.sequences.append(seq)

        if not self.sequences:
            raise RuntimeError(f'No sequences created for {split}')

        labels = np.array([int(s.iloc[-1]['label_graph']) for s in self.sequences])
        patients = [s.iloc[-1]['patient'] for s in self.sequences]
        print(
            f'{split.upper()} INVARIANT: {len(self.sequences)} seqs | '
            f'neg={(labels == 0).sum()} | pos={(labels == 1).sum()} | '
            f'unique patients={len(set(patients))}'
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        node_features = []
        edge_index = None

        for _, row in seq.iterrows():
            path = row['invariant_graph_path']
            with np.load(path, allow_pickle=False) as g:
                x = g['x'].astype(np.float32)
                adjacency = g['adjacency']

            if edge_index is None:
                edge_index = np.vstack(np.where(adjacency > 0)).astype(np.int64)
            node_features.append(x)

        x = np.stack(node_features, axis=0)
        y = int(seq.iloc[-1]['label_graph'])
        patient_str = str(seq.iloc[-1]['patient'])
        patient_idx = self.patient_to_idx.get(patient_str, -1)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(patient_idx, dtype=torch.long),
        )