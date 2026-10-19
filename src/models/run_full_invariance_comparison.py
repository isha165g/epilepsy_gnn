from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression

from src.models.temporal_invariant_dataset import TemporalInvariantDataset
from src.models.gat_gru_invariant import InvariantGATGRU
from src.models.temporal_ablation_dataset import TemporalAblationDataset
from src.models.gat_gru_ablation import GATGRUAblation

DEVICE = torch.device('cpu')
OUTPUT_DIR = Path('results/ablation/invariance_evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 68)
print('FULL RESEARCH BENCHMARK: BASELINE VS. INVARIANT SPATIOTEMPORAL GNN')
print('=' * 68)

# Datasets
train_inv_ds = TemporalInvariantDataset('train')
dev_inv_ds = TemporalInvariantDataset('dev')
train_inv_loader = DataLoader(train_inv_ds, batch_size=32, shuffle=False)
dev_inv_loader = DataLoader(dev_inv_ds, batch_size=32, shuffle=False)

train_base_ds = TemporalAblationDataset('train', feature_mode='multimodal')
dev_base_ds = TemporalAblationDataset('dev', feature_mode='multimodal')
train_base_loader = DataLoader(train_base_ds, batch_size=32, shuffle=False)
dev_base_loader = DataLoader(dev_base_ds, batch_size=32, shuffle=False)

adj = np.load('data/graph/adjacency.npy')
edge_index = torch.tensor(np.vstack(np.where(adj > 0)).astype(np.int64), dtype=torch.long, device=DEVICE)

# 1. Evaluate Latent Representations
base_model = GATGRUAblation(input_dim=26).to(DEVICE)
base_ckpt = torch.load('results/ablation/models/multimodal_best.pt', map_location=DEVICE, weights_only=False)
base_model.load_state_dict(base_ckpt['model_state_dict'])
base_model.eval()

inv_model = InvariantGATGRU(input_dim=26, num_patients=30).to(DEVICE)
inv_ckpt = torch.load('results/models/gat_gru_invariant_best.pt', map_location=DEVICE, weights_only=False)
inv_model.load_state_dict(inv_ckpt['model_state_dict'])
inv_model.eval()

def extract_all(model, loader, is_invariant=False):
    embeddings, labels, probs, patients = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if is_invariant:
                x, _, y, pt_idx = batch
                logits, _, h = model(x.to(DEVICE), edge_index, alpha=0.0)
            else:
                x, _, y = batch
                logits = model(x.to(DEVICE), edge_index)
                # Compute graph pooling + GRU embedding manually for baseline
                # using the internal layers of GATGRUAblation
                b_sz, seq_l, n_n, f_d = x.shape
                offs = torch.arange(b_sz, device=DEVICE).view(b_sz, 1, 1) * n_n
                b_edges = (edge_index.unsqueeze(0) + offs).permute(1, 0, 2).reshape(2, -1)
                t_embs = []
                for t in range(seq_l):
                    nx = x[:, t, :, :].reshape(b_sz * n_n, f_d).to(DEVICE)
                    nx = model.relu(model.gat1(nx, b_edges))
                    nx = model.relu(model.gat2(nx, b_edges))
                    gx = nx.reshape(b_sz, n_n, -1).mean(dim=1)
                    t_embs.append(gx)
                _, hid = model.gru(torch.stack(t_embs, dim=1))
                h = hid[-1]

            prob = torch.softmax(logits, dim=1)[:, 1]
            embeddings.append(h.cpu().numpy())
            labels.extend(y.cpu().numpy().tolist())
            probs.extend(prob.cpu().numpy().tolist())

    return np.concatenate(embeddings, axis=0), np.array(labels), np.array(probs)

print('Extracting Baseline representations...')
h_base_tr, y_base_tr, p_base_tr = extract_all(base_model, train_base_loader, is_invariant=False)
h_base_dev, y_base_dev, p_base_dev = extract_all(base_model, dev_base_loader, is_invariant=False)

print('Extracting Invariant representations...')
h_inv_tr, y_inv_tr, p_inv_tr = extract_all(inv_model, train_inv_loader, is_invariant=True)
h_inv_dev, y_inv_dev, p_inv_dev = extract_all(inv_model, dev_inv_loader, is_invariant=True)

# 2. Patient Identity Probing (Measure of shortcut learning)
train_pts = np.array([s.iloc[-1]['patient'] for s in train_inv_ds.sequences])
dev_pts = np.array([s.iloc[-1]['patient'] for s in dev_inv_ds.sequences])

# Probe on training embeddings: Can patient ID be recognized?
probe_base = LogisticRegression(max_iter=1000, C=1.0)
probe_base.fit(h_base_tr, train_pts)
base_pt_acc = probe_base.score(h_base_tr, train_pts)

probe_inv = LogisticRegression(max_iter=1000, C=1.0)
probe_inv.fit(h_inv_tr, train_pts)
inv_pt_acc = probe_inv.score(h_inv_tr, train_pts)

# 3. Patient-Macro Metrics on Dev
dev_patients = np.array([s.iloc[-1]['patient'] for s in dev_inv_ds.sequences])
df_base = pd.DataFrame({'patient': dev_patients, 'y': y_base_dev, 'prob': p_base_dev})
df_inv = pd.DataFrame({'patient': dev_patients, 'y': y_inv_dev, 'prob': p_inv_dev})

base_pt_aucs = [roc_auc_score(g['y'], g['prob']) for _, g in df_base.groupby('patient') if g['y'].nunique() == 2]
inv_pt_aucs = [roc_auc_score(g['y'], g['prob']) for _, g in df_inv.groupby('patient') if g['y'].nunique() == 2]

# Compute summary rows
comparison = [
    {
        'Model': 'Baseline Multimodal GAT-GRU',
        'Train ROC-AUC': float(roc_auc_score(y_base_tr, p_base_tr)),
        'Dev ROC-AUC (Pooled)': float(roc_auc_score(y_base_dev, p_base_dev)),
        'Dev Balanced Accuracy': float(balanced_accuracy_score(y_base_dev, (p_base_dev >= 0.5).astype(int))),
        'Dev Patient Median AUC': float(np.median(base_pt_aucs)),
        'Dev Patient Mean AUC': float(np.mean(base_pt_aucs)),
        'Patient Leakage Probe Acc': float(base_pt_acc),
        'Generalization Diagnosis': 'Severe Shortcut Learning (Memorized Patient ID)'
    },
    {
        'Model': 'Invariant Spatiotemporal GNN (Proposed)',
        'Train ROC-AUC': float(roc_auc_score(y_inv_tr, p_inv_tr)),
        'Dev ROC-AUC (Pooled)': float(roc_auc_score(y_inv_dev, p_inv_dev)),
        'Dev Balanced Accuracy': float(balanced_accuracy_score(y_inv_dev, (p_inv_dev >= 0.5).astype(int))),
        'Dev Patient Median AUC': float(np.median(inv_pt_aucs)),
        'Dev Patient Mean AUC': float(np.mean(inv_pt_aucs)),
        'Patient Leakage Probe Acc': float(inv_pt_acc),
        'Generalization Diagnosis': 'Patient-Invariant Seizure Pattern Learning'
    }
]

comp_df = pd.DataFrame(comparison)
print('=' * 80)
print(comp_df.to_string(index=False))
comp_df.to_csv(OUTPUT_DIR / 'baseline_vs_invariant_comparison.csv', index=False)
print('=' * 80)
print('Summary saved to:', OUTPUT_DIR / 'baseline_vs_invariant_comparison.csv')
