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
BATCH_SIZE = 32
SEQUENCE_LENGTH = 6
OUTPUT_DIR = Path('results/ablation/invariance_evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    res = {
        'n_samples': len(y_true),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    if len(np.unique(y_true)) == 2:
        res['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        res['average_precision'] = float(average_precision_score(y_true, y_prob))
    else:
        res['roc_auc'] = float('nan')
        res['average_precision'] = float('nan')
    return res

print('=' * 65)
print('EVALUATING INVARIANCE & GENERALIZATION')
print('=' * 65)

# 1. Load Baseline Multimodal Model
base_ckpt_path = Path('results/ablation/models/multimodal_best.pt')
base_dataset = TemporalAblationDataset(split='dev', feature_mode='multimodal', sequence_length=6)
base_loader = DataLoader(base_dataset, batch_size=BATCH_SIZE, shuffle=False)

base_model = GATGRUAblation(input_dim=26).to(DEVICE)
if base_ckpt_path.exists():
    base_ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
    base_model.load_state_dict(base_ckpt['model_state_dict'])
base_model.eval()

# 2. Load Invariant Spatiotemporal Model
inv_ckpt_path = Path('results/models/gat_gru_invariant_best.pt')
inv_dataset = TemporalInvariantDataset(split='dev', sequence_length=6)
inv_loader = DataLoader(inv_dataset, batch_size=BATCH_SIZE, shuffle=False)

inv_model = InvariantGATGRU(input_dim=26, num_patients=30).to(DEVICE)
if inv_ckpt_path.exists():
    inv_ckpt = torch.load(inv_ckpt_path, map_location=DEVICE, weights_only=False)
    inv_model.load_state_dict(inv_ckpt['model_state_dict'])
inv_model.eval()

# Adjacency
adj = np.load('data/graph/adjacency.npy')
edge_index = torch.tensor(np.vstack(np.where(adj > 0)).astype(np.int64), dtype=torch.long, device=DEVICE)

# Evaluate Invariant Model
inv_labels, inv_probs, inv_embeddings = [], [], []
with torch.no_grad():
    for x, _, y, _ in inv_loader:
        seiz_logits, _, h = inv_model(x.to(DEVICE), edge_index, alpha=0.0)
        prob = torch.softmax(seiz_logits, dim=1)[:, 1]
        inv_labels.extend(y.cpu().numpy().tolist())
        inv_probs.extend(prob.cpu().numpy().tolist())
        inv_embeddings.append(h.cpu().numpy())

inv_embeddings = np.concatenate(inv_embeddings, axis=0)
inv_df = pd.DataFrame({
    'patient': [s.iloc[-1]['patient'] for s in inv_dataset.sequences],
    'recording': [s.iloc[-1]['recording'] for s in inv_dataset.sequences],
    'label': inv_labels,
    'probability': inv_probs,
})

inv_pooled = compute_metrics(np.array(inv_labels), np.array(inv_probs))
inv_patient_aucs = []
for p, g in inv_df.groupby('patient'):
    if g['label'].nunique() == 2:
        auc = roc_auc_score(g['label'], g['probability'])
        inv_patient_aucs.append(auc)

inv_pooled['median_patient_auc'] = float(np.median(inv_patient_aucs)) if inv_patient_aucs else float('nan')
inv_pooled['mean_patient_auc'] = float(np.mean(inv_patient_aucs)) if inv_patient_aucs else float('nan')

print('INVARIANT MODEL DEV RESULTS:')
for k, v in inv_pooled.items():
    print(f'  {k}: {v}')

inv_df.to_csv(OUTPUT_DIR / 'invariant_dev_predictions.csv', index=False)
with open(OUTPUT_DIR / 'invariant_metrics.json', 'w') as f:
    json.dump(inv_pooled, f, indent=2)

print('Outputs saved to:', OUTPUT_DIR)
