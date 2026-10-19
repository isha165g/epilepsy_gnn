from pathlib import Path
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score

from src.models.temporal_invariant_dataset import TemporalInvariantDataset
from src.models.gat_gru_invariant import InvariantGATGRU

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
PATIENCE = 6
SEQUENCE_LENGTH = 6

CHECKPOINT_PATH = Path('results/models/gat_gru_invariant_best.pt')
SUMMARY_PATH = Path('results/ablation/invariant_training_summary.json')
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cpu')

print('=' * 68)
print('DOMAIN-ADVERSARIAL INVARIANT GAT + GRU TRAINING')
print('=' * 68)
print('Device:', device)

train_dataset = TemporalInvariantDataset('train', sequence_length=SEQUENCE_LENGTH)
dev_dataset = TemporalInvariantDataset('dev', sequence_length=SEQUENCE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_labels = [train_dataset.sequences[i].iloc[-1]['label_graph'] for i in range(len(train_dataset))]
class_counts = np.bincount(train_labels, minlength=2)
total_samples = len(train_labels)
class_weights = torch.tensor(
    [total_samples / (2.0 * class_counts[0]), total_samples / (2.0 * class_counts[1])],
    dtype=torch.float32,
    device=device,
)
print('Class weights:', class_weights.tolist())

model = InvariantGATGRU(
    input_dim=26,
    gat_hidden=32,
    embedding_dim=64,
    gru_hidden=64,
    num_patients=train_dataset.num_train_patients,
    dropout=0.3,
).to(device)

criterion_task = nn.CrossEntropyLoss(weight=class_weights)
criterion_patient = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def evaluate(loader):
    model.eval()
    dev_labels, dev_probs = [], []
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for x, batch_edges, y, _ in loader:
            x, y = x.to(device), y.to(device)
            seiz_logits, _, _ = model(x, batch_edges[0].to(device), alpha=0.0)
            loss = criterion_task(seiz_logits, y)
            probs = torch.softmax(seiz_logits, dim=1)[:, 1]

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            dev_labels.extend(y.cpu().numpy().tolist())
            dev_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(dev_labels)
    y_prob = np.array(dev_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    avg_loss = total_loss / total_samples
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return avg_loss, auc, ap, bacc

best_metric = -1.0
best_epoch = 0
epochs_no_improve = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    # Dynamic GRL schedule: starts at 0 and asymptotes to 1.0
    p = float(epoch) / MAX_EPOCHS
    alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    total_task_loss = 0.0
    total_pt_loss = 0.0
    total_pt_acc = 0.0
    total_samples = 0

    for x, batch_edges, y, pt_idx in train_loader:
        x, y, pt_idx = x.to(device), y.to(device), pt_idx.to(device)
        optimizer.zero_grad()

        seiz_logits, pt_logits, _ = model(x, batch_edges[0].to(device), alpha=alpha)
        loss_task = criterion_task(seiz_logits, y)
        loss_pt = criterion_patient(pt_logits, pt_idx)

        # Joint adversarial loss
        loss = loss_task + 0.5 * alpha * loss_pt
        loss.backward()
        optimizer.step()

        total_task_loss += loss_task.item() * y.size(0)
        total_pt_loss += loss_pt.item() * y.size(0)
        total_pt_acc += (pt_logits.argmax(dim=1) == pt_idx).sum().item()
        total_samples += y.size(0)

    train_task_loss = total_task_loss / total_samples
    train_pt_acc = total_pt_acc / total_samples

    dev_loss, dev_auc, dev_ap, dev_bacc = evaluate(dev_loader)

    print(
        f'Epoch {epoch:02d} | alpha: {alpha:.2f} | '
        f'Train Task Loss: {train_task_loss:.4f} | Pt Acc: {train_pt_acc:.3f} | '
        f'Dev Loss: {dev_loss:.4f} | Dev AUC: {dev_auc:.4f} | Dev BAcc: {dev_bacc:.4f}'
    )

    history.append({
        'epoch': epoch,
        'alpha': alpha,
        'train_task_loss': train_task_loss,
        'train_pt_acc': train_pt_acc,
        'dev_loss': dev_loss,
        'dev_auc': dev_auc,
        'dev_ap': dev_ap,
        'dev_bacc': dev_bacc,
    })

    # Save model based on Dev Balanced Accuracy / ROC-AUC
    target_metric = (dev_auc + dev_bacc) / 2.0
    if target_metric > best_metric:
        best_metric = target_metric
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_metric': best_metric,
            'dev_auc': dev_auc,
            'dev_bacc': dev_bacc,
        }, CHECKPOINT_PATH)
        print(f'  Saved new best checkpoint at epoch {epoch} (AUC={dev_auc:.4f}, BAcc={dev_bacc:.4f})')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f'Early stopping triggered after {epoch} epochs.')
            break

with open(SUMMARY_PATH, 'w') as f:
    json.dump(history, f, indent=2)

print('=' * 68)
print('TRAINING COMPLETE')
print('Best epoch:', best_epoch, 'Best target metric:', best_metric)
print('Saved checkpoint:', CHECKPOINT_PATH)
