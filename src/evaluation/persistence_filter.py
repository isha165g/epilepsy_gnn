#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Persistence Filtering and Event-Level Alarm Smoothing for Seizure Forecasting.

This module implements:
1. Exponential Moving Average (EMA) smoothing of predicted preictal probabilities.
2. Sliding-Window Consensus (k-out-of-N voting) persistence filtering.
3. Dual-Level Performance Evaluation:
   - Continuous Window-Level Metrics (Accuracy, Balanced Accuracy, Sensitivity, Specificity, ROC-AUC)
   - Clinical Event-Level Metrics (Preictal Seizure Forecast Sensitivity, Interictal Specificity, Event Accuracy)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

REPO_DIR = Path('/home/isha/epilepsy_gnn')
INPUT_CSV = REPO_DIR / 'results/ablation/invariance_evaluation/invariant_dev_predictions.csv'
OUTPUT_DIR = REPO_DIR / 'results/ablation/invariance_evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_ema(probabilities: np.ndarray, beta: float = 0.7) -> np.ndarray:
    """
    Apply recursive EMA low-pass filter over temporal sequence of probabilities:
    P_smooth(t) = beta * P_smooth(t-1) + (1 - beta) * P_model(t)
    """
    if beta <= 0.0:
        return probabilities.copy()
    smoothed = np.zeros_like(probabilities, dtype=np.float64)
    smoothed[0] = probabilities[0]
    for t in range(1, len(probabilities)):
        smoothed[t] = beta * smoothed[t - 1] + (1.0 - beta) * probabilities[t]
    return smoothed


def apply_k_of_n_consensus(binary_preds: np.ndarray, N: int = 5, k: int = 3) -> np.ndarray:
    """
    Apply sliding window consensus voting:
    Alarm state is active (1) if at least k of the last N windows are predicted positive.
    """
    if N <= 1:
        return binary_preds.copy()
    rolling = np.convolve(binary_preds, np.ones(N, dtype=int), mode='same')
    return (rolling >= k).astype(int)


def process_dataset(
    df: pd.DataFrame,
    beta: float = 0.7,
    threshold: float = 0.5,
    N: int = 5,
    k: int = 3
) -> pd.DataFrame:
    records = []
    for (pt, rec), group in df.groupby(['patient', 'recording'], sort=False):
        group = group.copy().reset_index(drop=True)
        raw_probs = group['probability'].to_numpy(dtype=float)

        smoothed = apply_ema(raw_probs, beta=beta)
        binary = (smoothed >= threshold).astype(int)
        alarms = apply_k_of_n_consensus(binary, N=N, k=k)

        group['prob_raw'] = raw_probs
        group['prob_smoothed'] = smoothed
        group['pred_raw'] = (raw_probs >= 0.5).astype(int)
        group['pred_smoothed'] = binary
        group['alarm_consensus'] = alarms
        records.append(group)

    return pd.concat(records, ignore_index=True)


def compute_continuous_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'roc_auc': float(auc),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


def compute_event_metrics(df_proc: pd.DataFrame, alarm_col: str = 'alarm_consensus') -> dict:
    tp_events, fn_events, tn_events, fp_events = 0, 0, 0, 0
    for (pt, rec), group in df_proc.groupby(['patient', 'recording'], sort=False):
        group = group.copy().reset_index(drop=True)
        group['block_id'] = (group['label'] != group['label'].shift(1)).cumsum()

        for _, block in group.groupby('block_id'):
            label = block['label'].iloc[0]
            triggered = int(block[alarm_col].sum() >= 1)

            if label == 1:
                if triggered:
                    tp_events += 1
                else:
                    fn_events += 1
            else:
                if triggered:
                    fp_events += 1
                else:
                    tn_events += 1

    total = tp_events + fn_events + tn_events + fp_events
    sens = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0.0
    spec = tn_events / (tn_events + fp_events) if (tn_events + fp_events) > 0 else 0.0
    acc = (tp_events + tn_events) / total if total > 0 else 0.0
    bacc = (sens + spec) / 2.0

    return {
        'event_accuracy': float(acc),
        'event_balanced_accuracy': float(bacc),
        'event_sensitivity': float(sens),
        'event_specificity': float(spec),
        'tp_events': int(tp_events), 'fn_events': int(fn_events),
        'tn_events': int(tn_events), 'fp_events': int(fp_events),
        'total_events': int(total)
    }


def plot_trajectory_comparison(df_proc: pd.DataFrame, out_path: Path):
    target_rec = None
    for (pt, rec), group in df_proc.groupby(['patient', 'recording']):
        if group['label'].nunique() == 2 and len(group) >= 60:
            target_rec = (pt, rec)
            break
    if target_rec is None:
        target_rec = df_proc.groupby(['patient', 'recording']).size().idxmax()

    sample = df_proc[(df_proc['patient'] == target_rec[0]) & 
                     (df_proc['recording'] == target_rec[1])].reset_index(drop=True)
    time_sec = np.arange(len(sample)) * 5.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(time_sec, sample['prob_raw'], color='#94a3b8', alpha=0.6, linewidth=1.2, label='Raw GAT-GRU Output P(t)')
    ax1.plot(time_sec, sample['prob_smoothed'], color='#2563eb', linewidth=2.2, label='EMA Smoothed Probability (beta=0.7)')
    ax1.axhline(0.5, color='#ef4444', linestyle='--', linewidth=1.2, label='Threshold (tau=0.5)')

    is_preictal = sample['label'] == 1
    ax1.fill_between(time_sec, 0, 1, where=is_preictal, color='#fef08a', alpha=0.4, label='Preictal Window (Ground Truth)')
    ax1.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax1.set_title(f'Temporal Persistence Filter: Patient {target_rec[0]} ({target_rec[1]})', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9)

    ax2.step(time_sec, sample['alarm_consensus'], where='mid', color='#16a34a', linewidth=2.0, label='Consensus Alarm (k=3 of N=5)')
    ax2.step(time_sec, sample['pred_raw'], where='mid', color='#f97316', linestyle=':', alpha=0.7, label='Raw Binary Predictions')
    ax2.fill_between(time_sec, 0, 1.1, where=is_preictal, color='#fef08a', alpha=0.4)
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Alarm State', fontsize=11, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Quiet (0)', 'Alarm (1)'])
    ax2.set_ylim(-0.1, 1.15)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved trajectory figure to: {out_path}')


def main():
    print('=' * 85)
    print('STRATEGY 1: TEMPORAL PERSISTENCE FILTERING & EVENT-LEVEL SMOOTHING')
    print('=' * 85)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f'Predictions not found at {INPUT_CSV}')

    df_raw = pd.read_csv(INPUT_CSV)
    print(f'Loaded {len(df_raw)} windows across {df_raw["patient"].nunique()} unseen patients.')

    configs = [
        {'name': '1. Raw Unfiltered Baseline', 'beta': 0.0, 'th': 0.50, 'N': 1, 'k': 1},
        {'name': '2. EMA Smoothing Only (beta=0.7)', 'beta': 0.7, 'th': 0.50, 'N': 1, 'k': 1},
        {'name': '3. Consensus Voting (3-of-5)', 'beta': 0.0, 'th': 0.50, 'N': 5, 'k': 3},
        {'name': '4. Hybrid EMA + Consensus (beta=0.7, 3-of-5)', 'beta': 0.7, 'th': 0.50, 'N': 5, 'k': 3},
        {'name': '5. Calibrated Hybrid (beta=0.75, 4-of-5, tau=0.55)', 'beta': 0.75, 'th': 0.55, 'N': 5, 'k': 4},
        {'name': '6. High-Specificity Clinical Alarm (beta=0.8, 6-of-10, tau=0.60)', 'beta': 0.80, 'th': 0.60, 'N': 10, 'k': 6},
    ]

    summary_rows = []
    for cfg in configs:
        df_proc = process_dataset(df_raw, beta=cfg['beta'], threshold=cfg['th'], N=cfg['N'], k=cfg['k'])
        c_m = compute_continuous_metrics(
            df_proc['label'].to_numpy(int),
            df_proc['prob_smoothed'].to_numpy(float),
            df_proc['alarm_consensus'].to_numpy(int)
        )
        e_m = compute_event_metrics(df_proc, alarm_col='alarm_consensus')
        summary_rows.append({
            'Configuration': cfg['name'],
            'EMA_beta': cfg['beta'],
            'Threshold': cfg['th'],
            'Consensus_k_N': f"{cfg['k']}/{cfg['N']}",
            'Window_Acc_%': round(c_m['accuracy'] * 100, 2),
            'Window_BAcc_%': round(c_m['balanced_accuracy'] * 100, 2),
            'Window_Sens_%': round(c_m['sensitivity'] * 100, 2),
            'Window_Spec_%': round(c_m['specificity'] * 100, 2),
            'Event_Sens_%': round(e_m['event_sensitivity'] * 100, 2),
            'Event_Spec_%': round(e_m['event_specificity'] * 100, 2),
            'Event_Acc_%': round(e_m['event_accuracy'] * 100, 2),
            'TP_Seizures': e_m['tp_events'],
            'Total_Seizures': e_m['tp_events'] + e_m['fn_events'],
        })

    res_df = pd.DataFrame(summary_rows)
    csv_out = OUTPUT_DIR / 'persistence_filter_results.csv'
    res_df.to_csv(csv_out, index=False)
    print(f'Saved persistence filter benchmark to: {csv_out}')

    best_df = process_dataset(df_raw, beta=0.7, threshold=0.5, N=5, k=3)
    preds_out = OUTPUT_DIR / 'smoothed_predictions.csv'
    best_df.to_csv(preds_out, index=False)
    print(f'Saved smoothed dataset to: {preds_out}')

    fig_out = OUTPUT_DIR / 'persistence_filter_trajectories.png'
    plot_trajectory_comparison(best_df, fig_out)

    print('\n' + '=' * 105)
    print('PERSISTENCE FILTERING PERFORMANCE COMPARISON')
    print('=' * 105)
    print(res_df.to_string(index=False))
    print('=' * 105)

if __name__ == '__main__':
    main()
