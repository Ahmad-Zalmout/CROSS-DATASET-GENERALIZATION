"""
RLVD Out-of-Domain Evaluation
==============================
Runs the UCF-Crime SavedModel on pre-extracted RLVD features and computes
binary Violence/NonViolence metrics.  This is the cross-dataset (out-of-domain)
result — compare it with results/ucf_test_metrics.json to quantify domain shift.

Prerequisites:
  - scripts/01_baseline_eval.py must have run  → results/ucf_test_metrics.json
  - scripts/00_extract_rlvd_features.py must have run → results/rlvd_features/

Binary mapping (same as baseline):
  score       = 1 - P(Normal)   (Normal = class index 7)
  Violence    = score >= threshold
  NonViolence = score <  threshold

Outputs:
  results/rlvd_eval_metrics.json
  results/rlvd_eval_predictions.csv
  results/analysis/confusion_matrix_rlvd.png
  results/analysis/score_distribution_rlvd.png

Usage:
    conda activate public_safety
    python scripts/02_rlvd_eval.py
    python scripts/02_rlvd_eval.py --threshold 0.3
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH    = (PROJECT_ROOT / 'dataset' / 'UCF-Crime'
                 / 'Model_Trained' / 'Model_Trained' / 'saved_model' / 'my_model')
FEATURES_DIR  = PROJECT_ROOT / 'results' / 'rlvd_features'
MANIFEST_PATH = FEATURES_DIR / 'manifest.csv'
RESULTS_DIR   = PROJECT_ROOT / 'results'
ANALYSIS_DIR  = RESULTS_DIR / 'analysis'

NORMAL_IDX = 7   # the single NonViolence class index in the 14-class head


# ── Helpers ────────────────────────────────────────────────────────────────

def predict_binary(model, features: np.ndarray, normal_idx: int, threshold: float):
    x      = features[np.newaxis, ...].astype(np.float32)   # (1, seq_len, 1024)
    logits = model(x, training=False)                         # (1, 14)
    probs  = tf.nn.softmax(logits).numpy()[0]                 # (14,)
    score  = float(1.0 - probs[normal_idx])                   # P(anomaly)
    return score, int(score >= threshold), probs


def plot_confusion_matrix(cm, accuracy, f1, auc, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['NonViolence', 'Violence'],
        yticklabels=['NonViolence', 'Violence'],
        ax=ax,
    )
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{title}\nAcc={accuracy:.3f}  F1={f1:.3f}  AUC={auc:.3f}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def plot_score_distribution(scores, labels, threshold, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels == 0], bins=40, alpha=0.6, label='NonViolence (GT)', color='steelblue')
    ax.hist(scores[labels == 1], bins=40, alpha=0.6, label='Violence (GT)',    color='tomato')
    ax.axvline(x=threshold, color='k', linestyle='--', label=f'Threshold={threshold}')
    ax.set_xlabel('Violence Score (1 - P(Normal))')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def print_comparison(rlvd_metrics: dict) -> None:
    ucf_path = RESULTS_DIR / 'ucf_test_metrics.json'
    if not ucf_path.exists():
        print('\n  (UCF-Crime baseline metrics not found — run 01_baseline_eval.py first)')
        return

    with open(ucf_path) as f:
        ucf = json.load(f)

    keys = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    labels = ['Accuracy', 'F1 (Violence)', 'Precision', 'Recall', 'ROC-AUC']

    print('\n' + '=' * 58)
    print('DOMAIN SHIFT COMPARISON')
    print(f'{"Metric":<18} {"UCF-Crime (in-domain)":>20} {"RLVD (out-of-domain)":>18}')
    print('-' * 58)
    for key, label in zip(keys, labels):
        ucf_val  = ucf.get(key, float('nan'))
        rlvd_val = rlvd_metrics.get(key, float('nan'))
        delta    = rlvd_val - ucf_val
        sign     = '+' if delta >= 0 else ''
        print(f'{label:<18} {ucf_val:>20.4f} {rlvd_val:>18.4f}  ({sign}{delta:.4f})')
    print('=' * 58)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='RLVD out-of-domain evaluation')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Violence score threshold (default: 0.5)')
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    print('\n[1/3] Loading UCF-Crime model...')
    model = tf.keras.models.load_model(str(MODEL_PATH))
    seq_len = model.input_shape[1]
    print(f'  Input : {model.input_shape}  →  seq_len = {seq_len}')
    print(f'  Output: {model.output_shape}')

    # ── 2. Load RLVD features from manifest ───────────────────────────────
    print('\n[2/3] Loading RLVD features...')

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f'Manifest not found: {MANIFEST_PATH}\n'
            'Run scripts/00_extract_rlvd_features.py first.'
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f'  Manifest entries : {len(manifest)}')
    print(f'  Violence         : {(manifest["label"] == 1).sum()}')
    print(f'  NonViolence      : {(manifest["label"] == 0).sum()}')

    feature_paths, gt_labels, video_names = [], [], []
    missing = []

    for _, row in manifest.iterrows():
        feat_path = FEATURES_DIR / row['feature_file']
        if not feat_path.exists():
            missing.append(row['video'])
            continue
        feature_paths.append(feat_path)
        gt_labels.append(int(row['label']))
        video_names.append(Path(row['video']).stem)

    if missing:
        print(f'  WARNING: {len(missing)} feature file(s) missing — skipped')

    print(f'  Evaluating       : {len(feature_paths)} videos')

    # ── 3. Inference & metrics ─────────────────────────────────────────────
    print(f'\n[3/3] Running inference (threshold={args.threshold})...')

    scores, predictions, labels = [], [], []

    for fp, gt in tqdm(zip(feature_paths, gt_labels), total=len(feature_paths), unit='video'):
        feats = np.load(str(fp)).astype(np.float32)   # (40, 1024)

        # Pad or truncate to model's expected seq_len (should already be 40)
        T = feats.shape[0]
        if T > seq_len:
            feats = feats[:seq_len]
        elif T < seq_len:
            feats = np.concatenate(
                [feats, np.zeros((seq_len - T, feats.shape[1]), dtype=np.float32)]
            )

        score, pred, _ = predict_binary(model, feats, NORMAL_IDX, args.threshold)
        scores.append(score)
        predictions.append(pred)
        labels.append(gt)

    scores      = np.array(scores,      dtype=np.float32)
    predictions = np.array(predictions, dtype=np.int32)
    labels      = np.array(labels,      dtype=np.int32)

    # ── Metrics ────────────────────────────────────────────────────────────
    accuracy  = accuracy_score(labels, predictions)
    f1        = f1_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float('nan')

    print('\n' + '=' * 50)
    print('RLVD EVALUATION RESULTS  (out-of-domain)')
    print('=' * 50)
    print(f'  Videos evaluated  : {len(labels)}')
    print(f'  Accuracy          : {accuracy:.4f}')
    print(f'  F1  (Violence)    : {f1:.4f}')
    print(f'  Precision         : {precision:.4f}')
    print(f'  Recall            : {recall:.4f}')
    print(f'  ROC-AUC           : {auc:.4f}')
    print('=' * 50)
    print('\nClassification Report:')
    print(classification_report(labels, predictions,
                                target_names=['NonViolence', 'Violence'],
                                zero_division=0))

    # ── Save results ───────────────────────────────────────────────────────
    metrics = {
        'dataset'            : 'RLVD',
        'phase'              : 'baseline_ood',
        'n_samples'          : int(len(labels)),
        'n_violence'         : int(labels.sum()),
        'n_non_violence'     : int((labels == 0).sum()),
        'threshold'          : args.threshold,
        'accuracy'           : round(float(accuracy),  4),
        'f1'                 : round(float(f1),        4),
        'precision'          : round(float(precision), 4),
        'recall'             : round(float(recall),    4),
        'roc_auc'            : round(float(auc),       4),
        'binary_mapping'     : 'score = 1 - P(Normal)',
        'normal_class_index' : NORMAL_IDX,
    }
    with open(RESULTS_DIR / 'rlvd_eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'video'         : video_names,
        'gt_label'      : labels,
        'gt_name'       : ['Violence' if l == 1 else 'NonViolence' for l in labels],
        'violence_score': scores,
        'prediction'    : predictions,
        'pred_name'     : ['Violence' if p == 1 else 'NonViolence' for p in predictions],
        'correct'       : (labels == predictions).astype(int),
    }).to_csv(RESULTS_DIR / 'rlvd_eval_predictions.csv', index=False)

    print(f'\nSaved to {RESULTS_DIR}:')
    print('  rlvd_eval_metrics.json')
    print('  rlvd_eval_predictions.csv')

    # ── Plots ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, accuracy, f1, auc,
                          'RLVD Evaluation (Out-of-Domain)',
                          ANALYSIS_DIR / 'confusion_matrix_rlvd.png')
    plot_score_distribution(scores, labels, args.threshold,
                            'RLVD: Violence Score Distribution',
                            ANALYSIS_DIR / 'score_distribution_rlvd.png')

    # ── Domain shift comparison ────────────────────────────────────────────
    print_comparison(metrics)

    print(f'\nNext: python scripts/03_finetune.py')


if __name__ == '__main__':
    main()
