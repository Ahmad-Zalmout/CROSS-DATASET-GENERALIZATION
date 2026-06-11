"""
UCF-Crime Test Evaluation
=========================
Evaluates the UCF-Crime SavedModel on its own test split using pre-extracted
per-frame Inception features (no video decoding, no feature extraction).

Binary framing: 14 UCF-Crime classes → Anomaly/Violence (1) / NonViolence (0)
The model's actual learned boundary is Normal vs. everything anomalous.
NonViolence = Normal class only (index 7).
Violence    = all 13 anomaly classes.
Score       = 1 - P(Normal)  — no arbitrary class subset needed.

Test data:
  - Anomaly : 290 videos listed in Anomaly_Test_Inception.txt
  - Normal  : 50 videos in Normal_Videos_for_Event_Recognition-txt/

Outputs:
  results/ucf_test_metrics.json
  results/ucf_test_predictions.csv
  results/analysis/confusion_matrix_ucf.png
  results/analysis/score_distribution_ucf.png

Usage:
    conda activate public_safety
    python scripts/01_baseline_eval.py
    python scripts/01_baseline_eval.py --threshold 0.3
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
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
UCF_ROOT       = PROJECT_ROOT / 'dataset' / 'UCF-Crime'
MODEL_PATH     = UCF_ROOT / 'Model_Trained' / 'Model_Trained' / 'saved_model' / 'my_model'
FEATURES_ROOT  = (
    UCF_ROOT / 'Per_Frame_Features2D-Inception'
    / 'Per_Frame_Features2D' / 'Per_Frame_Features2D'
)
ANOMALY_TEST_TXT  = UCF_ROOT / 'Anomaly_Test_Inception.txt'
NORMAL_FEATS_DIR  = FEATURES_ROOT / 'Normal_Videos_for_Event_Recognition-txt' / 'Normal_Videos_for_Event_Recognition'
RESULTS_DIR    = PROJECT_ROOT / 'results'
ANALYSIS_DIR   = RESULTS_DIR / 'analysis'

# ── Class mapping ──────────────────────────────────────────────────────────
UCF_CLASSES = [
    'Abuse',         # 0  ← anomaly
    'Arrest',        # 1  ← anomaly
    'Arson',         # 2  ← anomaly
    'Assault',       # 3  ← anomaly
    'Burglary',      # 4  ← anomaly
    'Explosion',     # 5  ← anomaly
    'Fighting',      # 6  ← anomaly
    'Normal',        # 7  ← the ONLY non-violence class
    'RoadAccidents', # 8  ← anomaly
    'Robbery',       # 9  ← anomaly
    'Shooting',      # 10 ← anomaly
    'Shoplifting',   # 11 ← anomaly
    'Stealing',      # 12 ← anomaly
    'Vandalism',     # 13 ← anomaly
]
NORMAL_IDX      = 7   # the single NonViolence class
CATEGORY_TO_IDX = {c: i for i, c in enumerate(UCF_CLASSES)}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_and_pad(txt_path: Path, seq_len: int) -> np.ndarray:
    """Load (T, 1024) feature file and pad/truncate to (seq_len, 1024)."""
    feats = np.loadtxt(str(txt_path), dtype=np.float32)  # (T, 1024)
    T = feats.shape[0]
    if T > seq_len:
        feats = feats[:seq_len]
    elif T < seq_len:
        feats = np.concatenate(
            [feats, np.zeros((seq_len - T, feats.shape[1]), dtype=np.float32)]
        )
    return feats  # (seq_len, 1024)


def predict_binary(model, features: np.ndarray, normal_idx: int, threshold: float):
    x      = features[np.newaxis, ...].astype(np.float32)  # (1, seq_len, 1024)
    logits = model(x, training=False)                        # (1, 14)
    probs  = tf.nn.softmax(logits).numpy()[0]                # (14,)
    score  = float(1.0 - probs[normal_idx])                  # P(anomaly) = 1 - P(Normal)
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
    ax.set_xlabel('Violence Score (sum of violence-class probs)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='UCF-Crime test evaluation (binary Violence/NonViolence)')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Violence score threshold for binary decision (default: 0.5)')
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    print(f'\n[1/3] Loading UCF-Crime model...')
    model = tf.keras.models.load_model(str(MODEL_PATH))
    seq_len = model.input_shape[1]   # 40
    print(f'  Input : {model.input_shape}  →  seq_len = {seq_len}')
    print(f'  Output: {model.output_shape}')

    # ── 2. Build UCF test set ──────────────────────────────────────────────
    print('\n[2/3] Loading UCF-Crime test features...')

    feature_paths, gt_labels, video_names = [], [], []

    # Anomaly test videos — label derived from category folder name
    with open(ANOMALY_TEST_TXT) as f:
        anomaly_entries = [l.strip() for l in f if l.strip()]

    missing_anomaly = 0
    for entry in anomaly_entries:
        # e.g. "Anomaly-Videos-Part-1-txt/Abuse/Abuse028_x264.txt"
        parts = entry.split('/')
        if len(parts) < 3:
            missing_anomaly += 1
            continue
        category  = parts[1]           # 'Abuse'
        feat_path = FEATURES_ROOT / entry

        if not feat_path.exists():
            missing_anomaly += 1
            continue

        class_idx  = CATEGORY_TO_IDX.get(category, -1)
        binary_lbl = 0 if class_idx == NORMAL_IDX else 1  # Normal=0, all anomalies=1

        feature_paths.append(feat_path)
        gt_labels.append(binary_lbl)
        video_names.append(feat_path.stem)

    print(f'  Anomaly test : {len(anomaly_entries)} listed  →  {len(anomaly_entries) - missing_anomaly} found  ({missing_anomaly} missing)')

    # Normal test videos — label = 0 (NonViolence)
    normal_files = sorted(NORMAL_FEATS_DIR.glob('*.txt'))
    for fp in normal_files:
        feature_paths.append(fp)
        gt_labels.append(0)
        video_names.append(fp.stem)

    print(f'  Normal test  : {len(normal_files)} videos')
    print(f'  Total        : {len(feature_paths)} videos  '
          f'({sum(gt_labels)} Violence, {len(gt_labels) - sum(gt_labels)} NonViolence)')

    # ── 3. Inference & metrics ─────────────────────────────────────────────
    print(f'\n[3/3] Running inference (threshold={args.threshold})...')

    scores, predictions, labels = [], [], []
    class_probs_list = []

    for fp, gt in tqdm(zip(feature_paths, gt_labels), total=len(feature_paths), unit='video'):
        feats              = load_and_pad(fp, seq_len)
        score, pred, probs = predict_binary(model, feats, NORMAL_IDX, args.threshold)
        scores.append(score)
        predictions.append(pred)
        labels.append(gt)
        class_probs_list.append(probs)

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
    print('UCF-CRIME TEST RESULTS  (binary Violence/NonViolence)')
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
        'dataset'              : 'UCF-Crime',
        'phase'                : 'baseline',
        'n_samples'            : int(len(labels)),
        'n_violence'           : int(labels.sum()),
        'n_non_violence'       : int((labels == 0).sum()),
        'threshold'            : args.threshold,
        'accuracy'             : round(float(accuracy),  4),
        'f1'                   : round(float(f1),        4),
        'precision'            : round(float(precision), 4),
        'recall'               : round(float(recall),    4),
        'roc_auc'              : round(float(auc),       4),
        'binary_mapping'       : 'score = 1 - P(Normal)',
        'normal_class_index'   : NORMAL_IDX,
        'normal_class_name'    : UCF_CLASSES[NORMAL_IDX],
    }
    with open(RESULTS_DIR / 'ucf_test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        'video'         : video_names,
        'gt_label'      : labels,
        'gt_name'       : ['Violence' if l == 1 else 'NonViolence' for l in labels],
        'violence_score': scores,
        'prediction'    : predictions,
        'pred_name'     : ['Violence' if p == 1 else 'NonViolence' for p in predictions],
        'correct'       : (labels == predictions).astype(int),
    }).to_csv(RESULTS_DIR / 'ucf_test_predictions.csv', index=False)

    print(f'\nSaved to {RESULTS_DIR}:')
    print('  ucf_test_metrics.json')
    print('  ucf_test_predictions.csv')

    # ── Plots ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, accuracy, f1, auc,
                          'UCF-Crime Test (Binary)',
                          ANALYSIS_DIR / 'confusion_matrix_ucf.png')
    plot_score_distribution(scores, labels, args.threshold,
                            'UCF-Crime Test: Violence Score Distribution',
                            ANALYSIS_DIR / 'score_distribution_ucf.png')

    print(f'\nNext: run scripts/00_extract_rlvd_features.py, then scripts/02_rlvd_eval.py')


if __name__ == '__main__':
    main()
