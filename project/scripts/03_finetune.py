"""
Classification Head Fine-Tuning on RLVD
=========================================
Adapts the UCF-Crime SavedModel to binary Violence/NonViolence classification
on RLVD via linear probing: the frozen backbone extracts 512-d features from
its Dense(512) layer, and a logistic regression classifier is trained on those
features.

Why logistic regression over a trainable Keras head:
  The Dense(32) bottleneck at the end of the backbone is over-compressed (32-d)
  and trained for 14 UCF-Crime classes — too constrained for a new binary task.
  The Dense(512) layer is richer. Logistic regression on 512-d features is the
  standard "linear probing" approach in transfer learning and is optimal for the
  ~140 training samples available here (no gradient instability, guaranteed
  convergence, well-regularised via C parameter).

Strategy:
  - Freeze entire backbone, extract Dense(512) output as feature vector
  - Fit sklearn LogisticRegression on 70% of 200 RLVD videos
  - Validate on 15%, evaluate on 15%
  - Save a Keras wrapper model for consistent downstream use

Prerequisites:
  - scripts/00_extract_rlvd_features.py must have run → results/rlvd_features/
  - scripts/02_rlvd_eval.py must have run → results/rlvd_eval_metrics.json

Outputs:
  results/adapted_model/             ← Keras feature extractor (SavedModel)
  results/adapted_logreg.pkl         ← fitted LogisticRegression
  results/adapted_metrics.json
  results/adapted_predictions.csv
  results/analysis/confusion_matrix_adapted.png
  results/analysis/score_distribution_adapted.png

Usage:
    conda activate public_safety
    python scripts/03_finetune.py
    python scripts/03_finetune.py --C 0.1 --threshold 0.5
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODEL_PATH     = (PROJECT_ROOT / 'dataset' / 'UCF-Crime'
                  / 'Model_Trained' / 'Model_Trained' / 'saved_model' / 'my_model')
FEATURES_DIR   = PROJECT_ROOT / 'results' / 'rlvd_features'
MANIFEST_PATH  = FEATURES_DIR / 'manifest.csv'
RESULTS_DIR    = PROJECT_ROOT / 'results'
ANALYSIS_DIR   = RESULTS_DIR / 'analysis'
ADAPTED_DIR    = RESULTS_DIR / 'adapted_model'

SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────

def load_raw_features(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load cached (40, 1024) per-video features and labels."""
    X, y = [], []
    for _, row in manifest.iterrows():
        path = FEATURES_DIR / row['feature_file']
        if path.exists():
            X.append(np.load(str(path)).astype(np.float32))
            y.append(int(row['label']))
    return np.stack(X), np.array(y, dtype=np.int32)


def build_feature_extractor(backbone: tf.keras.Model) -> tf.keras.Model:
    """
    Build a model that outputs the Dense(512) layer activations.
    Backbone layer order:
      InputLayer → LSTM(1024) → Dense(512) → Dropout → Dense(32) → Dropout → Dense(14)
    Dense(512) is at index 2 (0=Input, 1=LSTM, 2=Dense512).
    """
    # Find the Dense(512) layer by scanning for the right output shape
    dense512_layer = None
    for layer in backbone.layers:
        if hasattr(layer, 'units') and layer.units == 512:
            dense512_layer = layer
            break
    if dense512_layer is None:
        raise RuntimeError('Could not find Dense(512) layer in backbone.')

    extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=dense512_layer.output,
        name='feature_extractor',
    )
    extractor.trainable = False
    return extractor


def plot_confusion_matrix(cm, accuracy, f1, auc, title, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
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
    ax.hist(scores[labels == 0], bins=20, alpha=0.6, label='NonViolence (GT)', color='steelblue')
    ax.hist(scores[labels == 1], bins=20, alpha=0.6, label='Violence (GT)',    color='tomato')
    ax.axvline(x=threshold, color='k', linestyle='--', label=f'Threshold={threshold}')
    ax.set_xlabel('Violence Probability (logistic regression)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def print_comparison(adapted_metrics: dict) -> None:
    ood_path = RESULTS_DIR / 'rlvd_eval_metrics.json'
    if not ood_path.exists():
        print('\n  (RLVD baseline OOD metrics not found — run 02_rlvd_eval.py first)')
        return

    with open(ood_path) as f:
        ood = json.load(f)

    keys   = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    labels = ['Accuracy', 'F1 (Violence)', 'Precision', 'Recall', 'ROC-AUC']

    print('\n' + '=' * 62)
    print('ADAPTATION GAIN (RLVD test split)')
    print(f'{"Metric":<18} {"Before (OOD baseline)":>22} {"After (adapted)":>18}')
    print('-' * 62)
    for key, label in zip(keys, labels):
        before = ood.get(key, float('nan'))
        after  = adapted_metrics.get(key, float('nan'))
        delta  = after - before
        sign   = '+' if delta >= 0 else ''
        print(f'{label:<18} {before:>22.4f} {after:>18.4f}  ({sign}{delta:.4f})')
    print('=' * 62)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Linear probing adaptation on RLVD (LogisticRegression on Dense(512) features)')
    p.add_argument('--C',         type=float, default=1.0,  help='Logistic regression regularisation (default: 1.0)')
    p.add_argument('--threshold', type=float, default=0.5,  help='Decision threshold (default: 0.5)')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load backbone ───────────────────────────────────────────────────
    print('\n[1/5] Loading UCF-Crime backbone...')
    backbone = tf.keras.models.load_model(str(MODEL_PATH))
    print(f'  Backbone input : {backbone.input_shape}')
    print(f'  Backbone output: {backbone.output_shape}')

    # ── 2. Build feature extractor (Dense(512) output) ────────────────────
    print('\n[2/5] Building feature extractor (Dense(512) layer output)...')
    extractor = build_feature_extractor(backbone)
    print(f'  Feature extractor output: {extractor.output_shape}  (512-d per video)')

    # ── 3. Load RLVD raw features & extract 512-d representations ─────────
    print('\n[3/5] Loading RLVD features and extracting 512-d backbone representations...')
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f'Manifest not found: {MANIFEST_PATH}\n'
            'Run scripts/00_extract_rlvd_features.py first.'
        )
    manifest = pd.read_csv(MANIFEST_PATH)
    X_raw, y = load_raw_features(manifest)
    print(f'  Raw features : {X_raw.shape}')

    # Run frozen backbone to get 512-d features for every video
    X_512 = extractor.predict(X_raw, batch_size=16, verbose=1)   # (N, 512)
    print(f'  Extracted    : {X_512.shape}  (512-d per video)')
    print(f'  Violence: {int(y.sum())}   NonViolence: {int((y == 0).sum())}')

    # ── 4. Train / val / test split (70 / 15 / 15) ────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_512, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15 / 0.85,
        stratify=y_trainval,
        random_state=SEED,
    )
    print(f'\n  Train : {X_train.shape[0]}  ({int(y_train.sum())}V / {int((y_train==0).sum())}NV)')
    print(f'  Val   : {X_val.shape[0]}  ({int(y_val.sum())}V / {int((y_val==0).sum())}NV)')
    print(f'  Test  : {X_test.shape[0]}  ({int(y_test.sum())}V / {int((y_test==0).sum())}NV)')

    # ── 5. Fit logistic regression ─────────────────────────────────────────
    print(f'\n[4/5] Fitting LogisticRegression (C={args.C})...')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(C=args.C, max_iter=1000, random_state=SEED, solver='lbfgs')
    clf.fit(X_train_s, y_train)

    val_acc = clf.score(X_val_s, y_val)
    print(f'  Val accuracy : {val_acc:.4f}')

    # ── 6. Evaluate on test set ────────────────────────────────────────────
    print(f'\n[5/5] Evaluating on test set (threshold={args.threshold})...')
    scores      = clf.predict_proba(X_test_s)[:, 1]   # P(Violence)
    predictions = (scores >= args.threshold).astype(int)
    labels      = y_test.astype(int)

    accuracy  = accuracy_score(labels, predictions)
    f1        = f1_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float('nan')

    print('\n' + '=' * 50)
    print('ADAPTED MODEL — TEST SET RESULTS')
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

    # ── Save extractor + classifier ────────────────────────────────────────
    extractor.save(str(ADAPTED_DIR))
    with open(RESULTS_DIR / 'adapted_logreg.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'clf': clf}, f)
    print(f'  Feature extractor saved → {ADAPTED_DIR}')
    print(f'  LogisticRegression saved → {RESULTS_DIR}/adapted_logreg.pkl')

    # ── Save metrics ───────────────────────────────────────────────────────
    adapted_metrics = {
        'dataset'            : 'RLVD',
        'phase'              : 'adapted',
        'n_train'            : int(X_train.shape[0]),
        'n_val'              : int(X_val.shape[0]),
        'n_test'             : int(X_test.shape[0]),
        'C'                  : args.C,
        'threshold'          : args.threshold,
        'val_accuracy'       : round(float(val_acc),   4),
        'accuracy'           : round(float(accuracy),  4),
        'f1'                 : round(float(f1),        4),
        'precision'          : round(float(precision), 4),
        'recall'             : round(float(recall),    4),
        'roc_auc'            : round(float(auc),       4),
        'adaptation_strategy': 'linear probing: frozen backbone + LogisticRegression on Dense(512) features',
        'feature_layer'      : 'Dense(512)',
        'feature_dim'        : 512,
    }
    with open(RESULTS_DIR / 'adapted_metrics.json', 'w') as f:
        json.dump(adapted_metrics, f, indent=2)

    pd.DataFrame({
        'gt_label'      : labels,
        'gt_name'       : ['Violence' if l == 1 else 'NonViolence' for l in labels],
        'violence_score': scores,
        'prediction'    : predictions,
        'pred_name'     : ['Violence' if p == 1 else 'NonViolence' for p in predictions],
        'correct'       : (labels == predictions).astype(int),
    }).to_csv(RESULTS_DIR / 'adapted_predictions.csv', index=False)

    print(f'\nSaved to {RESULTS_DIR}:')
    print('  adapted_metrics.json')
    print('  adapted_predictions.csv')
    print('  adapted_logreg.pkl')

    # ── Plots ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, accuracy, f1, auc,
                          'Adapted Model — RLVD Test Set',
                          ANALYSIS_DIR / 'confusion_matrix_adapted.png')
    plot_score_distribution(scores, labels, args.threshold,
                            'Adapted Model: Violence Score Distribution (Test Set)',
                            ANALYSIS_DIR / 'score_distribution_adapted.png')

    # ── Domain shift comparison ────────────────────────────────────────────
    print_comparison(adapted_metrics)

    print(f'\nDone. Review results/analysis/ for plots.')


if __name__ == '__main__':
    main()
