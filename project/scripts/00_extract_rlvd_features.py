"""
RLVD Feature Extraction
=======================
Extracts per-frame GoogLeNet (InceptionV1) features from a stratified sample
of 200 Real-Life Violence Dataset videos (100 Violence + 100 NonViolence) and
saves them as .npy files for use in downstream evaluation and fine-tuning.

Each saved file: (40, 1024) float32 array  — 40 frames × 1024-d features.
40 matches the UCF-Crime model's fixed input sequence length.

Outputs:
  results/rlvd_features/<video_stem>.npy   — one file per video
  results/rlvd_features/manifest.csv       — video name, label, split (train/val/test)

Usage:
    conda activate public_safety
    python scripts/00_extract_rlvd_features.py
    python scripts/00_extract_rlvd_features.py --n_videos 400   # larger sample
    python scripts/00_extract_rlvd_features.py --force          # re-extract existing
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RLVD_ROOT    = PROJECT_ROOT / 'dataset' / 'Real Life Violence Dataset'
CACHE_DIR    = PROJECT_ROOT / 'results' / 'rlvd_features'

# Fixed sequence length must match the UCF-Crime model input: (None, 40, 1024)
SEQ_LEN    = 40
BATCH_SIZE = 16
SEED       = 42


# ── GoogLeNet feature extractor ────────────────────────────────────────────

class GoogLeNetFeatureExtractor(nn.Module):
    """GoogLeNet (InceptionV1) up to avgpool → 1024-d feature per frame."""

    def __init__(self):
        super().__init__()
        backbone = tv_models.googlenet(weights=tv_models.GoogLeNet_Weights.DEFAULT)
        self.conv1       = backbone.conv1
        self.maxpool1    = backbone.maxpool1
        self.conv2       = backbone.conv2
        self.conv3       = backbone.conv3
        self.maxpool2    = backbone.maxpool2
        self.inception3a = backbone.inception3a
        self.inception3b = backbone.inception3b
        self.maxpool3    = backbone.maxpool3
        self.inception4a = backbone.inception4a
        self.inception4b = backbone.inception4b
        self.inception4c = backbone.inception4c
        self.inception4d = backbone.inception4d
        self.inception4e = backbone.inception4e
        self.maxpool4    = backbone.maxpool4
        self.inception5a = backbone.inception5a
        self.inception5b = backbone.inception5b
        self.avgpool     = backbone.avgpool  # → (B, 1024, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x);    x = self.maxpool1(x)
        x = self.conv2(x);    x = self.conv3(x);     x = self.maxpool2(x)
        x = self.inception3a(x); x = self.inception3b(x); x = self.maxpool3(x)
        x = self.inception4a(x); x = self.inception4b(x)
        x = self.inception4c(x); x = self.inception4d(x)
        x = self.inception4e(x); x = self.maxpool4(x)
        x = self.inception5a(x); x = self.inception5b(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # (B, 1024)


PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Frame sampling & feature extraction ───────────────────────────────────

def sample_frames(video_path: Path, n_frames: int) -> list | None:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames or None


def extract_features(
    video_path: Path,
    extractor: nn.Module,
    seq_len: int,
    batch_size: int,
) -> np.ndarray | None:
    """Returns (seq_len, 1024) float32 array, zero-padded if video is short."""
    frames = sample_frames(video_path, seq_len)
    if not frames:
        return None

    tensors = torch.stack([PREPROCESS(f) for f in frames])  # (T, 3, 224, 224)
    parts   = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            parts.append(extractor(tensors[i : i + batch_size]).numpy())
    feats = np.concatenate(parts, axis=0).astype(np.float32)  # (T, 1024)

    # Pad to exactly seq_len (truncate is handled by sample_frames)
    T = feats.shape[0]
    if T < seq_len:
        feats = np.concatenate(
            [feats, np.zeros((seq_len - T, feats.shape[1]), dtype=np.float32)]
        )
    return feats  # (seq_len, 1024)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Extract GoogLeNet features from RLVD videos')
    p.add_argument('--n_videos', type=int, default=200,
                   help='Total videos to sample — must be even (default: 200 → 100V+100NV)')
    p.add_argument('--seed',     type=int, default=SEED,
                   help='Random seed for stratified sampling (default: 42)')
    p.add_argument('--force',    action='store_true',
                   help='Re-extract even if .npy file already exists')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    n_per_class = args.n_videos // 2

    # ── Stratified sample ──────────────────────────────────────────────────
    violence_videos     = sorted((RLVD_ROOT / 'Violence').glob('*.mp4'))
    non_violence_videos = sorted((RLVD_ROOT / 'NonViolence').glob('*.mp4'))

    random.shuffle(violence_videos)
    random.shuffle(non_violence_videos)

    selected_v  = violence_videos[:n_per_class]
    selected_nv = non_violence_videos[:n_per_class]

    print(f'RLVD available   : {len(violence_videos)} Violence, {len(non_violence_videos)} NonViolence')
    print(f'Sampled (seed={args.seed}): {len(selected_v)} Violence, {len(selected_nv)} NonViolence')

    all_videos = [(p, 1) for p in selected_v] + [(p, 0) for p in selected_nv]

    # ── Load extractor ─────────────────────────────────────────────────────
    print('\nLoading GoogLeNet feature extractor...')
    extractor = GoogLeNetFeatureExtractor()
    extractor.eval()
    with torch.no_grad():
        test_out = extractor(torch.randn(1, 3, 224, 224))
    print(f'  Feature dim: {test_out.shape[1]}  (expected 1024)')
    print(f'  Sequence length (seq_len): {SEQ_LEN}')

    # ── Extract & save ─────────────────────────────────────────────────────
    print(f'\nExtracting features → {CACHE_DIR}\n')
    failed  = []
    records = []

    for video_path, label in tqdm(all_videos, unit='video'):
        cache_file = CACHE_DIR / f'{video_path.stem}.npy'

        if cache_file.exists() and not args.force:
            records.append({'video': video_path.name, 'label': label,
                            'label_name': 'Violence' if label else 'NonViolence',
                            'feature_file': cache_file.name, 'status': 'cached'})
            continue

        feats = extract_features(video_path, extractor, SEQ_LEN, BATCH_SIZE)
        if feats is None:
            failed.append(video_path.name)
            continue

        np.save(str(cache_file), feats)
        records.append({'video': video_path.name, 'label': label,
                        'label_name': 'Violence' if label else 'NonViolence',
                        'feature_file': cache_file.name, 'status': 'extracted'})

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = pd.DataFrame(records)
    manifest.to_csv(CACHE_DIR / 'manifest.csv', index=False)

    # Summary
    n_extracted = (manifest['status'] == 'extracted').sum()
    n_cached    = (manifest['status'] == 'cached').sum()
    print(f'\nDone.')
    print(f'  Extracted : {n_extracted}')
    print(f'  Cached    : {n_cached}')
    if failed:
        print(f'  Failed    : {len(failed)}  ({", ".join(failed[:3])}{"..." if len(failed) > 3 else ""})')
    print(f'\n  Manifest  : {CACHE_DIR / "manifest.csv"}')
    print(f'  Features  : {CACHE_DIR}/*.npy  shape=({SEQ_LEN}, 1024)')
    print(f'\nNext: python scripts/02_rlvd_eval.py')


if __name__ == '__main__':
    main()
