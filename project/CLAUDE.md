# Cross-Dataset Anomaly Detection — Project Instructions

## Quick Reference

**Task:** Evaluate generalization of a UCF-Crime-trained anomaly detection model on the Real-Life Violence Situations Dataset (RLVD). Binary classification: Violence vs. Non-Violence.

**Active model:** `dataset/UCF-Crime/Model_Trained/Model_Trained/saved_model/my_model`
Architecture: `LSTM(1024) → Dense(512) → Dropout → Dense(32) → Dropout → Dense(14)`
Input: `(batch, T, 1024)` — per-frame GoogLeNet features

**Binary mapping:** `score = 1 - P(Normal)` — Normal (index 7) is the only NonViolence class. All 13 anomaly categories map to Violence. See `docs/notes.md` for the rationale.

**Python env:** `conda activate public_safety` (Python 3.10, TF 2.15, PyTorch 2.11+cpu)

**Script execution order:**
```
conda activate public_safety
python scripts/01_baseline_eval.py           # Step 1: UCF-Crime test evaluation (in-domain)
python scripts/00_extract_rlvd_features.py   # Step 2: extract 200 RLVD features (~30 min)
python scripts/02_rlvd_eval.py               # Step 3: RLVD evaluation (out-of-domain) [TODO]
python scripts/03_finetune.py                # Step 4: fine-tune head on RLVD [TODO]
```

---

## Project Goal

Evaluate cross-dataset generalization of anomaly detection on UCF-Crime → Real-Life Violence transfer. **Binary classification only: Violence vs. Non-Violence.**

## Project Overview

Public safety monitoring via CCTV is a critical ML application. Models trained on one surveillance dataset often fail when deployed in a new environment (different camera angles, lighting, crowd density, incident types) — a problem known as **domain shift**.

This project systematically studies and improves the generalization ability of anomaly detection models across two public-safety datasets:
- **Source:** UCF-Crime Anomaly Detection Dataset (training domain)
- **Target:** Real-Life Violence Situations Dataset (evaluation + adaptation domain)

The goal is to understand *where and why* performance degrades and to apply lightweight adaptations to recover it.

---

## Model Description

The saved model (`dataset/UCF-Crime/Model_Trained/Model_Trained/saved_model/my_model`) is:

```
LSTM(1024) → Dense(512) → Dropout → Dense(32) → Dropout → Dense(14)
```

- **Input:** Variable-length sequence of 1024-d per-frame GoogLeNet (InceptionV1) features → shape `(batch, T, 1024)`
- **Output:** 14-class logits (13 UCF-Crime anomaly categories + Normal)
- **Training features:** `Per_Frame_Features2D-Inception/` — each `.txt` file is `(T, 1024)` with one feature vector per row

### UCF-Crime Class Index Mapping (alphabetical, 0–13)

| Index | Class         | Binary label |
|-------|---------------|--------------|
| 0     | Abuse         | Violence (1) |
| 1     | Arrest        | Violence (1) |
| 2     | Arson         | Violence (1) |
| 3     | Assault       | Violence (1) |
| 4     | Burglary      | Violence (1) |
| 5     | Explosion     | Violence (1) |
| 6     | Fighting      | Violence (1) |
| 7     | **Normal**    | **NonViolence (0)** |
| 8     | RoadAccidents | Violence (1) |
| 9     | Robbery       | Violence (1) |
| 10    | Shooting      | Violence (1) |
| 11    | Shoplifting   | Violence (1) |
| 12    | Stealing      | Violence (1) |
| 13    | Vandalism     | Violence (1) |

**Binary mapping:** `score = 1 - P(Normal)` — Normal (index 7) is the only NonViolence class. All 13 anomaly categories map to Violence. See `docs/notes.md` for the rationale.

> Note: `model.json` in the project root describes an older Keras 1.1.0 binary MIL model (C3D features, 4096-d input). This is NOT the model being used — use `my_model` SavedModel only.

---

## Datasets

### UCF-Crime (Source / Training Reference)
- ~128 hours of untrimmed CCTV-style surveillance video
- 13 anomaly categories + Normal
- Pre-extracted per-frame GoogLeNet features available at `Per_Frame_Features2D-Inception/`
- Pre-extracted global C3D features (4096-d) also available at `Extracted C3D Features/`
- Train/test split files: `Anomaly_Train_Inception.txt`, `Anomaly_Test_Inception.txt`, `Normal_Train_Inception.txt`
- Temporal annotations: `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`

### Real-Life Violence Situations Dataset — RLVD (Target / Evaluation)
- 1000 Violence + 1000 NonViolence short trimmed `.mp4` clips
- Binary classification task (violence vs. non-violence)
- Handheld/non-standard camera viewpoints, varied lighting & backgrounds
- **No pre-extracted features** — must be extracted at runtime using GoogLeNet

---

## Key Constraints & Principles

- **Hardware:** CPU-only, no GPU. Batch sizes 2–16 max.
- **Timeline:** 8–10 days. Lightweight and fast approaches only.
- **Approach:** Use pre-trained model weights as-is for baseline. Fine-tune classification head only (not the full network).
- **Focus:** Demonstrate data science rigor — analysis, metrics, honest documentation of domain shift.

---

## Python Environment

- **Conda env:** `public_safety` (Python 3.10)
- **Activation:** `conda activate public_safety`
- **Jupyter kernel:** `Python (public_safety)` (registered for JupyterLab)
- **Key packages:** TensorFlow 2.15, Keras 2.15, PyTorch 2.11+cpu, torchvision 0.26+cpu, OpenCV 4.13, scikit-learn, numpy 1.26, pandas, matplotlib, seaborn, tqdm, h5py

---

## Workflow

### Phase 1 — Baseline Evaluation

**Step 1 — UCF-Crime test evaluation (`scripts/01_baseline_eval.py`):**
- Load model, load pre-extracted per-frame Inception features from `Per_Frame_Features2D-Inception/`
- Test split: `Anomaly_Test_Inception.txt` (anomaly) + `Normal_Videos_for_Event_Recognition-txt/` (50 normal test videos)
- Map 14 classes → binary, run inference, compute metrics
- Saves `results/ucf_test_metrics.json`, `results/ucf_test_predictions.csv`
- **Known issue:** only 140/290 anomaly test feature files present in dataset (Kaggle download incomplete); 50 normal test videos always available → 190 total test videos

**Step 2 — RLVD feature extraction (`scripts/00_extract_rlvd_features.py`):**
- Stratified sample of 200 RLVD videos (100V + 100NV, seed=42)
- Extracts per-frame GoogLeNet (1024-d) features, saves to `results/rlvd_features/*.npy`
- Also saves `results/rlvd_features/manifest.csv`

**Step 3 — RLVD evaluation (`scripts/02_rlvd_eval.py`):**
- Load extracted RLVD features, run inference with UCF-Crime model
- Same binary mapping as Step 1
- Saves `results/rlvd_eval_metrics.json`
- **This is the out-of-domain result — compare with Step 1 to show domain shift**

### Phase 2 — Lightweight Adaptation (`scripts/03_finetune.py`)
1. Freeze LSTM + Dense(512) layers
2. Replace Dense(14) head with Dense(2) or Dense(1)+sigmoid
3. Fine-tune only the new head on RLVD (train/val/test split)
4. LR = 0.0001, early stopping, 5–10 epochs max
5. Save adapted model to `results/adapted_model/`
6. Save metrics to `results/adapted_metrics.json`

### Phase 3 — Analysis & Visualization
- Side-by-side confusion matrices (baseline vs. adapted)
- Score distribution plots
- t-SNE of feature space (optional, small sample)
- Misclassified examples analysis
- Save to `results/analysis/`

---

## Technical Specifications

- **Feature extraction:** GoogLeNet (InceptionV1) avgpool → 1024-d per frame
- **Frames per video:** 40 (matches model input sequence length)
- **Batch size:** 16 frames per GoogLeNet forward pass (CPU-friendly)
- **Inference:** `model(x, training=False)` → softmax → `1 - P(Normal)`
- **Binary threshold:** 0.5 (configurable via `--threshold`)
- **Metrics:** Accuracy, F1, Precision, Recall, ROC-AUC (sklearn)
- **Feature cache:** `results/rlvd_features/<video_stem>.npy` — re-extraction skipped if exists

---

## Potential Improvements

- **Domain Adaptation / Fine-tuning:** Freeze lower layers, train only classification head; gradual unfreezing
- **Task Alignment:** Map 14 UCF-Crime classes → binary Violence/Non-Violence
- **Temporal Enhancement:** Temporal attention, longer clip windows
- **Data Augmentation:** Motion blur, low light, compression artifacts, temporal jitter
- **Multi-Dataset Training:** Joint training with dataset-specific heads

---

## Expected Results

- Baseline: measurable performance drop vs. trivial baseline (domain shift evidence)
- Post-adaptation: significant gains in F1 and recall on violence class
- Practical guidelines for cross-dataset generalization

---

## Folder Structure

```
project/
├── CLAUDE.md                        ← this file (auto-loaded by Claude Code)
├── dataset/
│   ├── UCF-Crime/
│   │   ├── Extracted C3D Features/         ← (T=32, 4096-d) per video, C3D backbone
│   │   ├── Per_Frame_Features2D-Inception/ ← (T, 1024-d) per video, GoogLeNet backbone ← USED
│   │   ├── Extracted_Features-Inception/   ← global (1024-d) pooled vector per video
│   │   ├── Augmented_2D_Features/
│   │   ├── Model_Trained/Model_Trained/saved_model/my_model/ ← ACTIVE MODEL (TF SavedModel)
│   │   ├── Anomaly_Train_Inception.txt
│   │   ├── Anomaly_Test_Inception.txt      ← 290 listed; only ~140 feature files present
│   │   ├── Normal_Train_Inception.txt
│   │   ├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
│   │   ├── model.json                      ← OLD Keras 1.1.0 MIL model (NOT used)
│   │   └── weights_L1L2.mat
│   └── Real Life Violence Dataset/
│       ├── Violence/       ← 1000 × .mp4 clips  (label=1)
│       └── NonViolence/    ← 951 × .mp4 clips   (label=0)  [49 files missing/non-mp4]
├── docs/
│   └── notes.md                     ← working observations & design decisions (doc draft)
├── scripts/
│   ├── 00_extract_rlvd_features.py  ← extract GoogLeNet features from 200 RLVD videos
│   ├── 01_baseline_eval.py          ← UCF-Crime test evaluation — in-domain baseline
│   ├── 02_rlvd_eval.py              ← RLVD evaluation — out-of-domain
│   └── 03_finetune.py               ← fine-tune classification head on RLVD
├── results/
│   ├── rlvd_features/               ← cached (40, 1024) .npy per RLVD video + manifest.csv
│   ├── ucf_test_metrics.json        ← in-domain UCF-Crime test metrics
│   ├── ucf_test_predictions.csv
│   ├── rlvd_eval_metrics.json       ← out-of-domain RLVD metrics (after script 02)
│   ├── adapted_metrics.json         ← post fine-tuning metrics (after script 03)
│   └── analysis/
│       ├── confusion_matrix_ucf.png
│       ├── score_distribution_ucf.png
│       ├── confusion_matrix_rlvd.png        ← (after script 02)
│       └── score_distribution_rlvd.png      ← (after script 02)
├── environment.yml           ← conda env spec for public_safety
```

---

## What NOT to Do

❌ Train from scratch (weeks on CPU)
❌ Fine-tune the entire LSTM network (overfitting + slow)
❌ Oversell results — domain shift is expected and should be documented honestly
❌ Spend time on architecture innovation

---

## Reporting Expectations

- **Methods:** "Parameter-efficient fine-tuning / classification head adaptation"
- **Results:** Side-by-side metrics table (baseline vs. adapted)
- **Analysis:** Explain *why* improvements happened (not just numbers)
- **Limitations:** Hardware constraints, sample sizes, class mapping assumptions
- **Conclusion:** Practical insights for cross-dataset anomaly detection generalization

---

## Success Criteria

✓ Baseline evaluation showing domain shift (performance drop relative to random/majority)
✓ Lightweight adaptation with documented methodology
✓ Metrics demonstrating any improvement (even marginal)
✓ Analysis explaining domain gap and adaptation strategy
✓ Professional report with data science rigor (splits, metrics, visualizations)

---

## Update Rule

**Whenever the project file structure changes**, update the `Folder Structure` section in this file (`CLAUDE.md`).

---
**Remember:** This project is about demonstrating *understanding of generalization and domain adaptation*, not building the best model. Clear analysis and honest documentation are the priority.
