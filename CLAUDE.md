# Capstone Documentation — Session Notes

This file is the working memory for AI sessions helping to write the project documentation.
Read this file at the start of every new session before doing anything else.

---

## Project Summary

**Title:** Cross-Dataset Anomaly Detection — Generalisation from UCF-Crime to Real-Life Violence Situations

**What this is:** A capstone data-science project that evaluates how well a surveillance anomaly-detection model trained on one dataset (UCF-Crime) transfers to a different dataset (Real-Life Violence Situations Dataset, RLVD). The pipeline has three phases: in-domain baseline evaluation, out-of-domain evaluation (domain shift evidence), and lightweight adaptation via linear probing.

**Source code location:** `project/` subfolder (inside this same "Capstone documentation" folder). It contains all scripts, results, and the original CLAUDE.md that was used during development.

**Key files to re-read if context is lost:**
- `project/CLAUDE.md` — full technical specification, model architecture, dataset descriptions, workflow
- `project/docs/notes.md` — working observations and design decisions (feeds directly into the final report)
- `project/scripts/01_baseline_eval.py` — in-domain UCF-Crime evaluation
- `project/scripts/02_rlvd_eval.py` — out-of-domain RLVD evaluation
- `project/scripts/03_finetune.py` — linear probing adaptation (logistic regression on Dense(512) features)
- `project/results/ucf_test_metrics.json` — in-domain results
- `project/results/rlvd_eval_metrics.json` — out-of-domain results
- `project/results/adapted_metrics.json` — post-adaptation results

---

## Actual Results (verified from JSON files)

| Phase | Dataset | Accuracy | F1 (Violence) | Precision | Recall | ROC-AUC |
|-------|---------|----------|---------------|-----------|--------|---------|
| Baseline (in-domain) | UCF-Crime | 0.7368 | 0.8485 | 0.7368 | 1.0000 | 0.2197 |
| Baseline (out-of-domain) | RLVD | 0.5000 | 0.6667 | 0.5000 | 1.0000 | 0.4878 |
| Adapted (linear probing) | RLVD test | 0.7667 | 0.7742 | 0.7500 | 0.8000 | 0.8311 |

**Adaptation gain on RLVD test split:** Accuracy +26.7 pp, F1 +10.7 pp, Precision +25.0 pp, AUC +34.3 pp.

---

## Model Architecture

```
LSTM(1024) → Dense(512) → Dropout → Dense(32) → Dropout → Dense(14)
```
- Input: `(batch, 40, 1024)` — 40 frames × 1024-d GoogLeNet (InceptionV1 avgpool) features
- Output: 14-class logits (13 UCF-Crime anomaly categories + Normal at index 7)
- Binary mapping: `score = 1 - P(Normal)` — threshold 0.5
- Adaptation: frozen backbone → extract Dense(512) output → fit LogisticRegression (C=1.0) on 512-d features

---

## Documentation Chapters

These are the ten chapters that should be written for this project. **Each chapter must be written in its own `.md` file** inside this folder (`Capstone documentation/`). Do not combine chapters. Do not write documentation content in the chat.

### File naming convention

| Chapter | File |
|---------|------|
| 1. Abstract | `01_abstract.md` |
| 2. Introduction | `02_introduction.md` |
| 3. Related Work | `03_related_work.md` |
| 4. Datasets | `04_datasets.md` |
| 5. Model Architecture | `05_model_architecture.md` |
| 6. Methodology | `06_methodology.md` |
| 7. Experiments & Results | `07_experiments_results.md` |
| 8. Analysis & Discussion | `08_analysis_discussion.md` |
| 9. Limitations | `09_limitations.md` |
| 10. Conclusion | `10_conclusion.md` |

### Chapter descriptions

**01 — Abstract**
One page. State the problem (domain shift in surveillance anomaly detection), the approach (LSTM model transfer + linear probing), and the headline numbers (Accuracy 0.50 → 0.77, AUC 0.49 → 0.83). No citations needed.

**02 — Introduction**
- Why CCTV-based anomaly detection matters for public safety
- The domain shift problem: models trained on one dataset fail on another
- Scope: UCF-Crime → RLVD, binary violence classification
- Project contributions (baseline evaluation, domain shift quantification, lightweight adaptation)
- Document roadmap

**03 — Related Work**
- Anomaly detection in surveillance (MIL-based, weakly supervised: Sultani et al. 2018)
- Video feature extraction backbones (C3D, GoogLeNet/InceptionV1, I3D)
- Domain adaptation in video understanding (shallow vs. deep adaptation)
- Transfer learning with frozen backbones / linear probing
- Prior cross-dataset evaluations

**04 — Datasets**
Two subsections:
- UCF-Crime: ~128 hours, 13 anomaly categories + Normal, untrimmed CCTV, GoogLeNet per-frame features (1024-d), train/test splits, incomplete test set (190/340 due to Kaggle download)
- RLVD: 1000 Violence + 951 NonViolence trimmed MP4 clips, handheld cameras, sports/street scenes, no pre-extracted features
- Side-by-side comparison table highlighting the key domain differences (camera type, clip length, annotation granularity, class semantics)

**05 — Model Architecture**
- The LSTM(1024)→Dense(512)→Dense(32)→Dense(14) stack
- GoogLeNet feature extractor (avgpool → 1024-d per frame, 40 frames per video)
- Training setup (original UCF-Crime training, weakly supervised MIL context)
- Why the Dense(512) layer was chosen for linear probing (richer than Dense(32), avoids final 14-class head)
- Note on the deprecated model.json / root saved_model.pb (NOT used)

**06 — Methodology**
Three clear subsections:
1. Binary mapping design: why `1 - P(Normal)` is more principled than summing selected violent-class probabilities. Include the failed first attempt (F1=0.00 with the 5-class sum), the diagnosis (probability mass fragmentation), and the final design (single NonViolence class = Normal index 7). Source: `project/docs/notes.md` Observation 1.
2. Feature extraction pipeline: stratified 200-video RLVD sample (100V+100NV, seed=42), PyTorch GoogLeNet, 40 frames per video, cached `.npy` files.
3. Adaptation strategy: linear probing — freeze full backbone, extract Dense(512) activations, StandardScaler + LogisticRegression (C=1.0, lbfgs), 70/15/15 train/val/test split.

**07 — Experiments & Results**
- Experiment 1: UCF-Crime in-domain baseline (190 videos, 140 anomaly + 50 normal)
- Experiment 2: RLVD out-of-domain evaluation (200 videos, 100V + 100NV)
- Experiment 3: Adapted model on RLVD test split (30 videos)
- Master comparison table (all three rows, all five metrics)
- Describe the complete prediction collapse: model predicts ALL videos as Violence out-of-domain (Precision = 0.50 = chance, Recall = 1.0)

**08 — Analysis & Discussion**
Draw from `project/docs/notes.md` Observations 2 and 3:
- Why domain shift manifests as total NonViolence collapse: UCF-Crime's concept of "Normal" is static CCTV footage; RLVD NonViolence clips are dynamic handheld content
- The frame dilution effect in UCF-Crime: long untrimmed videos dilute anomalies across 40 sampled frames → P(Normal) stays high for crime clips
- AUC inversion on UCF-Crime (0.22) vs. near-random on RLVD (0.49): what each means and why they are not contradictory
- Score distribution analysis: complete overlap on RLVD pre-adaptation vs. separation post-adaptation
- Why linear probing on Dense(512) works: the backbone already encodes motion/appearance features relevant to violence; only the decision boundary needed relearning

**09 — Limitations**
Explicit list, each with a brief explanation:
1. Incomplete UCF-Crime test set (190/340 videos — Kaggle download issue)
2. Small RLVD evaluation sample (200 of 1951 videos) → high metric variance
3. Small adaptation training set (139 videos) → limited generalisation guarantees
4. Semantic mismatch: "all UCF-Crime anomalies = Violence" includes Shoplifting, Arson, Road Accidents which are not physical violence per RLVD's definition
5. CPU-only hardware — limited ability to tune deeper layers or run larger batches
6. No temporal annotations for RLVD — video-level labels only

**10 — Conclusion**
- Summary of findings (domain shift is severe and well-characterised; lightweight adaptation recovers significant performance)
- Practical insight: the right binary mapping matters as much as the model; frozen-backbone linear probing is effective with ~140 samples
- What would be needed for production deployment (more adaptation data, RLVD-native Normal examples)
- Future work: gradual unfreezing, data augmentation, multi-dataset training, test on unseen third dataset

---

## Writing Rules for Future Sessions

1. **Each chapter in its own file.** Use the filenames in the table above. Never merge chapters.
2. **No documentation content in the chat.** The chat is for status updates and questions only. All written content goes directly into the `.md` files in this folder.
3. **Use actual numbers.** All metrics must come from the verified results table above (or the JSON files). Do not approximate or invent numbers.
4. **Source notes.md.** Much of Methodology and Analysis is already drafted in `project/docs/notes.md` — use it as a primary source and elevate the language for the final report.
5. **Academic register.** Write in third-person academic prose. No bullet points inside the document text itself. Sections may have sub-headings.
6. **Figures.** Reference existing PNGs in `project/results/analysis/` where relevant (confusion matrices, score distributions). Do not recreate them unless asked.
7. **One chapter at a time.** Ask the user which chapter to write next. Do not proceed to a new chapter without being told to.
8. **No em dashes.** The em dash character (—) is strictly forbidden anywhere in the documentation. Use alternative constructions instead: rewrite the sentence, use a comma, a colon, or parentheses.

---

## Status Tracker

| Chapter | File | Status |
|---------|------|--------|
| 01 Abstract | `01_abstract.md` | not started |
| 02 Introduction | `02_introduction.md` | not started |
| 03 Related Work | `03_related_work.md` | not started |
| 04 Datasets | `04_datasets.md` | not started |
| 05 Model Architecture | `05_model_architecture.md` | not started |
| 06 Methodology | `06_methodology.md` | not started |
| 07 Experiments & Results | `07_experiments_results.md` | not started |
| 08 Analysis & Discussion | `08_analysis_discussion.md` | not started |
| 09 Limitations | `09_limitations.md` | not started |
| 10 Conclusion | `10_conclusion.md` | not started |

Update this table as chapters are completed.
