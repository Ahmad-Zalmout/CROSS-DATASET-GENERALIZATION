# Cross-Dataset Anomaly Detection: Generalisation from UCF-Crime to Real-Life Violence

A capstone data-science project that evaluates how well a surveillance anomaly-detection model trained on UCF-Crime transfers to the Real-Life Violence Situations Dataset (RLVD). The pipeline covers three phases: in-domain baseline evaluation, out-of-domain evaluation (domain shift evidence), and lightweight adaptation via linear probing.

---

## Results

| Phase | Dataset | Accuracy | F1 (Violence) | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (in-domain) | UCF-Crime test | 0.7368 | 0.8485 | 0.7368 | 1.0000 | 0.2197 |
| Baseline (out-of-domain) | RLVD | 0.5000 | 0.6667 | 0.5000 | 1.0000 | 0.4878 |
| Adapted (linear probing) | RLVD test | 0.7667 | 0.7742 | 0.7500 | 0.8000 | 0.8311 |

Adaptation gain on RLVD: Accuracy +26.7 pp, F1 +10.7 pp, Precision +25.0 pp, AUC +34.3 pp.

---

## Repository Structure

```
.
├── cross_dataset_anomaly_detection_report.pdf   ← full project report
├── pre-adaptation_model/                        ← UCF-Crime pre-trained LSTM (TF SavedModel)
│   └── Model_Trained/saved_model/my_model/
├── project/
│   ├── dataset/UCF-Crime/                       ← split lists & annotations (no raw videos)
│   ├── scripts/
│   │   ├── 00_extract_rlvd_features.py          ← extract GoogLeNet features from RLVD videos
│   │   ├── 01_baseline_eval.py                  ← UCF-Crime in-domain evaluation
│   │   ├── 02_rlvd_eval.py                      ← RLVD out-of-domain evaluation
│   │   └── 03_finetune.py                       ← linear probing adaptation
│   ├── results/
│   │   ├── analysis/                            ← confusion matrices & score distribution plots
│   │   ├── rlvd_features/                       ← cached GoogLeNet features (200 RLVD videos)
│   │   ├── adapted_model/                       ← Dense(512) feature extractor (TF SavedModel)
│   │   ├── adapted_logreg.pkl                   ← fitted StandardScaler + LogisticRegression
│   │   ├── ucf_test_metrics.json
│   │   ├── rlvd_eval_metrics.json
│   │   └── adapted_metrics.json
│   └── environment.yml
```

---

## Setup

```bash
conda env create -f project/environment.yml
conda activate public_safety
```

**Key packages:** Python 3.10, TensorFlow 2.15, PyTorch 2.1 (CPU), torchvision 0.16, scikit-learn, OpenCV 4.x, numpy 1.26.

---

## Reproducing the Experiments

Raw datasets are not included in this repository (see [Datasets](#datasets) below).
With the datasets in place, run the scripts in order:

```bash
conda activate public_safety

# Step 1 — UCF-Crime in-domain baseline (uses pre-extracted .txt features)
python project/scripts/01_baseline_eval.py

# Step 2 — Extract GoogLeNet features from 200 RLVD videos (~30 min on CPU)
python project/scripts/00_extract_rlvd_features.py

# Step 3 — RLVD out-of-domain evaluation
python project/scripts/02_rlvd_eval.py

# Step 4 — Linear probing adaptation
python project/scripts/03_finetune.py
```

Cached RLVD features (`project/results/rlvd_features/`) are already included, so Steps 2 and 3 can be skipped if you only want to reproduce the adaptation.

---

## Datasets

| Dataset | Source | Size |
|---|---|---|
| UCF-Crime | [Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) | ~128 hours untrimmed CCTV, 14 classes |
| Real-Life Violence Situations | [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) | 1951 trimmed clips, binary labels |

Place them at:
```
project/dataset/UCF-Crime/
project/dataset/Real Life Violence Dataset/
```

---

## Model Architecture

```
LSTM(1024) → Dense(512) → Dropout → Dense(32) → Dropout → Dense(14)
```

- **Input:** `(batch, 40, 1024)` — 40 frames × 1024-d GoogLeNet (InceptionV1 avgpool) features
- **Output:** 14-class logits (13 UCF-Crime anomaly categories + Normal at index 7)
- **Binary violence score:** `1 - P(Normal)`, threshold 0.5
- **Adaptation:** frozen backbone, Dense(512) output fed into StandardScaler + LogisticRegression (C=1.0)

---

## Loading the Models

### Pre-adaptation model (UCF-Crime LSTM backbone)

```python
import tensorflow as tf
import numpy as np

# Load the pre-trained backbone
backbone = tf.keras.models.load_model(
    "pre-adaptation_model/Model_Trained/saved_model/my_model"
)

# Input shape: (batch, 40, 1024) — 40 frames x 1024-d GoogLeNet features
# Load a cached feature file (shape: 40 x 1024)
features = np.load("project/results/rlvd_features/V_1.npy")   # (40, 1024)
features = features[np.newaxis, ...]                           # (1, 40, 1024)

# Run inference — Normal class is index 7
probs = tf.nn.softmax(backbone(features, training=False)).numpy()  # (1, 14)
violence_score = float(1.0 - probs[0, 7])                         # 1 - P(Normal)
prediction = "Violence" if violence_score >= 0.5 else "NonViolence"

print(f"Violence score: {violence_score:.4f}  →  {prediction}")
```

### Adapted model (Dense(512) extractor + LogisticRegression)

```python
import pickle
import tensorflow as tf
import numpy as np

# Load the Dense(512) feature extractor (frozen backbone, outputs 512-d vectors)
extractor = tf.keras.models.load_model("project/results/adapted_model")

# Load the fitted scaler and classifier (saved together as a dict)
with open("project/results/adapted_logreg.pkl", "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["scaler"]   # sklearn StandardScaler
clf    = bundle["clf"]      # sklearn LogisticRegression

# Load a cached feature file (shape: 40 x 1024)
features = np.load("project/results/rlvd_features/V_1.npy")   # (40, 1024)
features = features[np.newaxis, ...]                           # (1, 40, 1024)

# Extract 512-d representation, standardize, then classify
feat_512   = extractor.predict(features)              # (1, 512)
feat_512_s = scaler.transform(feat_512)               # standardize
violence_prob = clf.predict_proba(feat_512_s)[0, 1]   # P(Violence)
prediction = "Violence" if violence_prob >= 0.5 else "NonViolence"

print(f"Violence probability: {violence_prob:.4f}  →  {prediction}")
```
