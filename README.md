# Toss NEXT ML Challenge CTR Prediction

## Overview
This repository contains a refactored codebase for the Toss NEXT ML Challenge CTR prediction project.
The workflow is organized into three parts: a CatBoost-based machine learning branch, a Wide & Deep deep learning branch, and a final logit-space ensemble.

## Project Structure
```text
.
├── data/
├── models/
├── outputs/
├── reports/
├── src/
│   ├── dl_common.py
│   ├── dl_infer.py
│   ├── dl_train.py
│   ├── ensemble.py
│   ├── ml_common.py
│   ├── ml_infer.py
│   └── ml_train.py
├── blend_predictions.py
├── infer_dl.py
├── infer_ml.py
├── train_dl.py
└── train_ml.py
```

## Pipeline
### 1) Machine Learning Branch
- Disjoint negative bagging with 10 bags
- Stratified 5-fold CatBoost training
- Sequence, time, history, and interaction feature engineering
- Fold-wise WOE / target / frequency encoding
- Bag1-Fold1 feature pruning and shared kept feature list

### 2) Deep Learning Branch
- Wide & Deep CTR model
- Numeric features + categorical embeddings + sequence LSTM encoder
- 5-fold training with SAM optimizer
- Fold-average inference

### 3) Final Ensemble
- Combines the ML and DL predictions in logit space
- Default weight: 0.75 (ML) / 0.25 (DL)

## Data Files
Place the following files under `data/`:
- `train.parquet`
- `test.parquet`
- `sample_submission.csv`

## How to Run
### Machine Learning
```bash
python train_ml.py
python infer_ml.py
```

### Deep Learning
```bash
python train_dl.py
python infer_dl.py
```

### Ensemble
```bash
python blend_predictions.py
```

## Environment
Recommended environment:
- Python 3.10+
- Windows 11
- NVIDIA GPU for the deep learning branch

Install the main dependencies with:
```bash
pip install -r requirements.txt
```

## Output Files
- `models/ml_ctr/catboost/bXX_fYY.cbm`
- `models/ml_ctr/enc_maps/bXX_fYY.json`
- `models/ml_ctr/kept_features.txt`
- `models/ml_ctr/meta.json`
- `models/wide_deep/wide_deep_fold*.pt`
- `models/wide_deep/wide_deep_meta.json`
- `models/wide_deep/label_encoders.json`
- `outputs/final_submission.csv`
- `outputs/wide_deep_inference.csv`
- `outputs/submission_logit_w75_25.csv`
