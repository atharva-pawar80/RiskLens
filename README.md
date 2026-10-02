# Fraud Detection ML Pipeline

An end-to-end machine learning system that detects fraudulent credit card transactions — from raw data cleaning through model training, experiment tracking, API serving, and containerization.

Built as a hands-on learning project to practice real ML engineering workflows, not just model training in a notebook.

## Problem

Predict whether a transaction is fraudulent (`1`) or legitimate (`0`) using the ULB credit card fraud dataset — 170,799 real, anonymized European transactions, with only **~0.17% fraud** (a genuinely extreme class imbalance).

## Architecture

```
Raw data (CSV)
      │
      ▼
Data Cleaning  ──▶  handled missing values, duplicates, mixed data types
      │
      ▼
EDA  ──▶  class balance, distribution checks, feature-target correlation
      │
      ▼
Model Training  ──▶  Logistic Regression (+ scaling, + class weighting) vs Random Forest
      │              tracked via MLflow (params + metrics per run)
      ▼
Trained Model (fraud_detection.pkl)
      │
      ▼
FastAPI Service  ──▶  /predict endpoint, loads model once at startup
      │
      ▼
Docker Container  ──▶  packaged with all dependencies, runs identically anywhere
```

## Results

| Model | Precision (fraud) | Recall (fraud) | F1 |
|---|---|---|---|
| Logistic Regression | 0.78 | 0.78 | 0.78 |
| Logistic Regression (scaled) | 0.87 | 0.71 | 0.78 |
| Logistic Regression (`class_weight='balanced'`) | 0.05 | 0.98 | 0.10 |
| **Random Forest (final model)** | **0.96** | **0.83** | **0.89** |

**Note on accuracy:** all models score 99%+ accuracy — this is misleading with 0.17% fraud rate. A model predicting "never fraud" would also score ~99.8% while catching zero fraud. Precision/recall on the fraud class are the metrics that actually matter here.

## Tech Stack

- **Data & modeling:** pandas, scikit-learn
- **Experiment tracking:** MLflow
- **API:** FastAPI, Pydantic
- **Containerization:** Docker

## Project Structure

```
fraud-detection/
├── data/
│   └── fraud_cleaned.csv
├── train.py              # data loading, cleaning, training, MLflow logging
├── app.py                 # FastAPI service
├── fraud_detection.pkl    # trained model
├── requirements.txt
├── Dockerfile
└── README.md
```

## How to Run

### Option 1: Docker (recommended — no local setup needed)

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

### Option 2: Local Python

```bash
python -m venv venv
source venv/Scripts/activate      # Windows
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Either way, once running, open **`http://127.0.0.1:8000/docs`** for the interactive API docs.

### Health checks

- `GET /health` reports that the API process is running.
- `GET /ready` reports whether the model is loaded; it returns HTTP 503 if the model is unavailable.

### Example request

`V_features` must contain exactly 28 finite numeric values. `Time`, `Amount`, and each feature must also be finite numbers. Invalid requests return HTTP 422.

```json
POST /predict
{
  "Time": 136131.0,
  "V_features": [-0.224816, 1.821721, 0.131439, ... ],  // 28 values (V1–V28)
  "Amount": 0.9
}
```

Response:
```json
{ "is_fraud": false }
```

### Retraining

```bash
python train.py
```

Logs all runs to MLflow. View experiment history with:
```bash
mlflow ui
```
then open `http://127.0.0.1:5000`.

## What This Project Demonstrates

- Diagnosing and fixing real data quality issues (nulls, duplicates, inconsistent formatting)
- Correctly evaluating a model on extremely imbalanced data (not trusting accuracy alone)
- Comparing algorithms systematically and explaining *why* one outperformed another
- Reproducible experiment tracking instead of manual note-taking
- Debugging a real training-serving shape mismatch, traced back to a data-cleaning oversight
- Packaging a model as a portable, containerized service

## Known Limitations

- Test set has only 58 fraud examples — precision/recall estimates carry some statistical noise
- No monitoring/drift detection yet (planned future addition)
- No feature store — features are simple/pre-transformed (PCA), not engineered live

## Author

Built by Atharv Pawar as a self-directed learning project in ML engineering and MLOps.