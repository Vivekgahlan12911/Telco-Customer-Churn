# 🧠 Customer Churn Prediction with ANN (TensorFlow)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green.svg)](.github/workflows/ci.yml)

A **production-ready, modular ANN pipeline** for Customer Churn Prediction, built with TensorFlow/Keras.  
Includes full ML pipeline stages, MLflow tracking, Docker support, and GitHub CI/CD integration.

---

## 📁 Project Structure

```
customer-churn-ann/
├── 📂 config/                  # YAML configs (model, training, data)
├── 📂 data/
│   ├── raw/                    # Original CSV data
│   ├── processed/              # Encoded, scaled, split data
│   └── sample/                 # Auto-generated demo data
├── 📂 src/
│   ├── 📂 data/                # Loader, EDA, feature engineering
│   ├── 📂 models/              # ANN architectures + factory
│   ├── 📂 training/            # Trainer, callbacks, LR scheduler
│   ├── 📂 evaluation/          # Metrics, plots, SHAP explainability
│   ├── 📂 inference/           # Predictor + batch scoring
│   └── 📂 utils/               # Logger, helpers
├── 📂 pipelines/               # End-to-end ML pipelines
├── 📂 scripts/                 # CLI entry points
├── 📂 tests/                   # Unit & integration tests
├── 📂 docker/                  # Dockerfile
└── 📂 .github/workflows/       # CI/CD automation
```

---

## 🚀 Quick Start

```bash
# 1. Clone & Install
git clone https://github.com/yourusername/customer-churn-ann.git
cd customer-churn-ann
pip install -r requirements.txt
pip install -e .

# 2. Generate sample data (or place your own CSV in data/raw/)
python scripts/generate_sample_data.py

# 3. Train
python scripts/train.py --config config/config.yaml

# 4. Evaluate
python scripts/evaluate.py --model_path checkpoints/best_model.keras

# 5. Predict
python scripts/predict.py --input data/sample/new_customers.csv
```

---

## 🏗️ ML Pipeline Stages

```
RAW CSV  →  EDA & Validation  →  Feature Engineering  →  Encoding & Scaling
                                                                  ↓
REPORT  ←  SHAP Explainability  ←  Evaluation  ←  ANN Training  ←  DataLoaders
```

| Stage | Module | Description |
|-------|--------|-------------|
| **Data Ingestion** | `src/data/loader.py` | Load, validate, sample imbalanced data |
| **Feature Engineering** | `src/data/feature_engineering.py` | Encode, scale, create derived features |
| **Model Building** | `src/models/ann_model.py` | Modular ANN with Dropout, BN, residual |
| **Training** | `src/training/trainer.py` | EarlyStopping, LR Scheduler, MLflow |
| **Evaluation** | `src/evaluation/evaluator.py` | AUC, F1, confusion matrix, ROC curve |
| **Explainability** | `src/evaluation/explainer.py` | SHAP feature importance |
| **Inference** | `src/inference/predictor.py` | Single & batch prediction |

---

## 🧠 ANN Architectures

| Model | Hidden Layers | Best AUC |
|-------|---------------|----------|
| `simple_ann` | [64, 32] | ~0.83 |
| `deep_ann` | [256, 128, 64, 32] | ~0.87 |
| `wide_ann` | [512, 256] | ~0.85 |
| `residual_ann` | [128, 128, 64] + skip | ~0.88 |
| `attention_ann` | [256, 128] + self-attn | ~0.89 |

---

## 📊 MLflow Tracking

```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

---

## 🐳 Docker

```bash
docker build -t churn-ann -f docker/Dockerfile .
docker run -v $(pwd)/data:/app/data churn-ann python scripts/train.py
```

---

## ✅ Tests

```bash
pytest tests/ -v --cov=src
```

---

## 📄 License
MIT License
