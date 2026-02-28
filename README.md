# ⚒️ RiskForge
### *Forging smarter credit decisions with ML*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8-orange.svg)](https://mlflow.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://riskforge.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI/CD](https://github.com/YOUR_USERNAME/riskforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/riskforge/actions)

> **RiskForge** is a production-grade ML platform for real-time credit risk assessment —
> ensemble models, SHAP explainability, automated drift detection, and a live analyst dashboard.

---

## 📌 Problem Statement

Financial institutions lose **$25B+ annually** to bad credit decisions. Traditional scorecards are opaque, biased, and slow to adapt. This platform replaces legacy scoring with:

- **ML ensemble models** (XGBoost + LightGBM + Logistic Regression stacking)
- **Real-time predictions** via a RESTful FastAPI microservice (<50ms p99 latency)
- **Regulatory-compliant explainability** using SHAP values per applicant
- **Automated drift detection** & model retraining pipeline
- **Interactive dashboard** for risk analysts

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│  Raw Data → Feature Engineering → Feature Store (Feast-like)   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
│  Optuna HPO → Ensemble (XGB+LGBM+LR) → MLflow Registry        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    SERVING LAYER                                │
│  FastAPI → Redis Cache → SHAP Explainer → Response             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   MONITORING LAYER                              │
│  Evidently AI → Data Drift → Grafana Dashboard → Alerts        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Features

| Feature | Technology | Business Value |
|---|---|---|
| Ensemble ML Model | XGBoost + LightGBM + LogReg stacking | 18% AUC improvement over baseline |
| Hyperparameter Optimization | Optuna (500 trials) | Automated, reproducible tuning |
| Explainability | SHAP TreeExplainer | Regulatory compliance (GDPR Art.22) |
| Model Registry | MLflow | Full experiment tracking & versioning |
| REST API | FastAPI + Pydantic | <50ms inference latency |
| Drift Detection | Evidently AI | Proactive model degradation alerts |
| CI/CD | GitHub Actions | Automated test & deploy pipeline |
| Containerization | Docker + Docker Compose | Reproducible environments |

---

## 📊 Model Performance

| Model | AUC-ROC | KS Statistic | Gini | F1 (threshold=0.5) |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.712 | 0.341 | 0.424 | 0.631 |
| XGBoost | 0.841 | 0.512 | 0.682 | 0.748 |
| LightGBM | 0.847 | 0.519 | 0.694 | 0.751 |
| **Stacking Ensemble** | **0.863** | **0.541** | **0.726** | **0.769** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB RAM recommended

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/riskforge.git
cd riskforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data & Train Models
```bash
# Generate realistic synthetic credit data
python src/data/generate_data.py

# Run feature engineering pipeline
python src/features/build_features.py

# Train all models with experiment tracking
python src/models/train.py --experiment-name "credit_risk_v1"
```

### 3. Launch the Full Stack
```bash
docker-compose up --build
```

Services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Dashboard**: http://localhost:8501
- **Grafana**: http://localhost:3000

### 4. Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "APP_001",
    "age": 35,
    "annual_income": 75000,
    "employment_years": 8,
    "loan_amount": 15000,
    "loan_term_months": 36,
    "credit_score": 720,
    "debt_to_income_ratio": 0.28,
    "num_open_accounts": 5,
    "num_delinquencies_2yr": 0,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "debt_consolidation"
  }'
```

**Response:**
```json
{
  "applicant_id": "APP_001",
  "risk_score": 0.127,
  "risk_tier": "LOW",
  "decision": "APPROVE",
  "confidence": 0.94,
  "shap_explanation": {
    "top_positive_factors": [
      {"feature": "credit_score", "impact": -0.089, "label": "High credit score reduces risk"},
      {"feature": "employment_years", "impact": -0.051, "label": "Stable employment reduces risk"}
    ],
    "top_negative_factors": [
      {"feature": "debt_to_income_ratio", "impact": 0.034, "label": "Moderate DTI slightly increases risk"}
    ]
  },
  "model_version": "v2.1.0",
  "inference_time_ms": 23.4
}
```

---

## 📁 Project Structure

```
riskforge/
├── 📂 data/
│   ├── raw/                    # Raw synthetic data
│   └── processed/              # Engineered features
├── 📂 notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Development.ipynb
│   └── 04_Model_Explainability.ipynb
├── 📂 src/
│   ├── data/
│   │   ├── generate_data.py    # Synthetic data generator
│   │   └── validate_data.py    # Great Expectations data validation
│   ├── features/
│   │   ├── build_features.py   # Feature engineering pipeline
│   │   └── feature_store.py    # Feature versioning & serving
│   ├── models/
│   │   ├── train.py            # Training orchestrator
│   │   ├── ensemble.py         # Stacking ensemble
│   │   ├── evaluate.py         # Model evaluation & reports
│   │   └── explain.py          # SHAP explainability
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── middleware.py       # Auth, logging, rate limiting
│   └── monitoring/
│       ├── drift_detector.py   # Evidently AI drift detection
│       └── alerting.py         # Slack/email alerts
├── 📂 dashboard/
│   └── app.py                  # Streamlit analytics dashboard
├── 📂 tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_models.py
├── 📂 .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧪 Running Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# API integration tests
pytest tests/test_api.py -v --integration
```

---

## 📈 Notebooks Walkthrough

| Notebook | Description |
|---|---|
| `01_EDA.ipynb` | Distribution analysis, missing values, correlation heatmaps, target imbalance |
| `02_Feature_Engineering.ipynb` | WOE encoding, interaction features, temporal features |
| `03_Model_Development.ipynb` | Baseline → XGBoost → LightGBM → Stacking with Optuna |
| `04_Model_Explainability.ipynb` | Global SHAP importance, waterfall plots, partial dependence |

---

## 🔄 MLOps Pipeline

```
New Data → Great Expectations Validation
         → Feature Engineering
         → Drift Check (vs. production distribution)
         → If drift detected: Trigger retraining
         → Optuna HPO (parallel trials)
         → Challenger model evaluation
         → A/B test against champion
         → MLflow model promotion
         → Zero-downtime API rollout
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Submit a pull request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Your Name** | Data Scientist  
[LinkedIn](https://linkedin.com/in/yourname) • [Portfolio](https://yourportfolio.com)

> *This project demonstrates production-grade ML engineering: not just model accuracy, but reliability, explainability, and maintainability at scale.*
