# ML-Based Churn Prediction API

A production-ready REST API for real-time customer churn prediction, built with FastAPI and scikit-learn.

---

## Architecture

```
churn-api/
├── app/
│   └── main.py            ← FastAPI app (all endpoints)
├── model/
│   └── train.py           ← Training script + feature engineering pipeline
├── tests/
│   └── test_api.py        ← 15 pytest test cases
├── artifacts/             ← Generated after training
│   ├── churn_pipeline.joblib
│   ├── metrics.json
│   └── feature_schema.json
├── dashboard.html         ← Interactive API explorer
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates artifacts/)
python model/train.py

# 3. Start the API
uvicorn app.main:app --reload --port 8000

# 4. Run tests
pytest tests/test_api.py -v

# 5. Open interactive docs
open http://localhost:8000/docs
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model readiness probe |
| GET | `/model/info` | Training metrics, feature schema, CV scores |
| POST | `/predict` | Single customer churn probability |
| POST | `/predict/batch` | Up to 1,000 customers per call |
| GET | `/predict/explain/{feature}` | Feature importance lookup |
| GET | `/docs` | Swagger UI |

---

## Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 8,
    "monthly_charges": 75.50,
    "total_charges": 604.0,
    "num_products": 2,
    "support_calls": 4,
    "days_since_login": 30,
    "has_tech_support": 0,
    "plan": "standard",
    "contract_type": "month-to-month",
    "payment_method": "e-check"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.3241,
  "churn_prediction": false,
  "risk_tier": "medium",
  "model_version": "1.0.0"
}
```

---

## Batch Request

```python
import requests

customers = [
    { "tenure": 8, "monthly_charges": 75.50, ... },
    { "tenure": 48, "monthly_charges": 85.0, ... },
]

r = requests.post("http://localhost:8000/predict/batch",
                  json={"customers": customers})

print(r.json())
# {
#   "predictions": [...],
#   "total": 2,
#   "high_risk_count": 1,
#   "processing_time_ms": 4.3
# }
```

---

## ML Pipeline

```
Input features (10)
     │
     ▼
ColumnTransformer
  ├── StandardScaler  (7 numeric features)
  └── OneHotEncoder   (3 categorical features)
     │
     ▼
GradientBoostingClassifier
  ├── 200 estimators
  ├── learning_rate=0.08
  ├── max_depth=4
  └── subsample=0.8
     │
     ▼
predict_proba → churn probability [0–1]
```

**Input features:**

| Feature | Type | Description |
|---------|------|-------------|
| `tenure` | int | Months as a customer (0–120) |
| `monthly_charges` | float | Monthly bill amount (USD) |
| `total_charges` | float | Lifetime spend (USD) |
| `num_products` | int | Subscribed products (1–10) |
| `support_calls` | int | Calls in last 90 days |
| `days_since_login` | int | Days since last login |
| `has_tech_support` | int | Tech add-on (0 or 1) |
| `plan` | str | basic / standard / premium |
| `contract_type` | str | month-to-month / one-year / two-year |
| `payment_method` | str | credit_card / bank_transfer / e-check / mailed_check |

**Risk tiers:**
- `low` — probability < 0.25
- `medium` — probability 0.25–0.55
- `high` — probability > 0.55

---

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC (test) | 0.640 |
| CV ROC-AUC (5-fold) | 0.643 ± 0.019 |
| Accuracy | 85.5% |
| Churn rate (train) | 13.6% |

---

## Response Headers

Every response includes `X-Processing-Time-Ms` for latency monitoring.

```
X-Processing-Time-Ms: 2.47
```
