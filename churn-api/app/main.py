"""
Churn Prediction REST API
FastAPI application exposing prediction, batch, and health endpoints.

New features (v2.0.0):
  - Custom threshold control on /predict and /predict/batch
  - Prediction history log (last 50 predictions in memory)
  - Customer risk summary endpoint GET /customers/summary
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

BASE      = Path(__file__).parent.parent
ARTIFACTS = BASE / "artifacts"

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Real-time and batch churn-probability scoring. "
        "v2.0.0 adds custom thresholds, prediction history, and risk summaries."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline       = None
_metrics: dict  = {}
_schema:  dict  = {}
_history: deque = deque(maxlen=50)


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        model_path = ARTIFACTS / "churn_pipeline.joblib"
        if not model_path.exists():
            raise RuntimeError("Model artifact not found. Run `python model/train.py` first.")
        _pipeline = joblib.load(model_path)
    return _pipeline


def get_metrics() -> dict:
    global _metrics
    if not _metrics:
        p = ARTIFACTS / "metrics.json"
        _metrics = json.loads(p.read_text()) if p.exists() else {}
    return _metrics


def get_schema() -> dict:
    global _schema
    if not _schema:
        p = ARTIFACTS / "feature_schema.json"
        _schema = json.loads(p.read_text()) if p.exists() else {}
    return _schema


class CustomerFeatures(BaseModel):
    tenure:           int   = Field(..., ge=0,  le=120)
    monthly_charges:  float = Field(..., ge=0)
    total_charges:    float = Field(..., ge=0)
    num_products:     int   = Field(..., ge=1,  le=10)
    support_calls:    int   = Field(..., ge=0)
    days_since_login: int   = Field(..., ge=0)
    has_tech_support: int   = Field(..., ge=0,  le=1)
    plan:             str
    contract_type:    str
    payment_method:   str

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, v):
        if v not in {"basic", "standard", "premium"}:
            raise ValueError("plan must be basic | standard | premium")
        return v

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v):
        if v not in {"month-to-month", "one-year", "two-year"}:
            raise ValueError("contract_type must be month-to-month | one-year | two-year")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment(cls, v):
        if v not in {"credit_card", "bank_transfer", "e-check", "mailed_check"}:
            raise ValueError("payment_method must be credit_card | bank_transfer | e-check | mailed_check")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 8, "monthly_charges": 75.50, "total_charges": 604.00,
                "num_products": 2, "support_calls": 4, "days_since_login": 30,
                "has_tech_support": 0, "plan": "standard",
                "contract_type": "month-to-month", "payment_method": "e-check",
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_tier:         str
    threshold_used:    float
    model_version:     str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions:        List[dict]
    total:              int
    high_risk_count:    int
    medium_risk_count:  int
    low_risk_count:     int
    threshold_used:     float
    processing_time_ms: float


class RiskSummaryResponse(BaseModel):
    total_predictions:    int
    high_risk_count:      int
    medium_risk_count:    int
    low_risk_count:       int
    average_churn_prob:   float
    max_churn_prob:       float
    min_churn_prob:       float
    high_risk_percentage: float


def get_risk_tier(prob: float) -> str:
    if prob < 0.25:   return "low"
    elif prob < 0.55: return "medium"
    return "high"


def log_prediction(prob: float, tier: str, source: str = "single"):
    _history.append({
        "timestamp":         datetime.utcnow().isoformat(),
        "churn_probability": prob,
        "risk_tier":         tier,
        "source":            source,
    })


def features_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - start) * 1000
    response.headers["X-Processing-Time-Ms"] = f"{elapsed:.2f}"
    return response


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Churn Prediction API", "version": "2.0.0",
        "endpoints": {
            "health":       "GET  /health",
            "predict":      "POST /predict?threshold=0.5",
            "batch":        "POST /predict/batch?threshold=0.5",
            "explain":      "GET  /predict/explain/{feature}",
            "history":      "GET  /predict/history",
            "clear_history":"DELETE /predict/history",
            "risk_summary": "GET  /customers/summary",
            "model_info":   "GET  /model/info",
            "docs":         "GET  /docs",
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    try:
        get_pipeline(); model_ok = True
    except Exception: model_ok = False
    return JSONResponse(
        content={"status": "healthy" if model_ok else "degraded", "model_loaded": model_ok},
        status_code=200 if model_ok else 503,
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "model_type":     "GradientBoostingClassifier",
        "pipeline_steps": ["ColumnTransformer (StandardScaler + OHE)", "GradientBoostingClassifier (200 trees, depth 4)"],
        "metrics":         get_metrics(),
        "feature_schema":  get_schema(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(
    customer:  CustomerFeatures,
    threshold: float = Query(default=0.5, ge=0.01, le=0.99,
                             description="Decision threshold (default 0.5). Lower = higher recall."),
):
    """Score a single customer. Use ?threshold=0.3 to flag more customers as churners."""
    try:
        prob = float(get_pipeline().predict_proba(features_to_df(customer))[0, 1])
        tier = get_risk_tier(prob)
        log_prediction(round(prob, 4), tier, "single")
        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=prob >= threshold,
            risk_tier=tier,
            threshold_used=threshold,
            model_version="2.0.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    batch:     BatchRequest,
    threshold: float = Query(default=0.5, ge=0.01, le=0.99,
                             description="Decision threshold applied to all customers."),
):
    """Score up to 1,000 customers. Returns per-customer predictions + risk breakdown."""
    try:
        t0    = time.perf_counter()
        df    = pd.DataFrame([c.model_dump() for c in batch.customers])
        probs = get_pipeline().predict_proba(df)[:, 1]

        predictions = [
            {"index": i, "churn_probability": round(float(p), 4),
             "churn_prediction": bool(p >= threshold), "risk_tier": get_risk_tier(float(p))}
            for i, p in enumerate(probs)
        ]
        for pred in predictions:
            log_prediction(pred["churn_probability"], pred["risk_tier"], "batch")

        elapsed = (time.perf_counter() - t0) * 1000
        return BatchPredictionResponse(
            predictions=predictions, total=len(predictions),
            high_risk_count=  sum(1 for p in predictions if p["risk_tier"] == "high"),
            medium_risk_count=sum(1 for p in predictions if p["risk_tier"] == "medium"),
            low_risk_count=   sum(1 for p in predictions if p["risk_tier"] == "low"),
            threshold_used=threshold,
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/explain/{feature}", tags=["Prediction"])
def feature_importance(feature: str):
    """Return relative importance of a feature by name."""
    pipeline  = get_pipeline()
    clf       = pipeline.named_steps["classifier"]
    pre       = pipeline.named_steps["preprocessor"]
    num_names = get_schema().get("numeric_features", [])
    cat_names = list(pre.named_transformers_["cat"].named_steps["ohe"]
                     .get_feature_names_out(get_schema().get("categorical_features", [])))
    importances = {f: round(float(v), 6)
                   for f, v in zip(num_names + cat_names, clf.feature_importances_)}
    matches = {k: v for k, v in importances.items() if feature.lower() in k.lower()}
    if not matches:
        raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found.")
    return {"query": feature, "matches": dict(sorted(matches.items(), key=lambda x: -x[1]))}


# ── NEW FEATURE 1: Prediction History Log ──

@app.get("/predict/history", tags=["Prediction"])
def prediction_history(
    limit: int = Query(default=20, ge=1, le=50, description="Number of recent predictions (max 50)")
):
    """Returns the last N predictions with timestamp, probability, tier, and source."""
    recent = list(reversed(list(_history)))[:limit]
    return {"count": len(recent), "predictions": recent}


@app.delete("/predict/history", tags=["Prediction"])
def clear_history():
    """Clear the in-memory prediction history log."""
    _history.clear()
    return {"message": "Prediction history cleared.", "count": 0}


# ── NEW FEATURE 2: Customer Risk Summary ──

@app.get("/customers/summary", response_model=RiskSummaryResponse, tags=["Customers"])
def customer_risk_summary():
    """
    Aggregate risk stats across all history entries.
    Call /predict or /predict/batch first to populate history.
    """
    if not _history:
        raise HTTPException(
            status_code=404,
            detail="No predictions recorded yet. Call /predict or /predict/batch first."
        )
    probs  = [h["churn_probability"] for h in _history]
    tiers  = [h["risk_tier"]         for h in _history]
    total  = len(probs)
    high   = tiers.count("high")
    return RiskSummaryResponse(
        total_predictions=total,
        high_risk_count=high,
        medium_risk_count=tiers.count("medium"),
        low_risk_count=tiers.count("low"),
        average_churn_prob=round(float(np.mean(probs)), 4),
        max_churn_prob=    round(float(np.max(probs)),  4),
        min_churn_prob=    round(float(np.min(probs)),  4),
        high_risk_percentage=round(high / total * 100,  2),
    )