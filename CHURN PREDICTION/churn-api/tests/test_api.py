"""
API test suite — runs against the FastAPI app in-process via httpx.
"""

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from app.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure": 8,
    "monthly_charges": 75.50,
    "total_charges": 604.00,
    "num_products": 2,
    "support_calls": 4,
    "days_since_login": 30,
    "has_tech_support": 0,
    "plan": "standard",
    "contract_type": "month-to-month",
    "payment_method": "e-check",
}

LOW_RISK_CUSTOMER = {
    "tenure": 48,
    "monthly_charges": 85.0,
    "total_charges": 4080.0,
    "num_products": 4,
    "support_calls": 0,
    "days_since_login": 2,
    "has_tech_support": 1,
    "plan": "premium",
    "contract_type": "two-year",
    "payment_method": "credit_card",
}


class TestGeneral:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "endpoints" in r.json()

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_model_info(self):
        r = client.get("/model/info")
        assert r.status_code == 200
        d = r.json()
        assert "metrics" in d
        assert "feature_schema" in d


class TestSinglePrediction:
    def test_valid_prediction(self):
        r = client.post("/predict", json=VALID_CUSTOMER)
        assert r.status_code == 200
        d = r.json()
        assert 0 <= d["churn_probability"] <= 1
        assert d["risk_tier"] in {"low", "medium", "high"}
        assert isinstance(d["churn_prediction"], bool)

    def test_low_risk_customer(self):
        r = client.post("/predict", json=LOW_RISK_CUSTOMER)
        assert r.status_code == 200
        assert r.json()["risk_tier"] == "low"

    def test_invalid_plan_rejected(self):
        bad = {**VALID_CUSTOMER, "plan": "elite"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_contract_rejected(self):
        bad = {**VALID_CUSTOMER, "contract_type": "quarterly"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_negative_tenure_rejected(self):
        bad = {**VALID_CUSTOMER, "tenure": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_probability_in_range(self):
        for _ in range(10):
            r = client.post("/predict", json=VALID_CUSTOMER)
            p = r.json()["churn_probability"]
            assert 0 <= p <= 1, f"Out-of-range probability: {p}"


class TestBatchPrediction:
    def test_batch_single(self):
        r = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER]})
        assert r.status_code == 200
        d = r.json()
        assert d["total"] == 1
        assert len(d["predictions"]) == 1

    def test_batch_multiple(self):
        customers = [VALID_CUSTOMER, LOW_RISK_CUSTOMER, VALID_CUSTOMER]
        r = client.post("/predict/batch", json={"customers": customers})
        assert r.status_code == 200
        d = r.json()
        assert d["total"] == 3
        assert "high_risk_count" in d
        assert "processing_time_ms" in d

    def test_batch_empty_rejected(self):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 422

    def test_batch_indices_sequential(self):
        customers = [VALID_CUSTOMER] * 5
        r = client.post("/predict/batch", json={"customers": customers})
        indices = [p["index"] for p in r.json()["predictions"]]
        assert indices == list(range(5))


class TestFeatureExplain:
    def test_known_feature(self):
        r = client.get("/predict/explain/tenure")
        assert r.status_code == 200
        d = r.json()
        assert "matches" in d
        assert len(d["matches"]) > 0

    def test_unknown_feature(self):
        r = client.get("/predict/explain/nonexistent_feature_xyz")
        assert r.status_code == 404
