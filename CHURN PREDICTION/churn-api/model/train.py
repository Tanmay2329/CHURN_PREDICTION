"""
Churn Prediction Model — Training Script
Builds and serializes a scikit-learn pipeline (preprocessing + classifier).
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
import joblib
import json
import os

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────
# 1.  Synthetic dataset  (representative schema)
# ──────────────────────────────────────────────────

def generate_dataset(n: int = 5_000) -> pd.DataFrame:
    plans = np.random.choice(["basic", "standard", "premium"], n, p=[0.4, 0.35, 0.25])
    tenure = np.random.exponential(scale=24, size=n).clip(1, 72).astype(int)
    monthly_charges = np.where(
        plans == "basic",    np.random.normal(30, 5, n),
        np.where(plans == "standard", np.random.normal(55, 8, n),
                                      np.random.normal(85, 12, n))
    ).clip(10, 150)
    total_charges = tenure * monthly_charges * np.random.uniform(0.85, 1.05, n)
    num_products   = np.random.randint(1, 6, n)
    support_calls  = np.random.poisson(lam=1.5, size=n)
    contract_type  = np.random.choice(["month-to-month", "one-year", "two-year"], n, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(["credit_card", "bank_transfer", "e-check", "mailed_check"], n)
    has_tech_support = np.random.choice([0, 1], n, p=[0.6, 0.4])
    days_since_login = np.random.exponential(scale=15, size=n).clip(0, 180).astype(int)

    # churn probability depends on real signals
    churn_logit = (
        -2.0
        + 0.5  * (plans == "basic").astype(float)
        - 0.04 * tenure
        + 0.01 * monthly_charges
        + 0.15 * support_calls
        - 0.3  * (contract_type == "two-year").astype(float)
        + 0.2  * (contract_type == "month-to-month").astype(float)
        - 0.25 * has_tech_support
        + 0.008* days_since_login
        - 0.1  * num_products
        + np.random.normal(0, 0.5, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (np.random.uniform(size=n) < churn_prob).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "num_products": num_products,
        "support_calls": support_calls,
        "days_since_login": days_since_login,
        "has_tech_support": has_tech_support,
        "plan": plans,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "churn": churn,
    })


# ──────────────────────────────────────────────────
# 2.  Feature definitions
# ──────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "support_calls", "days_since_login",
    "has_tech_support",
]
CATEGORICAL_FEATURES = ["plan", "contract_type", "payment_method"]
TARGET = "churn"


# ──────────────────────────────────────────────────
# 3.  Build preprocessing + model pipeline
# ──────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,  NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        random_state=SEED,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])


# ──────────────────────────────────────────────────
# 4.  Train, evaluate, serialise
# ──────────────────────────────────────────────────

def train():
    print("Generating dataset …")
    df = generate_dataset()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Churn rate: {y.mean():.1%}")

    pipeline = build_pipeline()
    print("Training pipeline …")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ──
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc   = roc_auc_score(y_test, y_proba)
    cv    = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nROC-AUC (test): {auc:.4f}")
    print(f"CV ROC-AUC:     {cv.mean():.4f} ± {cv.std():.4f}")
    print("\n" + classification_report(y_test, y_pred))

    metrics = {
        "roc_auc_test":  round(auc, 4),
        "cv_roc_auc_mean": round(float(cv.mean()), 4),
        "cv_roc_auc_std":  round(float(cv.std()),  4),
        "precision_churned": round(report["1"]["precision"], 4),
        "recall_churned":    round(report["1"]["recall"],    4),
        "f1_churned":        round(report["1"]["f1-score"],  4),
        "support_churned":   int(report["1"]["support"]),
        "confusion_matrix":  confusion_matrix(y_test, y_pred).tolist(),
        "churn_rate_train":  round(float(y_train.mean()), 4),
    }

    # ── Persist ──
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/churn_pipeline.joblib")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature schema (for API validation docs)
    schema = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_values": {
            "plan":            ["basic", "standard", "premium"],
            "contract_type":   ["month-to-month", "one-year", "two-year"],
            "payment_method":  ["credit_card", "bank_transfer", "e-check", "mailed_check"],
        },
    }
    with open("artifacts/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("\nArtifacts saved to ./artifacts/")
    return pipeline, metrics


if __name__ == "__main__":
    train()
