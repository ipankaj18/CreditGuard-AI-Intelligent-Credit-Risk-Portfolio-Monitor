"""Train baseline and challenger default models and persist evaluation artifacts."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACT_DIR, RANDOM_STATE, ensure_dirs

NUMERIC = [
    "age", "years_employed", "annual_income", "num_dependents", "years_at_residence",
    "loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag", "bureau_lines",
    "bureau_outstanding_total", "bureau_dpd_mean", "bureau_dpd_max", "bureau_written_off_count",
    "bureau_utilization", "observed_months", "paid_ratio_mean", "repayment_dpd_mean",
    "repayment_dpd_max", "late_rate", "missed_rate", "latest_balance", "latest_dpd",
    "collection_events", "amount_recovered", "collection_escalations", "income_to_loan",
    "customer_tenure_years", "interest_rate_missing", "years_employed_missing",
]
CATEGORICAL = ["employment_type", "region", "city_tier", "housing_type", "loan_type", "disbursal_channel"]
FEATURES = NUMERIC + CATEGORICAL


def ks_score(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def _metrics(y, scores) -> dict[str, float]:
    auc = float(roc_auc_score(y, scores))
    return {"auc": auc, "ks": ks_score(y, scores), "gini": 2 * auc - 1}


def train_models(data: pd.DataFrame | None = None) -> dict:
    ensure_dirs()
    if data is None:
        data = pd.read_parquet(ARTIFACT_DIR / "analysis_dataset.parquet")
    X, y = data[FEATURES], data["target_default"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    baseline = Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    challenger = Pipeline([("prep", preprocessor), ("model", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, max_leaf_nodes=24, l2_regularization=1.0, random_state=RANDOM_STATE))])
    results = {}
    for name, pipeline in (("logistic_regression", baseline), ("hist_gradient_boosting", challenger)):
        pipeline.fit(X_train, y_train)
        scores = pipeline.predict_proba(X_test)[:, 1]
        results[name] = _metrics(y_test, scores)
    winner_name = max(results, key=lambda n: results[n]["auc"])
    winner = baseline if winner_name == "logistic_regression" else challenger
    # Refit the selected pipeline on all observations after unbiased holdout reporting.
    winner.fit(X, y)
    data = data.copy()
    data["risk_score"] = winner.predict_proba(X)[:, 1]
    data["risk_segment"] = pd.cut(data["risk_score"], [-0.01, .1, .2, .4, 1.01], labels=["Low", "Moderate", "High", "Very High"])
    data.to_parquet(ARTIFACT_DIR / "scored_accounts.parquet", index=False)
    joblib.dump(winner, ARTIFACT_DIR / "default_model.joblib")

    sample = X_test.sample(min(1500, len(X_test)), random_state=RANDOM_STATE)
    importance = permutation_importance(winner, sample, y_test.loc[sample.index], n_repeats=4, scoring="roc_auc", random_state=RANDOM_STATE)
    feature_importance = sorted(
        ({"feature": f, "importance": float(v)} for f, v in zip(FEATURES, importance.importances_mean)),
        key=lambda x: x["importance"], reverse=True,
    )
    payload = {"models": results, "selected_model": winner_name, "test_rows": len(X_test), "feature_importance": feature_importance}
    (ARTIFACT_DIR / "model_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = train_models()
    print(json.dumps(result["models"], indent=2))
