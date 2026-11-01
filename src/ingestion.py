"""Load, clean and aggregate the five source tables to account grain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, RAW_DIR, ensure_dirs

FILES = {
    "customers": "customers.csv",
    "loans": "loan_accounts.csv",
    "bureau": "bureau_credit.csv",
    "repayments": "repayment_history.csv",
    "collections": "collections_events.csv",
}


def _read(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame.columns = [prefix + "_" + "_".join(c).strip("_") if isinstance(c, tuple) else prefix + "_" + c for c in frame.columns]
    return frame.reset_index().rename(columns={frame.index.name or "index": "account_id"})


def build_analysis_dataset(raw_dir: Path = RAW_DIR, output: Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    customers = _read("customers", raw_dir)
    loans = _read("loans", raw_dir)
    bureau = _read("bureau", raw_dir)
    repayments = _read("repayments", raw_dir)
    collections = _read("collections", raw_dir)

    # Explicit quality decisions: normalize aliases, preserve missingness flags, and
    # impute later inside the model pipeline to avoid leakage.
    customers["gender"] = customers["gender"].replace({"M": "Male", "F": "Female"})
    for col in ("years_employed", "num_dependents", "years_at_residence"):
        customers[f"{col}_missing"] = customers[col].isna().astype("int8")
    loans["interest_rate_missing"] = loans["interest_rate_pct"].isna().astype("int8")
    bureau["bureau_dpd_missing"] = bureau["days_past_due"].isna().astype("int8")

    for frame, cols in (
        (customers, ["customer_since"]),
        (loans, ["origination_date"]),
        (bureau, ["date_opened", "date_closed"]),
        (collections, ["event_date"]),
    ):
        for col in cols:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")

    # Bureau is customer-grain after aggregation, then safely joins to accounts.
    bureau["is_written_off"] = bureau["account_status"].eq("Written-Off").astype(int)
    bureau["is_active"] = bureau["account_status"].eq("Active").astype(int)
    bureau_agg = bureau.groupby("customer_id").agg(
        bureau_lines=("bureau_id", "count"),
        bureau_sanctioned_total=("sanctioned_amount", "sum"),
        bureau_outstanding_total=("outstanding_amount", "sum"),
        bureau_dpd_mean=("days_past_due", "mean"),
        bureau_dpd_max=("days_past_due", "max"),
        bureau_written_off_count=("is_written_off", "sum"),
        bureau_active_count=("is_active", "sum"),
        bureau_dpd_missing_count=("bureau_dpd_missing", "sum"),
    ).reset_index()
    bureau_agg["bureau_utilization"] = (
        bureau_agg["bureau_outstanding_total"] / bureau_agg["bureau_sanctioned_total"].replace(0, np.nan)
    )

    repayments["paid_ratio"] = repayments["emi_paid"] / repayments["emi_due"].replace(0, np.nan)
    repayments["is_late"] = repayments["payment_status"].eq("Late").astype(int)
    repayments["is_missed"] = repayments["payment_status"].eq("Missed").astype(int)
    repayments = repayments.sort_values(["account_id", "payment_month"])
    repay_agg = repayments.groupby("account_id").agg(
        observed_months=("payment_month", "count"),
        paid_ratio_mean=("paid_ratio", "mean"),
        repayment_dpd_mean=("days_past_due", "mean"),
        repayment_dpd_max=("days_past_due", "max"),
        late_months=("is_late", "sum"),
        missed_months=("is_missed", "sum"),
        latest_balance=("outstanding_balance", "last"),
        latest_dpd=("days_past_due", "last"),
    ).reset_index()
    repay_agg["late_rate"] = repay_agg["late_months"] / repay_agg["observed_months"]
    repay_agg["missed_rate"] = repay_agg["missed_months"] / repay_agg["observed_months"]

    collections["payment_received"] = collections["outcome"].eq("Payment Received").astype(int)
    collections["escalated"] = collections["outcome"].eq("Escalated").astype(int)
    coll_agg = collections.groupby("account_id").agg(
        collection_events=("collection_event_id", "count"),
        amount_recovered=("amount_recovered", "sum"),
        collection_payments=("payment_received", "sum"),
        collection_escalations=("escalated", "sum"),
    ).reset_index()

    data = loans.merge(customers, on="customer_id", how="left", validate="many_to_one")
    data = data.merge(bureau_agg, on="customer_id", how="left", validate="many_to_one")
    data = data.merge(repay_agg, on="account_id", how="left", validate="one_to_one")
    data = data.merge(coll_agg, on="account_id", how="left", validate="one_to_one")
    data["income_to_loan"] = data["annual_income"] / data["loan_amount"].replace(0, np.nan)
    data["customer_tenure_years"] = (data["origination_date"] - data["customer_since"]).dt.days / 365.25
    data["age_band"] = pd.cut(data["age"], [0, 29, 39, 49, 59, 200], labels=["<30", "30-39", "40-49", "50-59", "60+"])
    data["income_band"] = pd.qcut(data["annual_income"], 4, labels=["Q1 Low", "Q2", "Q3", "Q4 High"], duplicates="drop")
    for col in ["bureau_lines", "bureau_sanctioned_total", "bureau_outstanding_total", "bureau_written_off_count", "bureau_active_count", "collection_events", "amount_recovered", "collection_payments", "collection_escalations"]:
        data[col] = data[col].fillna(0)

    if len(data) != len(loans) or data["account_id"].duplicated().any():
        raise ValueError("Account-grain integrity check failed after joins")

    output = output or ARTIFACT_DIR / "analysis_dataset.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    quality = {
        "rows": len(data), "unique_accounts": int(data.account_id.nunique()),
        "unique_customers": int(data.customer_id.nunique()),
        "default_rate": float(data.target_default.mean()),
        "missing_by_column": {k: int(v) for k, v in data.isna().sum().items() if v},
        "decisions": [
            "M/F normalized to Male/Female", "missingness flags retained for known incomplete numeric fields",
            "bureau aggregated to customer grain", "repayments and collections aggregated to account grain",
            "model imputation fitted on training data only", "zero used only where absence means no bureau/collection activity",
        ],
    }
    (ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    frame = build_analysis_dataset()
    print(f"Built {len(frame):,} account rows at {ARTIFACT_DIR / 'analysis_dataset.parquet'}")
