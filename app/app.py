from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import detect_anomalies
from src.config import ARTIFACT_DIR
from src.ingestion import build_analysis_dataset
from src.model import train_models
from src.portfolio_qa import answer_question

st.set_page_config(page_title="Credit Risk Portfolio Monitor", page_icon="🏦", layout="wide")


def ensure_artifacts() -> None:
    """Create ephemeral build artifacts on a fresh cloud deployment."""
    required = [
        ARTIFACT_DIR / "scored_accounts.parquet",
        ARTIFACT_DIR / "model_metrics.json",
        ARTIFACT_DIR / "default_model.joblib",
    ]
    if all(path.exists() for path in required):
        return
    with st.spinner("Preparing portfolio data and training risk models for this deployment…"):
        try:
            dataset = build_analysis_dataset()
            train_models(dataset)
        except Exception as exc:
            st.error("The initial data/model build failed. Open the Streamlit deployment logs for details.")
            st.exception(exc)
            st.stop()


@st.cache_data
def load_data() -> pd.DataFrame:
    path = ARTIFACT_DIR / "scored_accounts.parquet"
    return pd.read_parquet(path)


@st.cache_data
def load_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    return detect_anomalies(data)


ensure_artifacts()
data = load_data()
metrics = json.loads((ARTIFACT_DIR / "model_metrics.json").read_text(encoding="utf-8"))

st.title("Credit Risk Portfolio Monitor")
st.caption("Account-grain portfolio monitoring · grounded Q&A · explainable peer anomalies")

with st.sidebar:
    st.header("Portfolio filters")
    regions = st.multiselect("Region", sorted(data.region.dropna().unique()), default=[])
    loan_types = st.multiselect("Loan type", sorted(data.loan_type.dropna().unique()), default=[])
    employment = st.multiselect("Employment type", sorted(data.employment_type.dropna().unique()), default=[])
    risk = st.multiselect("Risk segment", ["Low", "Moderate", "High", "Very High"], default=[])
    filtered = data.copy()
    for column, values in (("region", regions), ("loan_type", loan_types), ("employment_type", employment), ("risk_segment", risk)):
        if values:
            filtered = filtered[filtered[column].astype(str).isin(values)]
    st.caption(f"Showing {len(filtered):,} of {len(data):,} accounts")

portfolio_tab, account_tab, ask_tab, anomaly_tab, model_tab = st.tabs(
    ["Portfolio", "Account review", "Ask the Portfolio", "Anomaly review", "Model governance"]
)

with portfolio_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{len(filtered):,}")
    c2.metric("Originated exposure", f"${filtered.loan_amount.sum()/1e6:,.1f}M")
    c3.metric("Observed default rate", f"{filtered.target_default.mean():.2%}" if len(filtered) else "–")
    c4.metric("High / very-high risk", f"{filtered.risk_segment.astype(str).isin(['High','Very High']).mean():.2%}" if len(filtered) else "–")
    left, right = st.columns(2)
    by_loan = filtered.groupby("loan_type", observed=True).agg(accounts=("account_id", "count"), default_rate=("target_default", "mean"), exposure=("loan_amount", "sum")).reset_index()
    left.plotly_chart(px.bar(by_loan, x="loan_type", y="default_rate", color="exposure", title="Default rate and exposure by loan type", labels={"default_rate": "Default rate"}), use_container_width=True)
    by_region = filtered.groupby(["region", "employment_type"], observed=True).agg(default_rate=("target_default", "mean"), accounts=("account_id", "count")).reset_index()
    right.plotly_chart(px.scatter(by_region, x="region", y="default_rate", size="accounts", color="employment_type", title="Cross-dimensional cut: region × employment"), use_container_width=True)
    by_risk = filtered.groupby("risk_segment", observed=True).agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum")).reset_index()
    st.plotly_chart(px.bar(by_risk, x="risk_segment", y="exposure", color="risk_segment", title="Exposure by predicted risk segment", category_orders={"risk_segment": ["Low", "Moderate", "High", "Very High"]}), use_container_width=True)

with account_tab:
    account_id = st.selectbox("Account", filtered.account_id.tolist() if len(filtered) else data.account_id.tolist())
    row = data.loc[data.account_id.eq(account_id)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{row.risk_score:.1%}")
    c2.metric("Risk segment", str(row.risk_segment))
    c3.metric("Loan amount", f"${row.loan_amount:,.0f}")
    c4.metric("Observed outcome", "Default" if row.target_default else "Performing")
    st.subheader("Key account drivers")
    drivers = pd.DataFrame({"Driver": ["Maximum repayment DPD", "Missed-payment rate", "Bureau maximum DPD", "Bureau write-offs", "External utilization"], "Value": [row.repayment_dpd_max, row.missed_rate, row.bureau_dpd_max, row.bureau_written_off_count, row.bureau_utilization]})
    st.dataframe(drivers, hide_index=True, use_container_width=True)
    st.caption("Drivers are observed inputs, not causal claims. Compare them with portfolio feature importance under Model governance.")

with ask_tab:
    st.subheader("Ask the Portfolio")
    st.write("Questions are converted into approved aggregations over the currently filtered account data. An LLM may phrase the verified result but cannot calculate or introduce figures.")
    question = st.text_input("Question", placeholder="Which loan type has the highest default rate?")
    if st.button("Answer", type="primary"):
        try:
            key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
            model = st.secrets.get("OPENROUTER_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/free"))
        except FileNotFoundError:
            key = os.getenv("OPENROUTER_API_KEY", "")
            model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        st.info(answer_question(filtered, question, api_key=key, model=model))
    with st.expander("Controls and human review"):
        st.markdown("""**Trigger:** analyst submits a supported portfolio question.  
**False positive:** a technically correct aggregation is misleading because filters or denominators were misunderstood.  
**False negative:** a valid question is declined because its phrasing is unsupported.  
**Human step:** analyst checks the cited grouping and active filters before using the answer. Unsupported questions are declined. With no OpenRouter key—or when a free model is rate-limited—the verified deterministic answer still works.""")

with anomaly_tab:
    st.subheader("Peer-relative anomaly review")
    anomalies = load_anomalies(filtered)
    flagged = anomalies[anomalies.is_anomaly].head(200)
    st.caption(f"{len(anomalies[anomalies.is_anomaly]):,} accounts flagged at ≥3 robust standard deviations from loan-type and income-band peers.")
    if "reviews" not in st.session_state:
        st.session_state.reviews = {}
    display = flagged[["account_id", "customer_id", "loan_type", "risk_score", "anomaly_score", "anomaly_reason"]]
    st.dataframe(display, hide_index=True, use_container_width=True)
    if len(flagged):
        review_id = st.selectbox("Review flagged account", flagged.account_id.tolist())
        decision = st.radio("Decision", ["Pending", "Confirm", "Dismiss"], horizontal=True, key=f"decision_{review_id}")
        note = st.text_input("Analyst note", key=f"note_{review_id}")
        if st.button("Record review"):
            st.session_state.reviews[review_id] = {"decision": decision, "note": note}
            st.success("Decision recorded for this session.")
    with st.expander("Controls and human review"):
        st.markdown("""**Trigger:** any positive repayment/bureau feature is ≥3 robust SD above its loan-type and income-band peer median.  
**False positive:** a legitimately unusual account is flagged because its peer group is imperfect.  
**False negative:** coordinated or slowly developing deterioration remains within peer norms.  
**Human step:** analyst reviews account drivers and records Confirm/Dismiss. No automated adverse action is taken.""")

with model_tab:
    selected = metrics["selected_model"]
    result = metrics["models"][selected]
    st.subheader("Model performance and governance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Holdout AUC", f"{result['auc']:.3f}")
    c2.metric("Holdout KS", f"{result['ks']:.3f}")
    c3.metric("Holdout Gini", f"{result['gini']:.3f}")
    importance = pd.DataFrame(metrics["feature_importance"][:15])
    st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Permutation importance on holdout sample"), use_container_width=True)
    st.markdown("**Monitoring:** monthly discrimination (AUC/KS/Gini), calibration and observed default by score band; feature/score PSI and missingness; segment fairness checks; override and anomaly-review outcomes; champion–challenger comparison; retraining only after documented validation and approval.")
