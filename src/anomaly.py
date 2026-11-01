"""Explainable peer-relative anomaly detection for analyst review."""
from __future__ import annotations

import numpy as np
import pandas as pd


ANOMALY_FEATURES = ["missed_rate", "late_rate", "repayment_dpd_max", "bureau_dpd_max", "bureau_utilization", "bureau_written_off_count"]


def detect_anomalies(data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    frame = data.copy()
    peer_cols = ["loan_type", "income_band"]
    zscores = pd.DataFrame(index=frame.index)
    groups = [frame[c] for c in peer_cols]
    for feature in ANOMALY_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        median = values.groupby(groups, observed=True).transform("median")
        absolute_deviation = (values - median).abs()
        mad = absolute_deviation.groupby(groups, observed=True).transform("median")
        zscores[feature] = (0.6745 * (values - median) / mad.replace(0, np.nan)).clip(lower=0).fillna(0)
    frame["anomaly_score"] = zscores.max(axis=1)
    def explain(index) -> str:
        values = zscores.loc[index].sort_values(ascending=False)
        return "; ".join(
            f"{feature.replace('_', ' ')} is {score:.1f} robust SD above its {frame.at[index, 'loan_type']} / {frame.at[index, 'income_band']} peer median"
            for feature, score in values.items() if score >= threshold
        )[:1000]
    frame["anomaly_reason"] = [explain(index) for index in frame.index]
    frame["is_anomaly"] = frame["anomaly_score"].ge(threshold) & frame["anomaly_reason"].ne("")
    return frame.sort_values("anomaly_score", ascending=False)
