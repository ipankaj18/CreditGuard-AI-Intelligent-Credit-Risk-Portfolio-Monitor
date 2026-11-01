"""Grounded portfolio Q&A: calculate first, optionally let an LLM verbalize."""
from __future__ import annotations

import os
import re

import pandas as pd

ALLOWED_DIMENSIONS = {
    "loan type": "loan_type", "region": "region", "employment": "employment_type",
    "employment type": "employment_type", "age": "age_band", "income": "income_band",
    "risk segment": "risk_segment",
}


def _dimension(question: str) -> str | None:
    q = question.lower()
    return next((column for phrase, column in ALLOWED_DIMENSIONS.items() if phrase in q), None)


def answer_question(data: pd.DataFrame, question: str, api_key: str | None = None, model: str | None = None) -> str:
    q = re.sub(r"\s+", " ", question.strip().lower())
    dimension = _dimension(q)
    if not question.strip():
        return "Enter a portfolio question."
    if data.empty:
        return "No accounts match the active filters, so there is no portfolio result to calculate."
    if "overall" in q and ("default" in q or "risk" in q):
        return f"Overall, {int(data.target_default.sum()):,} of {len(data):,} accounts defaulted: {data.target_default.mean():.2%}. Source: filtered account-level portfolio data."
    if dimension and ("default" in q or "highest" in q or "lowest" in q):
        table = data.groupby(dimension, observed=True).agg(accounts=("account_id", "count"), defaults=("target_default", "sum"), default_rate=("target_default", "mean")).reset_index()
        ascending = "lowest" in q
        table = table.sort_values("default_rate", ascending=ascending)
        row = table.iloc[0]
        fact = f"{row[dimension]} has the {'lowest' if ascending else 'highest'} default rate at {row.default_rate:.2%} ({int(row.defaults):,} defaults across {int(row.accounts):,} accounts). Source: target_default grouped by {dimension}."
    elif dimension and any(term in q for term in ("exposure", "loan amount", "balance")):
        table = data.groupby(dimension, observed=True).agg(accounts=("account_id", "count"), exposure=("loan_amount", "sum")).reset_index().sort_values("exposure", ascending=False)
        row = table.iloc[0]
        fact = f"{row[dimension]} has the largest originated exposure: ${row.exposure:,.0f} across {int(row.accounts):,} accounts. Source: loan_amount grouped by {dimension}."
    elif dimension and any(term in q for term in ("how many", "count", "distribution")):
        table = data.groupby(dimension, observed=True).size().sort_values(ascending=False)
        fact = "; ".join(f"{idx}: {int(value):,}" for idx, value in table.items()) + f". Source: account count grouped by {dimension}."
    else:
        return "I can’t answer that reliably from the available portfolio fields. Try asking about default rate, exposure, or account counts by loan type, region, employment, age, income, or risk segment."

    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return fact
    try:
        from openai import OpenAI
        response = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"X-Title": "CreditGuard AI"},
        ).chat.completions.create(
            model=model or os.getenv("OPENROUTER_MODEL", "openrouter/free"), temperature=0,
            messages=[
                {"role": "system", "content": "Rewrite the supplied verified fact in concise analyst-friendly English. Do not add, infer, or change any number. Preserve the source statement."},
                {"role": "user", "content": f"Question: {question}\nVerified fact: {fact}"},
            ],
        )
        return response.choices[0].message.content or fact
    except Exception:
        return fact + " (OpenRouter unavailable or rate-limited; deterministic grounded answer shown.)"
