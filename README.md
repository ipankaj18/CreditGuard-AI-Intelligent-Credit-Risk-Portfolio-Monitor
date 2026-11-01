# CreditGuard AI — Intelligent Credit Risk Portfolio Monitor

A reproducible Streamlit credit-risk monitor built from five synthetic banking tables. It includes portfolio and account views, two evaluated default models, grounded natural-language portfolio Q&A, and explainable peer-relative anomaly review.

## Run locally

Python 3.12.1 was used. From the repository root, the complete workflow is three commands:

```powershell
uv sync
uv run python -m src.build
uv run streamlit run app/app.py
```

The second command creates the clean/scored datasets, model, metrics and data-quality artifacts. Generate the final PDFs after the build with `uv run python scripts/make_docs.py`; run checks with `uv run pytest`.

On a fresh Streamlit Community Cloud deployment, `app/app.py` automatically performs this build when the generated artifacts are absent. The first cold start can therefore take about one to two minutes; subsequent reruns reuse the generated files for the life of the cloud instance.

## LLM configuration

Create a free key at [OpenRouter](https://openrouter.ai/keys), then add `OPENROUTER_API_KEY` and optional `OPENROUTER_MODEL` to the hosted Streamlit secrets store. The default `openrouter/free` router selects an available free model. Never commit key values. Without a key, or when the free tier is rate-limited, Ask the Portfolio still returns its deterministic grounded answer. OpenRouter receives only the analyst question and a precomputed aggregate fact, not raw customer or account rows.

## Architecture and requirement map

| Brief requirement | Location |
|---|---|
| Ingestion, joins, quality decisions | `src/ingestion.py`, `artifacts/data_quality_report.json` |
| Analytics and filters | `app/app.py` → Portfolio |
| Logistic baseline + ML model; AUC/KS/Gini | `src/model.py`, `artifacts/model_metrics.json` |
| Account scores and drivers | `app/app.py` → Account review |
| Grounded LLM Q&A | `src/portfolio_qa.py`, `app/app.py` → Ask the Portfolio |
| Anomaly detection and analyst decision | `src/anomaly.py`, `app/app.py` → Anomaly review |
| Architecture PDF | `docs/architecture.pdf` |
| One-page solution summary | `docs/solution_overview.pdf` |
| Tests | `tests/` |

## Data and join logic

`loan_accounts` defines the final account grain. `customers` joins many-to-one by `customer_id`. Bureau lines are first aggregated to customer level, then joined many-to-one. Repayment months and collections events are independently aggregated to account level and joined one-to-one. The build asserts that final row count equals the original loan count and that `account_id` stays unique.

Intentional quality issues are handled explicitly: `M/F` aliases are normalized; missingness flags are retained; numeric and categorical imputation is learned only from model-training data; missing bureau/collections counts become zero only when absence means no observed activity. Raw rows are never silently dropped.

## Deployment

Deploy `app/app.py` on Streamlit Community Cloud, configure the API key in its secrets manager, verify the public no-login URL, and keep it live for at least 14 days. The local anomaly decisions use session state for demonstration; production requires authenticated durable storage and audit logging.

## Responsible use

Predictions, feature importance and anomaly flags are analyst decision support, not causal explanations or automated adverse actions. Human review, model validation, bias testing, data lineage, monitoring and bank governance are required before production use.
