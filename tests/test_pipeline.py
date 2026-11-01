import pandas as pd

from src.ingestion import build_analysis_dataset
from src.model import ks_score
from src.portfolio_qa import answer_question


def test_ingestion_preserves_account_grain(tmp_path):
    data = build_analysis_dataset(output=tmp_path / "analysis.parquet")
    loans = pd.read_csv("data/raw/loan_accounts.csv")
    assert len(data) == len(loans)
    assert data.account_id.is_unique
    assert set(data.target_default.unique()) <= {0, 1}


def test_grounded_answer_contains_computed_rate():
    data = pd.DataFrame({"account_id": ["a", "b", "c"], "loan_type": ["A", "A", "B"], "target_default": [1, 0, 0], "loan_amount": [1, 2, 3]})
    answer = answer_question(data, "Which loan type has the highest default rate?")
    assert "50.00%" in answer
    assert "2 accounts" in answer


def test_ks_is_bounded():
    value = ks_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert 0 <= value <= 1
