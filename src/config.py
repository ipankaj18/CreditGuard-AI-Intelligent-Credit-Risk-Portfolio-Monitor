from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in (RAW_DIR, ARTIFACT_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
