from src.ingestion import build_analysis_dataset
from src.model import train_models


if __name__ == "__main__":
    dataset = build_analysis_dataset()
    result = train_models(dataset)
    print(f"Build complete. Selected model: {result['selected_model']}")
