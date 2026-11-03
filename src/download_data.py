"""Tải dataset khi chạy: python -m src.download_data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Crop_recommendation.csv"


def download_data(output_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Tải dữ liệu từ Kaggle và lưu CSV vào đường dẫn được chỉ định."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "atharvaingle/crop-recommendation-dataset",
        "Crop_recommendation.csv",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = download_data()
    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
