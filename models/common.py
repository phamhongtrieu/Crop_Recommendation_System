"""Đường dẫn, đọc dữ liệu và lưu thông tin dùng chung cho các thuật toán."""

import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data" / "processed"


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")


def library_versions() -> dict:
    return {
        "Python": platform.python_version(), "scikit-learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
    }


def read_split(partition: str, data_dir: Path = DATA_DIR) -> tuple:
    X = pd.read_csv(Path(data_dir) / f"X_{partition}.csv")
    y = pd.read_csv(Path(data_dir) / f"y_{partition}.csv")["label"]
    return X, y


def write_json(path: Path, content: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.json")
    temporary_path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
