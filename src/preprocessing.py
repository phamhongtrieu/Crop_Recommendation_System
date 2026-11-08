"""Các hàm tiền xử lý dùng chung cho dữ liệu khuyến nghị cây trồng."""

import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = (
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
)
TARGET_COLUMN = "label"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu theo các quy tắc trong notebook, giữ nguyên dữ liệu đầu vào."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing_columns}")

    df_clean = df.drop_duplicates().copy()

    for column in FEATURE_COLUMNS:
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    df_clean = df_clean.dropna(subset=required_columns).copy()
    df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].str.strip()

    valid_rows = (
        (df_clean["N"] >= 0)
        & (df_clean["P"] >= 0)
        & (df_clean["K"] >= 0)
        & df_clean["humidity"].between(0, 100)
        & df_clean["ph"].between(0, 14)
        & (df_clean["rainfall"] >= 0)
    )
    return df_clean.loc[valid_rows].copy()


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia train/test có phân tầng theo nhãn để tái lập kết quả giữa các mô hình."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
