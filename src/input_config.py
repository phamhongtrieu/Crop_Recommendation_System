"""Giới hạn đầu vào dùng chung, lấy trực tiếp từ các đặc trưng trong CSV raw."""

import csv
import math
from pathlib import Path

FEATURE_COLUMNS = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Crop_recommendation.csv"
INPUT_STEPS = {
    "N": 1.0, "P": 1.0, "K": 1.0,
    "temperature": 0.01, "humidity": 0.01, "ph": 0.01, "rainfall": 0.01,
}


def load_input_limits(path: Path = RAW_DATA_PATH) -> dict[str, tuple[float, float]]:
    limits = {feature: [math.inf, -math.inf] for feature in FEATURE_COLUMNS}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not set(FEATURE_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("CSV raw thiếu các đặc trưng bắt buộc.")
        for row in reader:
            for feature in FEATURE_COLUMNS:
                value = float(row[feature])
                if not math.isfinite(value):
                    raise ValueError(f"CSV raw có giá trị không hữu hạn ở cột {feature}.")
                limits[feature][0] = min(limits[feature][0], value)
                limits[feature][1] = max(limits[feature][1], value)
    if any(not math.isfinite(lower) for lower, _ in limits.values()):
        raise ValueError("CSV raw không có mẫu dữ liệu.")
    return {feature: (bounds[0], bounds[1]) for feature, bounds in limits.items()}


def format_bound(value: float) -> str:
    """Hiển thị giới hạn với 2 chữ số thập phân, giữ nguyên giá trị kiểm tra."""
    return f"{value:.2f}"


INPUT_LIMITS = load_input_limits()
