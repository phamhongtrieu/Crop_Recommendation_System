"""Chọn model theo CV trong notebook 03 và xuất model dùng cho backend.

Chạy từ thư mục gốc: python -m src.export_model
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
CHECKPOINT_PATH = PROJECT_DIR / "notebooks" / "model_training_results.joblib"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILENAME = "crop_model.joblib"
METADATA_FILENAME = "model_metadata.json"


def select_best_model(artifact: dict) -> tuple[pd.Series, object]:
    """Chọn cấu hình GridSearch có CV F1-macro cao nhất, giống notebook 04."""
    summary = artifact["training_summary"]
    candidates = summary.loc[summary["Configuration"] == "GridSearch"]
    if candidates.empty:
        raise ValueError("Checkpoint không có kết quả GridSearch để chọn model.")
    if not np.isfinite(candidates["CV_f1_macro_mean"].to_numpy(dtype=float)).all():
        raise ValueError("Checkpoint có điểm CV F1-macro không hợp lệ.")

    # Dùng tên model để xử lý đồng hạng, thống nhất với notebook đánh giá.
    best = candidates.sort_values(
        ["CV_f1_macro_mean", "Model"], ascending=[False, True]
    ).iloc[0]
    model = artifact["fitted_models"][best["Model"]][best["Configuration"]]
    return best, model


def export_best_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = MODEL_DIR,
    data_dir: Path = DATA_DIR,
) -> dict:
    """Xuất pipeline đã fit trên train; tập test chỉ dùng đo chất lượng cuối."""
    checkpoint_path, output_dir, data_dir = map(
        Path, (checkpoint_path, output_dir, data_dir)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Chạy notebook 03_model_training.ipynb trước để tạo checkpoint."
        )
    artifact = joblib.load(checkpoint_path)
    if artifact["schema_version"] != 1:
        raise ValueError("Phiên bản checkpoint không được hỗ trợ.")
    if artifact["versions"]["scikit-learn"] != sklearn.__version__:
        raise ValueError(
            "Phiên bản scikit-learn khác lúc huấn luyện. Dùng cùng phiên bản "
            "hoặc chạy lại notebook 03 và 04 trước khi xuất model."
        )
    for filename, expected_hash in artifact["data_hashes"].items():
        data_path = data_dir / filename
        if not data_path.is_file():
            raise FileNotFoundError(f"Thiếu {filename}; chạy notebook 02 và 03 trước.")
        actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{filename} đã thay đổi; chạy lại notebook 03 và 04 trước khi xuất model."
            )

    best, model = select_best_model(artifact)
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")["label"]
    if X_test.columns.tolist() != artifact["features"]:
        raise ValueError("Tên hoặc thứ tự đặc trưng không khớp checkpoint.")
    predictions = model.predict(X_test)
    metric_options = {
        "labels": artifact["classes"], "average": "macro", "zero_division": 0
    }
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, **metric_options)),
        "recall_macro": float(recall_score(y_test, predictions, **metric_options)),
        "f1_macro": float(f1_score(y_test, predictions, **metric_options)),
    }
    metadata = {
        "schema_version": 1,
        "model_file": MODEL_FILENAME,
        "model_name": best["Model"],
        "estimator": type(model.named_steps["model"]).__name__,
        "configuration": best["Configuration"],
        "parameters": json.loads(best["Best_grid_parameters"]),
        "preprocessing": best["Preprocessing"],
        "input_format": "Unscaled feature values in the order listed in features",
        "features": artifact["features"],
        "classes": model.classes_.tolist(),
        "selection_metric": "CV_f1_macro_mean",
        "cv_f1_macro_mean": float(best["CV_f1_macro_mean"]),
        "cv_f1_macro_std": float(best["CV_f1_macro_std"]),
        "test_metrics": test_metrics,
        "train_samples": int(artifact["train_samples"]),
        "test_samples": len(y_test),
        "cv_method": artifact["cv_method"],
        "n_splits": int(artifact["n_splits"]),
        "training_run_id": artifact["run_id"],
        "training_finished_at": artifact["finished_at"],
        "data_hashes": artifact["data_hashes"],
        "versions": artifact["versions"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Lưu nguyên pipeline để giữ đúng tiền xử lý và nhãn cây dạng chuỗi.
    joblib.dump(model, output_dir / MODEL_FILENAME, compress=3)
    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    result = export_best_model()
    print(f"Model được chọn: {result['model_name']} ({result['estimator']})")
    print(f"Tham số: {result['parameters']}")
    print(f"CV F1-macro: {result['cv_f1_macro_mean']:.4%}")
    print(f"Test Accuracy: {result['test_metrics']['accuracy']:.4%}")
    print(f"Test F1-macro: {result['test_metrics']['f1_macro']:.4%}")
    print(f"Đã lưu: models/{MODEL_FILENAME} và models/{METADATA_FILENAME}")
