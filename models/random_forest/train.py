from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from models.common import library_versions, read_split, timestamp

# 1. Đọc dữ liệu train; Random Forest dùng trực tiếp đặc trưng gốc.
model_dir = Path(__file__).resolve().parent
X_train, y_train = read_split("train")

# 2. Khai báo lưới siêu tham số cần thử.
model = RandomForestClassifier(random_state=42, n_jobs=1)
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt"],
    "criterion": ["gini", "entropy"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Chọn tham số theo F1-macro và fit lại rừng tốt nhất trên toàn bộ train.
search = GridSearchCV(
    model, param_grid, cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    refit="f1_macro", return_train_score=True, n_jobs=1,
)
search.fit(X_train, y_train)

# 4. Lưu mô hình tốt nhất và toàn bộ kết quả tìm kiếm.
joblib.dump(search.best_estimator_, model_dir / "model.joblib", compress=3)
joblib.dump({
    "search": search, "finished_at": timestamp(), "versions": library_versions(),
}, model_dir / "training_results.joblib", compress=3)
for filename in (
    "model_metadata.json", "evaluation_results.json",
    "feature_importance.csv", "feature_importance.png", "feature_importance.svg",
):
    (model_dir / filename).unlink(missing_ok=True)

print("Số tổ hợp tham số:", len(search.cv_results_["params"]))
print("Siêu tham số tốt nhất:", search.best_params_)
print(f"CV F1-macro: {search.best_score_:.4%}")
