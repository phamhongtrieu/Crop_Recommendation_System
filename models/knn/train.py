from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.common import library_versions, read_split, timestamp

# 1. Đọc dữ liệu train; scaler sẽ được fit riêng bên trong từng fold.
model_dir = Path(__file__).resolve().parent
X_train, y_train = read_split("train")

# 2. Tạo pipeline và khai báo đúng lưới siêu tham số cần thử.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_jobs=1)),
])
param_grid = {
    "knn__n_neighbors": [3, 5, 11, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Chọn tham số theo F1-macro và fit lại pipeline tốt nhất trên toàn bộ train.
search = GridSearchCV(
    model, param_grid, cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    refit="f1_macro", return_train_score=True, n_jobs=1,
)
search.fit(X_train, y_train)

# 4. Lưu cả scaler, KNN tốt nhất và toàn bộ kết quả tìm kiếm.
joblib.dump(search.best_estimator_, model_dir / "model.joblib", compress=3)
joblib.dump({
    "search": search, "finished_at": timestamp(), "versions": library_versions(),
}, model_dir / "training_results.joblib", compress=3)
for filename in ("model_metadata.json", "evaluation_results.json"):
    (model_dir / filename).unlink(missing_ok=True)

print("Số tổ hợp tham số:", len(search.cv_results_["params"]))
print("Siêu tham số tốt nhất:", search.best_params_)
print(f"CV F1-macro: {search.best_score_:.4%}")
