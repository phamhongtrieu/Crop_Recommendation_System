from pathlib import Path

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from models.common import library_versions, read_split, timestamp

# 1. Đọc dữ liệu train; Decision Tree dùng trực tiếp đặc trưng gốc.
model_dir = Path(__file__).resolve().parent
X_train, y_train = read_split("train")

# 2. Khai báo đúng lưới siêu tham số cần thử.
model = DecisionTreeClassifier(random_state=42)
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [8, 12],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 3],
    "ccp_alpha": [0.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Chọn tham số theo F1-macro và fit lại cây tốt nhất trên toàn bộ train.
search = GridSearchCV(
    model, param_grid, cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    refit="f1_macro", return_train_score=True, n_jobs=1,
)
search.fit(X_train, y_train)

# 4. Lưu cây tốt nhất và toàn bộ kết quả tìm kiếm như các thuật toán khác.
joblib.dump(search.best_estimator_, model_dir / "model.joblib", compress=3)
joblib.dump({
    "search": search, "finished_at": timestamp(), "versions": library_versions(),
}, model_dir / "training_results.joblib", compress=3)
for filename in (
    "model_metadata.json", "evaluation_results.json",
    "best_decision_tree.png", "best_decision_tree.svg",
):
    (model_dir / filename).unlink(missing_ok=True)

print("Số tổ hợp tham số:", len(search.cv_results_["params"]))
print("Siêu tham số tốt nhất:", search.best_params_)
print(f"CV F1-macro: {search.best_score_:.4%}")
