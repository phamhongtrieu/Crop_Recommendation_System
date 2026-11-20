from pathlib import Path
import sys

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.common import library_versions, read_split, timestamp

# 1. Đọc dữ liệu train.
model_dir = Path(__file__).resolve().parent
X_train, y_train = read_split("train")

# 2. Tạo pipeline và khai báo các giá trị C cần thử.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)),
])
param_grid = {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. GridSearch chọn C theo F1-macro và tự fit lại model tốt nhất.
search = GridSearchCV(
    model, param_grid, cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    refit="f1_macro", return_train_score=True, n_jobs=1,
)

search.fit(X_train, y_train)

# 4. Lưu model tốt nhất và kết quả tìm kiếm 
joblib.dump(search.best_estimator_, model_dir / "model.joblib", compress=3)
joblib.dump({
    "search": search, "finished_at": timestamp(), "versions": library_versions(),
}, model_dir / "training_results.joblib", compress=3)
for filename in ("model_metadata.json", "evaluation_results.json"):
    (model_dir / filename).unlink(missing_ok=True)

print("Siêu tham số tốt nhất:", search.best_params_)
print(f"CV F1-macro: {search.best_score_:.4%}")
