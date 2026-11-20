# Hệ thống khuyến nghị cây trồng (Crop Recommendation System)

Đồ án Học máy dự đoán loại cây trồng từ 7 đặc trưng nông nghiệp và khí hậu:
`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

Dataset gồm **2.200 mẫu, 22 loại cây**, mỗi loại 100 mẫu. Dữ liệu được chia có
phân tầng theo nhãn: **1.760 mẫu train và 440 mẫu test**, `random_state=42`.

Hiện đã huấn luyện **Softmax và Decision Tree**. Các thư mục KNN, Naive Bayes,
SVM và Random Forest đang để trống. Backend hiện sử dụng **Softmax** tại
`models/softmax/model.joblib`.

## Cấu trúc thư mục

```text
data/
  raw/Crop_recommendation.csv        # Dataset gốc
  processed/                        # X_train, X_test, y_train, y_test (CSV)
notebooks/
  01_exploratory_data_analysis.ipynb
  02_data_preprocessing.ipynb
models/
  common.py                         # Đọc dữ liệu, thời gian và phiên bản thư viện
  reporting.py                      # Tổng hợp kết quả đánh giá thành Excel
  model_evaluation_results.xlsx     # Báo cáo chung cho các mô hình đã đánh giá
  softmax/
    train.py
    evaluate.ipynb
    export_excel.py
    model.joblib
    training_results.joblib
    model_metadata.json
    evaluation_results.json
  decision_tree/
    train.py
    evaluate.ipynb
    export_excel.py
    model.joblib
    training_results.joblib
    model_metadata.json
    evaluation_results.json
    best_decision_tree.png
    best_decision_tree.svg
  knn/                              # Chỉ có .gitkeep
  naive_bayes/                      # Chỉ có .gitkeep
  svm/                              # Chỉ có .gitkeep
  random_forest/                    # Chỉ có .gitkeep
src/
  preprocessing.py                  # Làm sạch và chia dữ liệu
  download_data.py                   # Tải dataset
  test_model.py                      # Kiểm tra Softmax, checkpoint và Excel
backend/                            # REST API FastAPI
frontend/                           # Giao diện Streamlit
tests/                              # Kiểm tra Backend
```

## Cài đặt

Chạy các lệnh từ **thư mục gốc dự án**, nơi có README này. Nếu đã có môi trường
`.venv` đang hoạt động, kích hoạt môi trường đó:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên máy mới, dùng **Python 3.12.9** để khớp metadata của các mô hình. Chỉ tạo
môi trường khi đường dẫn `.venv` chưa tồn tại:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài dependencies khi thiết lập môi trường:

```powershell
python -m pip install -r requirements-artifact.txt
python -m pip check
```

Nếu PowerShell chặn activation, thay `python` bằng `.\.venv\Scripts\python.exe`
trong các lệnh. Linux/macOS dùng `source .venv/bin/activate`.

Các phiên bản được ghi trong metadata của cả hai mô hình:

| Thành phần | Phiên bản |
| --- | --- |
| Python | 3.12.9 |
| scikit-learn | 1.9.0 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| Joblib | 1.5.3 |

`requirements-artifact.txt` ghim bốn thư viện ML trên và bao gồm
`requirements.txt`. File requirements chung dùng dependencies Backend/Frontend,
đồng thời bổ sung notebook, biểu đồ, Excel và công cụ tải dataset. Đây chưa phải
lockfile đầy đủ; quy trình cài mới toàn bộ môi trường chưa được xác minh.

## Chạy ứng dụng

Backend nạp `models/softmax/model.joblib` một lần khi khởi động. Nếu dùng model
được bàn giao, có thể chạy ứng dụng ngay mà không cần train lại.

Terminal 1, tại thư mục gốc và trong môi trường đã kích hoạt:

```powershell
python -m uvicorn backend.app:app --reload --port 8000
```

Terminal 2, tại cùng thư mục gốc:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

- Giao diện: <http://localhost:8501>.
- Tài liệu API: <http://127.0.0.1:8000/docs>.
- Trạng thái model: <http://127.0.0.1:8000/health>.

Khi model nạp được, `/health` trả `{"status":"ok","model_loaded":true}`.
Nếu model thiếu hoặc không nạp được, `model_loaded` là `false` và `/predict`
trả HTTP 503 với đầu vào hợp lệ. Khởi động lại Backend sau khi thay model.

URL Backend mặc định trên Frontend là `http://127.0.0.1:8000`. Nút Random chỉ
điền dữ liệu mẫu; nhấn **Dự đoán** để gửi yêu cầu tới API.

## Đầu vào và kết quả API

`POST /predict` nhận đúng bảy trường số JSON sau. Hai đầu khoảng đều được chấp
nhận; trường thừa, chuỗi số, boolean, null, NaN và Infinity bị từ chối với HTTP 422.

| Trường | Phạm vi demo |
| --- | --- |
| N | 0–140 |
| P | 0–145 |
| K | 0–205 |
| temperature | 0–50 °C |
| humidity | 0–100% |
| ph | 0–14 |
| rainfall | 0–300 mm |

Đây là phạm vi hỗ trợ demo. Đơn vị N/P/K và chu kỳ tổng hợp rainfall chưa được
xác minh từ nguồn dataset; Frontend gửi nguyên giá trị đầu vào.

Ví dụ gọi API bằng PowerShell, sau khi Backend đã chạy:

```powershell
$body = @{
    N = 90; P = 42; K = 43; temperature = 20.87974371
    humidity = 82.00274423; ph = 6.502985292; rainfall = 202.9355362
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
```

Mẫu này được Softmax hiện tại dự đoán là `rice`. Kết quả gồm
`recommended_crop` và `top_3`, mỗi phần tử có `crop` và `probability`.
Backend truyền dữ liệu gốc vào pipeline Softmax; pipeline tự chuẩn hóa bằng
StandardScaler. Xác suất Top 3 lấy trực tiếp từ `predict_proba`, không chuẩn hóa
riêng tổng Top 3 và không phải tỷ lệ đảm bảo cây trồng sẽ phát triển tốt.

## Chuẩn bị dữ liệu

Dữ liệu gốc và bốn CSV processed đã có trong dự án. Nếu cần tải lại dataset:

```powershell
python -m src.download_data
```

Chạy các cell trong [02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb)
để làm sạch và tạo `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` trong
`data/processed/`. Notebook dùng `clean_data` và `split_data` từ
`src/preprocessing.py`, chia có phân tầng 80/20 và giữ nguyên thang đo đặc trưng.
Các bước tải lại và lưu CSV có thể ghi đè dữ liệu hiện có.

## Huấn luyện và đánh giá

Cả hai thuật toán dùng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
và GridSearchCV với hai chỉ số Accuracy, F1-macro. Siêu tham số được chọn theo
**F1-macro validation trung bình** (`refit="f1_macro"`), sau đó mô hình tốt nhất
được fit lại trên toàn bộ train. Tập test chỉ dùng để đánh giá cuối cùng.

### Softmax

Chạy từ thư mục gốc:

```powershell
python -m models.softmax.train
```

Script dùng `Pipeline(StandardScaler, LogisticRegression)` với `solver="lbfgs"`,
`max_iter=5000`, `random_state=42`. StandardScaler được fit riêng trên train của
từng fold. Lưới tham số: `model__C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`.

Sau khi train, mở [evaluate.ipynb của Softmax](models/softmax/evaluate.ipynb),
chọn kernel Python của môi trường dự án và chạy **Run All**. Notebook hiển thị
siêu tham số tốt nhất, bảng tổng hợp, toàn bộ kết quả GridSearch và lưu JSON cùng
metadata.

### Decision Tree

```powershell
python -m models.decision_tree.train
```

DecisionTreeClassifier dùng `random_state=42` và nhận đặc trưng gốc.
Lưới tham số hiện tại gồm **16 tổ hợp**:

```python
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [8, 12],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 3],
    "ccp_alpha": [0.0],
}
```

Sau khi train, chạy **Run All** trong
[evaluate.ipynb của Decision Tree](models/decision_tree/evaluate.ipynb).
Phần 6 vẽ toàn bộ cây có siêu tham số tốt nhất và lưu
[PNG](models/decision_tree/best_decision_tree.png) cùng
[SVG](models/decision_tree/best_decision_tree.svg). Mở SVG để phóng to từng nút.

### Các file kết quả và báo cáo Excel

Mỗi thuật toán lưu kết quả trong thư mục riêng:

| File | Nội dung |
| --- | --- |
| `train.py` | Tìm siêu tham số, fit và lưu mô hình tốt nhất |
| `evaluate.ipynb` | Đánh giá mô hình, xem bảng kết quả và lưu JSON |
| `export_excel.py` | Tổng hợp các kết quả đã đánh giá vào Excel chung |
| `model.joblib` | Mô hình tốt nhất đã fit; Softmax lưu cả scaler trong pipeline |
| `training_results.joblib` | Dictionary gồm `search` (GridSearchCV đã fit), `finished_at`, `versions` |
| `model_metadata.json` | Đặc trưng, nhãn, tham số, phiên bản thư viện và điểm CV/test |
| `evaluation_results.json` | Hai bảng `Tong_hop` và `GridSearch` |

Sau khi chạy notebook đánh giá, xuất báo cáo chung:

```powershell
python -m models.reporting
```

Có thể dùng `python -m models.softmax.export_excel` hoặc
`python -m models.decision_tree.export_excel` để xuất cùng báo cáo.
Kết quả nằm tại [models/model_evaluation_results.xlsx](models/model_evaluation_results.xlsx):

| Sheet | Nội dung |
| --- | --- |
| `Tong_hop` | Một dòng cho mỗi mô hình tốt nhất đã đánh giá, gồm 13 cột |
| `Tieu_chi` | Giải thích các cột tổng hợp |
| `GridSearch` | Các cấu hình đã thử, điểm CV, độ lệch chuẩn, thứ hạng và cột `Selected` |

Chỉ các thư mục có `evaluation_results.json` mới được tổng hợp. Chạy lại train
sẽ ghi đè model/checkpoint và xóa metadata cùng kết quả JSON cũ của thuật toán;
Decision Tree còn xóa hình cây cũ. Sau đó cần chạy lại notebook và xuất Excel.
Đóng file Excel trước khi xuất báo cáo.

### Kết quả đã lưu

Các chỉ số dưới đây lấy từ `model_metadata.json` của từng thuật toán:

| Mô hình | CV F1-macro | Test Accuracy | Test F1-macro |
| --- | ---: | ---: | ---: |
| Softmax | 97,76% | 98,41% | 98,40% |
| Decision Tree | 98,47% | 97,95% | 97,94% |

Softmax được chọn với `C=100`. Decision Tree được chọn với `criterion="gini"`,
`max_depth=12`, `min_samples_split=2`, `min_samples_leaf=1`, `ccp_alpha=0.0`;
cây đã fit có độ sâu 12, 32 lá và 63 nút.
Điểm CV là điểm dùng trong tìm siêu tham số; điểm test là đánh giá holdout.

## Kiểm thử

Chạy toàn bộ kiểm thử Backend, Frontend và model:

```powershell
python -m pytest tests frontend/test_app.py src/test_model.py -q
```

Chỉ kiểm tra Softmax, checkpoint và Excel:

```powershell
python -m pytest src/test_model.py -q
```

Các kiểm thử Softmax cần có model, checkpoint, metadata, CSV processed và báo cáo
Excel tương ứng. `src/test_model.py` đối chiếu model đã lưu với checkpoint,
kiểm tra điểm test, xác suất và các bảng Excel. Kiểm thử không ghi đè các artifact đã lưu.
