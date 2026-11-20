# Hệ thống khuyến nghị cây trồng (Crop Recommendation System)

Đồ án Học máy dự đoán loại cây trồng từ 7 đặc trưng nông nghiệp và khí hậu:
`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

<<<<<<< HEAD
## Cấu trúc thư mục

```text
data/
  raw/                              # Dataset gốc
  processed/                        # X_train, X_test, y_train, y_test (CSV)
notebooks/
  01_exploratory_data_analysis.ipynb
  02_data_preprocessing.ipynb
models/
  __init__.py
  common.py                         # Đọc dữ liệu và thông tin lần chạy
  reporting.py                      # Tổng hợp kết quả đánh giá thành Excel
  model_evaluation_results.xlsx      # Báo cáo chung
  softmax/
    train.py                        # GridSearchCV và lưu model tốt nhất
    evaluate.ipynb                  # Chạy đánh giá và xem kết quả từng bước
    export_excel.py                 # Xuất kết quả đã đánh giá sang Excel
    model.joblib                    # Pipeline Softmax được chọn theo CV
    model_metadata.json             # Đặc trưng, nhãn, tham số, CV/test, phiên bản
    training_results.joblib         # Đối tượng GridSearch đã fit và thông tin lần train
    evaluation_results.json         # Hai bảng Tong_hop và GridSearch
  knn/                              # Trống, chỉ có .gitkeep
  naive_bayes/                      # Trống, chỉ có .gitkeep
  svm/                              # Trống, chỉ có .gitkeep
  decision_tree/                    # Trống, chỉ có .gitkeep
  random_forest/                    # Trống, chỉ có .gitkeep
src/
  preprocessing.py                  # Làm sạch và chia dữ liệu
  download_data.py                   # Tải dataset
  test_model.py                      # Kiểm tra model, checkpoint và Excel
backend/                            # REST API FastAPI
frontend/                           # Giao diện Streamlit
tests/                              # Kiểm tra hợp đồng API
=======
Dataset hiện có **2.200 mẫu, 7 đặc trưng và 22 loại cây**, mỗi loại 100 mẫu.
Train/test có phân tầng theo nhãn, tỷ lệ 80/20: 1.760 mẫu train và 440 mẫu test.

## Cấu trúc Đồ án
1. **Project 1: Backend RestAPI (`/backend`)**
   - Xây dựng RestAPI sử dụng FastAPI.
   - Nhận Input mẫu dạng JSON các chỉ số nông học, trả về kết quả dự đoán loại cây trồng phù hợp nhất.
2. **Project 2: Frontend UI (`/frontend`)**
   - Xây dựng giao diện ứng dụng tương tác bằng Streamlit.
   - Cho phép người dùng nhập dữ liệu hoặc chọn kịch bản để nhận khuyến nghị trực quan.
3. **Project 3: MLOps & Experimentation (`/notebooks`, `/src`, `/models`)**
   - Phân tích dữ liệu thăm dò (EDA).
   - Đã huấn luyện và so sánh 6 thuật toán: SoftMax (LogisticRegression), KNN, GaussianNB, SVM, Decision Tree và Random Forest; mỗi thuật toán có cấu hình mặc định và GridSearch.
   - Đóng gói và lưu trữ mô hình chuẩn vào thư mục `models/`.

## Hướng dẫn cài đặt và chạy trên Windows

Các lệnh cài đặt dưới đây dành cho **máy/môi trường mới**. Nếu đã có `.venv`
đang hoạt động, giữ nguyên môi trường đó; không chạy lại lệnh tạo hoặc cài đặt
để ghi đè nó. Có thể dùng một tên môi trường chưa tồn tại, ví dụ `.venv-artifact`,
và thay đường dẫn tương ứng trong các lệnh. Hai terminal đều phải đứng tại
**thư mục gốc project**, nơi có README này, `backend/`, `frontend/` và `models/`.

### 1. Clone repository

```powershell
git clone https://github.com/phamhongtrieu/Crop_Recommendation_System.git
cd Crop_Recommendation_System
```

Nếu đã có bản project, mở PowerShell tại thư mục gốc hiện có và bỏ qua bước clone.

### 2. Tạo và kích hoạt môi trường mới

Để khớp phiên bản Python trong metadata, cài **Python 3.12.9** trước.
`py -3.12` chọn bản Python 3.12 đã cài; kiểm tra phiên bản chính xác bằng lệnh sau:

```powershell
py -3.12 --version
# Chỉ chạy khi đường dẫn .venv chưa tồn tại.
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Nếu PowerShell chặn activation, dùng trực tiếp `.\.venv\Scripts\python.exe`
thay cho `python` trong các lệnh tiếp theo. Không cần thay đổi policy hệ thống.

### 3. Cài dependencies

Trong môi trường mới, cài cấu hình dành cho artifact được bàn giao:

```powershell
python -m pip install -r requirements-artifact.txt
python -m pip check
```

`requirements-artifact.txt` bao gồm `requirements.txt` và ghim bốn thư viện ML
đúng metadata. `requirements.txt` dùng chung requirements Backend/Frontend,
bổ sung notebook, Excel (`openpyxl`) và công cụ tải dataset; `pytest` và `httpx`
được khai báo trong requirements Backend. Đây chưa phải lockfile toàn bộ
dependency gián tiếp. **Cài đặt sạch và chạy kết hợp trên Python 3.12.9:
CHƯA XÁC MINH.** Xem phần môi trường bên dưới trước khi bàn giao.

### 4. Kiểm tra artifact

```powershell
Test-Path models/crop_model.joblib
Test-Path models/model_metadata.json
```

Cả hai phải trả `True`. Trong checkout được kiểm tra, model, metadata, dataset,
bốn CSV processed và checkpoint `notebooks/model_training_results.joblib`
đều được Git theo dõi. `.gitignore` hiện tại không bỏ qua các artifact này.
Chưa xác minh lại bằng một bản clone từ remote trên máy sạch.

Nếu bản bàn giao thiếu artifact, ưu tiên nhận **cặp model/metadata cùng lần
export** từ nhóm và đặt đúng đường dẫn. Chỉ nạp artifact đáng tin cậy.
Không cần huấn luyện hoặc export lại để demo khi artifact đã đủ.
Nếu cần tái tạo, xem quy trình ở phần tiền xử lý/xuất model; các bước đó có ghi
file và chỉ thực hiện khi chủ động muốn tạo lại artifact.

### 5. Chạy Backend — terminal 1

Từ thư mục gốc, với môi trường đã kích hoạt:

```powershell
python -m uvicorn backend.app:app --reload --port 8000
```

### 6. Kiểm tra Backend

Mở <http://127.0.0.1:8000/health> và <http://127.0.0.1:8000/docs>.
Khi model nạp được, `/health` trả:

```json
{"status":"ok","model_loaded":true}
```

Nếu model thiếu/hỏng, `/health` vẫn có thể trả HTTP 200 với `model_loaded:false`;
request hợp lệ tới `/predict` trả 503. Đặt lại artifact rồi khởi động lại Backend.

### 7. Chạy Frontend — terminal 2

Mở terminal khác tại **cùng thư mục gốc**, kích hoạt cùng môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

### 8. Mở giao diện và thử dự đoán

Mở <http://localhost:8501>. URL Backend mặc định là `http://127.0.0.1:8000`.
Nhấn Random chỉ điền dữ liệu; nhấn Dự đoán mới gửi `POST /predict`.
Mẫu tham chiếu ở phần model bên dưới cho kết quả `rice`, Top 3 lần lượt là
`rice`, `jute`, `coffee` với artifact hiện tại.

### 9. Chạy kiểm thử

Từ thư mục gốc, trong terminal đã kích hoạt môi trường:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -B -m pytest -p no:cacheprovider tests frontend/test_app.py src/test_model.py -q
```

Test không huấn luyện hoặc ghi lại model/dataset. Một test đường dẫn tạo thư mục
tạm mới trong `tests/` và tự dọn thư mục đó. Các test Frontend kiểm tra random
không tự dự đoán, validation, kết quả cũ và lỗi kết nối. Test model cần đủ CSV
processed và checkpoint. Kiểm thử riêng: thêm đường dẫn nhóm muốn chạy thay cho
ba đường dẫn trên. Model thiếu có thể khiến nhóm tích hợp bị skip; cần xem cả
số pass/fail/skip, không chỉ mã thoát.

## Môi trường và khả năng tái lập

| Thành phần | Lúc xuất model (metadata) | Môi trường đã kiểm tra |
| --- | --- | --- |
| Python | 3.12.9 | 3.14.5 |
| scikit-learn | 1.9.0 | 1.9.1 |
| NumPy | 2.5.2 | 2.5.4 |
| pandas | 3.0.5 | 3.1.1 |
| Joblib | 1.5.3 | 1.6.0 |

`requirements-artifact.txt` ghi lại bốn phiên bản thư viện ở cột metadata;
Python 3.12.9 phải được cài riêng. Metadata không ghi đủ mọi dependency gián tiếp
(ví dụ SciPy), nên chưa thể khẳng định tái tạo chính xác toàn bộ môi trường train.
Các dải phiên bản trong `requirements.txt` dành cho phát triển; nếu cài riêng file
đó, pip có thể chọn phiên bản khác artifact.

Môi trường kiểm tra dùng FastAPI 0.162.0, Pydantic 2.14.3, Uvicorn 0.54.0,
Streamlit 1.67.0, pytest 9.2.0 và httpx 0.28.1. Nạp model/dự đoán được nhưng còn
`InconsistentVersionWarning` do khác scikit-learn; đây không phải bằng chứng
tương thích đầy đủ. Không cài lại thư viện chỉ để tắt warning. scikit-learn
[không bảo đảm nạp model giữa các phiên bản khác nhau](https://scikit-learn.org/stable/model_persistence.html).

Frontend ghim Streamlit 1.67.0 theo bản đã chạy AppTest/widget/CSS. Mức cũ 1.25
không đủ cho `st.container(key=..., gap=..., border=...)`.
[Tài liệu 1.48](https://docs.streamlit.io/1.48.0/develop/api-reference/layout/st.container)
đã có các tham số container này, nhưng toàn bộ test/CSS chưa được kiểm chứng trên
mọi bản cũ hơn 1.67.0. Vì vậy đây là bản ghim đã kiểm thử, không phải khẳng định
1.67.0 là bản đầu tiên hỗ trợ các API đó.

Trước khi xác nhận bàn giao môi trường sạch: cài vào môi trường mới, chạy
`python -m pip check`, bộ test, `/health`, mẫu `rice` và kiểm tra giao diện trong
trình duyệt. Các gói notebook chưa được cài/kiểm thử trong `.venv` hiện tại;
đặc biệt `openpyxl` còn thiếu. Không có cài đặt/nâng cấp môi trường trong lần
cập nhật tài liệu này.

## Hợp đồng đầu vào API

`POST /predict` nhận đúng bảy trường số JSON, không nhận trường thừa, chuỗi số,
boolean, null, NaN hoặc Infinity. Hai đầu khoảng đều được chấp nhận:

| Trường | Phạm vi demo |
| --- | --- |
| N | 0–140 |
| P | 0–145 |
| K | 0–205 |
| temperature | 0–50 °C |
| humidity | 0–100% |
| ph | 0–14 |
| rainfall | 0–300 mm |

Đây là phạm vi hỗ trợ demo, không phải giới hạn vật lý tuyệt đối. Chưa xác minh
đơn vị N/P/K và chu kỳ tổng hợp rainfall từ nguồn dataset; không mặc định chúng
là kg/ha hoặc lượng mưa theo tháng/năm. Frontend gửi nguyên giá trị, không đổi
đơn vị. Dữ liệu không hợp lệ trả HTTP 422.

Ví dụ gọi API từ PowerShell:

```powershell
$body = @{
    N = 90; P = 42; K = 43; temperature = 20.87974371
    humidity = 82.00274423; ph = 6.502985292; rainfall = 202.9355362
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
>>>>>>> origin/main
```

Hiện chỉ triển khai và huấn luyện **Softmax**. Các thuật toán khác giữ trống
cho đến khi có quyết định tiếp theo. `notebooks/` chỉ còn EDA và tiền xử lý;
mã train, đánh giá và kết quả nằm trong `models/`.

<<<<<<< HEAD
## Cài đặt
=======
Quy trình tùy chọn để tái tạo dữ liệu; các lệnh tải dữ liệu và cell lưu CSV có
thể ghi đè file hiện có. Bỏ qua nếu dùng dữ liệu/artifact được bàn giao để demo.

Chạy từ thư mục gốc nếu cần tải dataset vào `data/raw/`:
>>>>>>> origin/main

Chạy từ thư mục gốc của dự án:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS dùng `source .venv/bin/activate` để kích hoạt môi trường.

## Chuẩn bị dữ liệu

Nếu chưa có dataset:

```powershell
python -m src.download_data
```

Chạy các cell trong `notebooks/02_data_preprocessing.ipynb` để tạo bốn file
`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` trong `data/processed/`.
Notebook dùng `clean_data` và `split_data` từ `src/preprocessing.py`, chia có
phân tầng 80% train / 20% test, `random_state=42`, giữ nguyên thang đo đặc trưng.

## Huấn luyện và đánh giá Softmax

```powershell
python -m models.softmax.train
```

Mở `models/softmax/evaluate.ipynb`, chọn kernel Python và chạy các cell từ trên
xuống (Run All). Notebook hiển thị siêu tham số tốt nhất, bảng tổng hợp,
GridSearch, rồi lưu kết quả JSON và metadata ở cell cuối.
Sau đó xuất Excel và kiểm tra:

<<<<<<< HEAD
```powershell
python models/softmax/export_excel.py
=======
Chỉ xuất khi cần tái tạo model và đã thống nhất môi trường. Notebook 03 ghi
checkpoint, notebook 04 ghi Excel, script export ghi đè model và metadata.
Script export yêu cầu scikit-learn khớp checkpoint (hiện là 1.9.0); không chạy
export bằng `.venv` kiểm tra đang dùng 1.9.1. Kiểm tra model bằng `src.test_model`
chỉ đọc artifact, không huấn luyện lại.

Chạy notebook `03_model_training.ipynb` và `04_model_evaluation.ipynb` trước,
sau đó chạy từ thư mục gốc:

```bash
python -m src.export_model
>>>>>>> origin/main
python -m src.test_model
```

Có thể mở `train.py` hoặc `export_excel.py` và nhấn Run trong VS Code.
Cách chạy trực tiếp để train: `python models/softmax/train.py`.

`train.py` là script chạy tuần tự: đọc dữ liệu → tạo pipeline → GridSearch →
lưu model và kết quả CV. Phần lập bảng thống kê và metadata nằm trong `evaluate.ipynb`.
`train.py` và `export_excel.py` không được import nên chạy tuần tự ở cấp file,
không dùng `if __name__ == "__main__"`.
Script dùng `Pipeline(StandardScaler, LogisticRegression)` với `solver="lbfgs"`,
`max_iter=5000`. Với bài toán nhiều lớp, LogisticRegression này dùng Softmax.
StandardScaler chỉ fit trên train của từng fold, rồi fit trên toàn bộ train
khi tạo pipeline cuối cùng.

GridSearch dùng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` với
`C = [0.001, 0.01, 0.1, 1, 10, 100]`. Chọn C bằng F1-macro validation trung bình;
`model.joblib` lưu pipeline GridSearch đã fit. Tập test không dùng để fit hay
tìm tham số. Checkpoint lưu đối tượng GridSearch, gồm model tốt nhất và kết quả CV.

Cột `Best_grid_parameters` chỉ ghi siêu tham số tốt nhất từ `search.best_params_`.
`Tong_hop` chỉ giữ kết quả của model được chọn. Bảng `GridSearch` trong notebook,
JSON và Excel lưu tất cả cấu hình đã thử, điểm CV, độ lệch chuẩn và thứ hạng.
Cột `Selected=True` đánh dấu cấu hình được chọn; toàn bộ đối tượng tìm kiếm được lưu trong checkpoint.

`evaluate.ipynb` gồm các cell tuần tự: đọc model và dữ liệu → xem thông số →
tính chỉ số đánh giá → xem GridSearch → lưu JSON.
Notebook chỉ tính các chỉ số CV/train/test cho `Tong_hop` và kết quả tìm tham số
cho `GridSearch`. `Tieu_chi` được tạo từ mô tả các cột trong `reporting.py`. Kết quả lưu tại
`models/softmax/evaluation_results.json`, metadata được tạo cùng điểm CV/test,
và `export_excel.py` đọc các kết quả này để cập nhật **`models/model_evaluation_results.xlsx`**.

Excel chỉ gồm 3 sheet:

| Sheet | Nội dung |
| --- | --- |
| `Tong_hop` | Một dòng kết quả Softmax tốt nhất, 13 cột thông tin và chỉ số CV/train/test |
| `Tieu_chi` | Giải thích từng cột tổng hợp |
| `GridSearch` | Tất cả siêu tham số đã thử, điểm CV, thứ hạng và cột đánh dấu cấu hình được chọn |

Chỉ những thư mục có kết quả đánh giá mới được tổng hợp. Có thể tạo lại
Excel từ các kết quả đã lưu mà không train lại:

```powershell
python -m models.reporting
```

Khi train lại, metadata và kết quả JSON cũ của Softmax bị xóa; chạy lại notebook
`evaluate.ipynb` rồi `export_excel.py` để cập nhật Excel. Đóng file Excel trước khi xuất báo cáo.
Điểm CV của tham số đã chọn là
điểm dùng trong tối ưu, không phải nested CV; test là đánh giá holdout cuối.

<<<<<<< HEAD
## Chạy ứng dụng

Backend tải `models/softmax/model.joblib` một lần khi khởi động:

```powershell
python -m uvicorn backend.app:app --reload --port 8000
```

Mở terminal khác để chạy frontend:

```powershell
python -m streamlit run frontend/app.py
```

Đầu vào là 7 giá trị gốc theo thứ tự đặc trưng nêu trên. Pipeline tự chuẩn hóa
và trả tên cây dạng chuỗi; `predict_proba` dùng thứ tự nhãn trong `model.classes_`.
Khi triển khai, mang theo `models/softmax/model.joblib` và metadata, dùng cùng
phiên bản scikit-learn được ghi trong metadata. Khởi động lại backend sau khi đổi model.

Kiểm tra API:

```powershell
python -m pip install -r backend/requirements_backend.txt
python -m pytest tests/test_backend.py -q
```
=======
`src/test_model.py` kiểm tra model đã lưu so với checkpoint, thứ tự đặc trưng,
nhãn cây, chất lượng trên toàn bộ tập test (F1-macro tối thiểu 95%), dự đoán từ
mẫu JSON, xác suất và tính nhất quán giữa dự đoán đơn lẻ/hàng loạt.
Model, metadata và checkpoint hiện được Git theo dõi trong checkout được kiểm
tra. Khi đóng gói, cần giữ chúng cùng dữ liệu processed để chạy đủ kiểm thử.
Backend không nạp scaler riêng và không chuẩn hóa lại đầu vào. Xác suất Top 3
lấy trực tiếp từ model, không chuẩn hóa riêng tổng Top 3; đây không phải tỷ lệ
đảm bảo cây trồng phát triển tốt. Calibration chưa được xác minh đầy đủ.
>>>>>>> origin/main
