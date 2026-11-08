# Backend – Crop Recommendation System

Project 1 cung cấp REST API bằng FastAPI và Pydantic v2 để nhận các chỉ số
nông nghiệp, khí hậu và trả tên cây trồng từ mô hình của nhóm ML.
Backend có thể demo ngay khi chưa có mô hình: `/health` báo trạng thái và
`/predict` trả 503. Không có dự đoán mặc định hoặc mô hình giả.

## Cấu trúc

```text
backend/
  __init__.py
  app.py                    # Endpoint, lifespan, CORS, xử lý lỗi
  schemas.py                # Kiểm tra request và schema response
  model_service.py          # Nạp artifact và gọi predict
  requirements_backend.txt
  README.md
tests/
  test_backend.py           # TestClient và mô hình mock trong bộ nhớ
models/
  crop_model.joblib         # Pipeline GaussianNB của nhóm ML
  model_metadata.json       # Hợp đồng đầu vào và phiên bản lúc export
```

## Cài đặt và chạy trên Windows

Mở PowerShell tại thư mục gốc `Crop_Recommendation`:

```powershell
# Chỉ tạo môi trường nếu chưa có .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements_backend.txt
python -m uvicorn backend.app:app --reload --port 8000
```

Nếu PowerShell chặn activation, có thể dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements_backend.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --port 8000
```

Swagger UI: <http://127.0.0.1:8000/docs>.
OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>.
Lệnh trên chạy từ thư mục gốc; đường dẫn artifact được tính từ vị trí mã nguồn,
không phụ thuộc thư mục làm việc. Khi chạy từ nơi khác cần đưa thư mục gốc vào
Python import path, ví dụ tùy chọn `--app-dir` của Uvicorn.

## API

| Endpoint | Kết quả |
| --- | --- |
| `GET /` | Tên API, phiên bản và đường dẫn tài liệu |
| `GET /health` | `{"status":"ok","model_loaded":false}` nếu mô hình chưa sẵn sàng |
| `POST /predict` | `{"recommended_crop":"rice"}` nếu mô hình thực sự trả về `rice` |

`/health` kiểm tra API còn hoạt động và artifact đã nạp; không chạy suy luận thử
và không đánh giá chất lượng mô hình. Khi thiếu hoặc không nạp được artifact,
endpoint này vẫn trả HTTP 200 với `model_loaded: false`.

Ví dụ request `/predict`:

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

Thử bằng PowerShell:

```powershell
$body = @{
    N = 90; P = 42; K = 43; temperature = 20.87
    humidity = 82.0; ph = 6.5; rainfall = 202.9
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body $body
```

Chưa có mô hình, request hợp lệ trả HTTP 503:

```json
{"detail":"Crop model is not ready. Please contact the backend team and retry after model deployment."}
```

Quy tắc dữ liệu:

- Bắt buộc đủ 7 trường, tên và chữ hoa/thường như ví dụ. Không nhận trường thừa.
- Chỉ nhận số JSON (số nguyên hoặc số thực), không nhận chuỗi số, boolean, null.
- N, P, K, rainfall không âm; humidity từ 0 đến 100; ph từ 0 đến 14, gồm hai đầu.
- Temperature có thể âm nhưng phải hữu hạn. Mọi trường đều từ chối NaN/Infinity.
- HTTP 422 khi request sai (kể cả khi chưa có mô hình); `detail` chứa vị trí,
  thông báo và loại lỗi, không phản chiếu dữ liệu đầu vào.
- HTTP 503 khi thiếu hoặc không nạp được mô hình.
- HTTP 500 khi suy luận lỗi hoặc kết quả không đúng hợp đồng. JSON chỉ có thông
  báo chung; chi tiết lỗi được ghi log phía server.

CORS chỉ cho phép `http://localhost:8501` và `http://127.0.0.1:8501`,
phương thức GET/POST và header Content-Type. CORS áp dụng cho trình duyệt;
Streamlit gọi từ server không bị ràng buộc bởi CORS.

## Kiểm thử

Chạy từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_backend.py -q
```

Các test trong `test_backend.py` thay `joblib.load` bằng mock, không tạo artifact,
không dùng artifact thật
và chạy được khi thư mục models trống. Test kiểm tra endpoint, validation, biên,
NaN/Infinity, lỗi nội bộ, CORS, Swagger/OpenAPI, số lần nạp, đường dẫn độc lập
thư mục hiện hành, thứ tự đặc trưng và pipeline có bước biến đổi riêng.

Trên Windows, test đổi working directory tạo thư mục tạm duy nhất trong `tests/`
và tự xóa sau khi khôi phục thư mục làm việc. Test không dùng thư mục chung
`%TEMP%\pytest-of-Admin`, vì thư mục này có thể do tài khoản chạy sandbox tạo
với ACL chỉ cho chủ sở hữu truy cập, khiến tài khoản chạy IDE gặp WinError 5.
Không cần chạy IDE bằng quyền Administrator hoặc thay đổi ACL thư mục hệ thống.

## Hợp đồng bàn giao mô hình với Bảo Kha

1. Artifact Joblib đáng tin cậy của nhóm nằm tại `models/crop_model.joblib`.
   Joblib có thể thực thi mã khi deserialize: chỉ nạp artifact đã xác nhận nguồn.
2. Artifact là estimator hoặc một pipeline hoàn chỉnh có `predict(X)`. Backend
   truyền DataFrame pandas một dòng, đúng thứ tự cột:
   `N, P, K, temperature, humidity, ph, rainfall`, các giá trị float chưa biến đổi.
   Nếu có `feature_names_in_` hoặc `n_features_in_`, chúng phải khớp hợp đồng này.
3. Pipeline tự thực hiện scaling, encoding hoặc các bước tiền xử lý đã huấn luyện.
   Backend không chuẩn hóa hoặc suy đoán preprocessing. Artifact dạng dictionary
   gồm nhiều đối tượng hoặc bộ tiền xử lý rời cần thống nhất adapter trước khi dùng.
4. `predict` phải trả mảng/danh sách một chiều chứa một tên cây trồng không rỗng
   cho mỗi dòng (ví dụ chuỗi `rice`). Nếu trả mã số, cần bàn giao mapping hoặc
   decoder và bổ sung adapter; hiện tại backend từ chối mã số bằng HTTP 500.
5. Cung cấp phiên bản Python, scikit-learn, pandas, NumPy, Joblib cùng dependency
   và custom transformer cần thiết để tái tạo môi trường khi export. File requirements
   hiện dùng các giới hạn sẵn có; chưa bảo đảm tương thích với artifact chưa bàn giao.
6. Xác nhận đơn vị, ý nghĩa và thứ tự 7 đặc trưng, tên nhãn, yêu cầu đầu vào
   DataFrame/array và vài mẫu đầu vào kèm nhãn dự kiến để đối chiếu tích hợp thực tế.
   Nếu chỉ nhận ndarray hoặc dùng transformer riêng, cần điều chỉnh tối thiểu sau
   khi nhận thông tin; không tự đổi định dạng trước khi thống nhất.

Sau khi đặt artifact, dừng và khởi động lại Uvicorn; chỉ nạp một lần cho mỗi vòng
đời ứng dụng/worker, không nạp lại theo request. Thiếu hoặc hỏng file lúc khởi động
cũng chỉ thử một lần; đặt file sau đó cần khởi động lại. Kiểm tra `/health`, rồi thử
`/predict` bằng các mẫu kiểm chứng của nhóm ML.

Hiện chưa thể xác nhận dự đoán thật, độ chính xác, tính tương thích artifact hoặc
preprocessing thực tế. Backend không huấn luyện, không tự tạo mô hình và không
cung cấp confidence hoặc batch prediction.
