# Hệ thống khuyến nghị cây trồng (Crop Recommendation System)

Đồ án Học máy dự đoán loại cây trồng từ 7 đặc trưng nông nghiệp và khí hậu:
`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

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
```

Hiện chỉ triển khai và huấn luyện **Softmax**. Các thuật toán khác giữ trống
cho đến khi có quyết định tiếp theo. `notebooks/` chỉ còn EDA và tiền xử lý;
mã train, đánh giá và kết quả nằm trong `models/`.

## Cài đặt

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

```powershell
python models/softmax/export_excel.py
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
