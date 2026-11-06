# Hệ thống khuyến nghị cây trồng (Crop Recommendation System)

Đồ án môn Học máy (Machine Learning) dựa trên các chỉ số nông nghiệp và khí hậu (N, P, K, Temperature, Humidity, pH, Rainfall).

## Cấu trúc Đồ án
1. **Project 1: Backend RestAPI (`/backend`)**
   - Xây dựng RestAPI sử dụng FastAPI.
   - Nhận Input mẫu dạng JSON các chỉ số nông học, trả về kết quả dự đoán loại cây trồng phù hợp nhất.
2. **Project 2: Frontend UI (`/frontend`)**
   - Xây dựng giao diện ứng dụng tương tác bằng Streamlit.
   - Cho phép người dùng nhập dữ liệu hoặc chọn kịch bản để nhận khuyến nghị trực quan.
3. **Project 3: MLOps & Experimentation (`/notebooks`, `/src`, `/models`)**
   - Phân tích dữ liệu thăm dò (EDA).
   - Huấn luyện và so sánh các thuật toán (Random Forest, Decision Tree, Naive Bayes, SVM, XGBoost,...).
   - Đóng gói và lưu trữ mô hình chuẩn vào thư mục `models/`.

## Hướng dẫn cài đặt và chạy
```bash
# 1. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 3. Chạy Backend FastAPI (Port 8000)
cd backend
uvicorn app:app --reload --port 8000

# 4. Chạy Frontend Streamlit (Port 8501)
cd frontend
streamlit run app.py
```

## Tiền xử lý dữ liệu

Chạy từ thư mục gốc nếu cần tải dataset vào `data/raw/`:

```bash
python -m src.download_data
```

Mở `notebooks/02_data_preprocessing.ipynb` và chạy các cell theo thứ tự. Notebook
gọi các hàm dùng chung trong `src/preprocessing.py`:

- `clean_data(df)`: làm sạch và kiểm tra các giá trị hợp lệ.
- `split_data(X, y, test_size=0.2, random_state=42)`: chia train/test có phân tầng theo nhãn.
- `scale_features(X_train, X_test)`: trả về train/test đã chuẩn hóa và scaler chỉ fit trên train.

Notebook lưu dữ liệu vào `data/processed/` và scaler vào `models/scaler.pkl`.
Scaler này dùng cho những mô hình cần chuẩn hóa khi làm việc ngoài pipeline.
Nếu chạy cross-validation, dùng dữ liệu chưa chuẩn hóa và đặt `StandardScaler`
trong `Pipeline` cùng mô hình để fit riêng trong từng fold.

## Xuất và kiểm tra model cho backend

Chạy notebook `03_model_training.ipynb` và `04_model_evaluation.ipynb` trước,
sau đó chạy từ thư mục gốc:

```bash
python -m src.export_model
python -m src.test_model
```

`src/export_model.py` chọn cấu hình GridSearch có `CV_f1_macro_mean` cao nhất
trong checkpoint huấn luyện. Khi đồng hạng, dùng thứ tự tên model giống notebook 04.
Script kiểm tra dữ liệu và phiên bản scikit-learn, rồi lưu pipeline đã fit trên train
vào `models/crop_model.joblib`. Tập test chỉ dùng để đánh giá cuối.

Với kết quả hiện tại, model được chọn là **Gaussian Naive Bayes**,
`var_smoothing=1e-12`: CV F1-macro **99,4871%**, test Accuracy **99,5455%**,
test F1-macro **99,5443%** (438/440 mẫu đúng).
`models/model_metadata.json` lưu tên model, tham số, thứ tự đặc trưng, nhãn cây,
điểm CV/test và phiên bản thư viện.

Backend có thể tải model một lần khi khởi động và dự đoán như sau:

```python
import joblib
import pandas as pd

model = joblib.load("models/crop_model.joblib")  # Đường dẫn tính từ thư mục gốc.
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
payload = {
    "N": 90, "P": 42, "K": 43, "temperature": 20.87974371,
    "humidity": 82.00274423, "ph": 6.502985292, "rainfall": 202.9355362,
}
sample = pd.DataFrame([payload], columns=features)
crop = model.predict(sample)[0]  # "rice"
probabilities = model.predict_proba(sample)[0]  # Theo thứ tự model.classes_.
```

Đầu vào luôn là **7 giá trị gốc**, theo đúng thứ tự trên. Pipeline giữ bước tiền xử lý
của model; Naive Bayes hiện tại không cần chuẩn hóa bằng `scaler.pkl`.
Model trả trực tiếp tên cây dạng chuỗi.

`src/test_model.py` kiểm tra model đã lưu so với checkpoint, thứ tự đặc trưng,
nhãn cây, chất lượng trên toàn bộ tập test (F1-macro tối thiểu 95%), dự đoán từ
mẫu JSON, xác suất và tính nhất quán giữa dự đoán đơn lẻ/hàng loạt.
Các file trong `models/` đang được bỏ qua bởi `.gitignore`; cần xuất lại hoặc
chép model và metadata khi triển khai.
