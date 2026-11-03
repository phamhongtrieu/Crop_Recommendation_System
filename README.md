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
Khi dự đoán, dùng scaler đã lưu để `transform` dữ liệu đầu vào.
Nếu chạy cross-validation, dùng dữ liệu chưa chuẩn hóa và đặt `StandardScaler`
trong `Pipeline` cùng mô hình để fit riêng trong từng fold.
