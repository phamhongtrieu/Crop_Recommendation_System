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
