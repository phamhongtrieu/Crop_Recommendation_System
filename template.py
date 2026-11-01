"""
=============================================================================
Đồ án Học Máy (Machine Learning): Hệ thống khuyến nghị cây trồng
File: template.py
Mục đích: Tự động khởi tạo cấu trúc thư mục và các file dự án chuẩn MLOps
=============================================================================
"""

import os
import sys
from pathlib import Path
import json

# Thiết lập UTF-8 cho console output trên Windows để in tiếng Việt không bị lỗi charmap
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Định nghĩa thư mục gốc dự án (mặc định là thư mục hiện tại '.')
BASE_DIR = Path(".")

# Danh sách cấu trúc file cần khởi tạo trong dự án
FILES_TO_CREATE = [
    # Data directories
    BASE_DIR / "data" / "raw" / ".gitkeep",
    BASE_DIR / "data" / "processed" / ".gitkeep",
    
    # Models directory
    BASE_DIR / "models" / ".gitkeep",
    
    # Notebooks (MLOps experiment & training)
    BASE_DIR / "notebooks" / "01_exploratory_data_analysis.ipynb",
    BASE_DIR / "notebooks" / "02_model_training_comparison.ipynb",
    BASE_DIR / "notebooks" / "03_model_export.ipynb",
    
    # Source code (Reusable modules)
    BASE_DIR / "src" / "__init__.py",
    BASE_DIR / "src" / "preprocessing.py",
    BASE_DIR / "src" / "utils.py",
    
    # Backend (Project 1: FastAPI RestAPI)
    BASE_DIR / "backend" / "__init__.py",
    BASE_DIR / "backend" / "app.py",
    BASE_DIR / "backend" / "schemas.py",
    BASE_DIR / "backend" / "requirements_backend.txt",
    
    # Frontend (Project 2: Streamlit UI)
    BASE_DIR / "frontend" / "app.py",
    BASE_DIR / "frontend" / "requirements_frontend.txt",
    
    # Project configs & docs
    BASE_DIR / ".gitignore",
    BASE_DIR / "requirements.txt",
    BASE_DIR / "README.md",
]

# Nội dung mẫu mặc định cho .gitignore
GITIGNORE_CONTENT = """# =============================================================================
# Python & Environment
# =============================================================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
.venv/
venv/
env/
ENV/
env.bak/
venv.bak/

# =============================================================================
# Jupyter Notebooks
# =============================================================================
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# =============================================================================
# Data & Datasets (Bỏ qua dữ liệu thô/lớn, giữ lại .gitkeep)
# =============================================================================
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# =============================================================================
# Machine Learning Model Artifacts
# =============================================================================
models/*
!models/.gitkeep
*.pkl
*.joblib
*.h5
*.onnx

# =============================================================================
# Environment Secrets & Logs
# =============================================================================
.env
.env.local
*.log

# =============================================================================
# IDE & System Files
# =============================================================================
.vscode/
.idea/
.DS_Store
Thumbs.db
"""

# Nội dung mẫu mặc định cho requirements.txt (toàn bộ dự án)
REQUIREMENTS_CONTENT = """# =============================================================================
# Crop Recommendation System - Dependencies
# =============================================================================

# Backend RestAPI (Project 1)
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic>=2.0.0

# Frontend User Interface (Project 2)
streamlit>=1.25.0
requests>=2.31.0

# Machine Learning & Data Processing (Project 3)
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0

# Data Visualization & Exploratory Analysis
matplotlib>=3.7.0
seaborn>=0.12.0

# Jupyter Environment
jupyter>=1.0.0
ipykernel>=6.25.0
"""

# Nội dung mẫu cho requirements_backend.txt
REQUIREMENTS_BACKEND_CONTENT = """fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
"""

# Nội dung mẫu cho requirements_frontend.txt
REQUIREMENTS_FRONTEND_CONTENT = """streamlit>=1.25.0
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
"""

# Nội dung mẫu cho file Jupyter Notebook rỗng hợp lệ (chuẩn JSON)
EMPTY_NOTEBOOK_JSON = json.dumps({
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}, indent=1)

# Nội dung mẫu cho README.md
README_CONTENT = """# Hệ thống khuyến nghị cây trồng (Crop Recommendation System)

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
venv\\Scripts\\activate   # Windows
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
"""

# Ánh xạ nội dung mặc định cho các file tương ứng
DEFAULT_FILE_CONTENTS = {
    ".gitignore": GITIGNORE_CONTENT,
    "requirements.txt": REQUIREMENTS_CONTENT,
    "requirements_backend.txt": REQUIREMENTS_BACKEND_CONTENT,
    "requirements_frontend.txt": REQUIREMENTS_FRONTEND_CONTENT,
    "README.md": README_CONTENT,
}


def create_project_template():
    """Tự động tạo cây thư mục và khởi tạo file theo chuẩn MLOps."""
    print("=" * 70)
    print(" BẮT ĐẦU KHỞI TẠO CẤU TRÚC DỰ ÁN: CROP RECOMMENDATION SYSTEM")
    print("=" * 70)

    created_folders = set()
    created_files_count = 0
    skipped_files_count = 0

    for file_path in FILES_TO_CREATE:
        path = Path(file_path)
        parent_dir = path.parent

        # 1. Kiểm tra và tạo thư mục cha nếu chưa tồn tại
        if parent_dir not in created_folders and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            created_folders.add(parent_dir)
            print(f"[THƯ MỤC MỚI] Tạo thư mục: {parent_dir}")

        # 2. Kiểm tra file đã tồn tại hay chưa
        if not path.exists():
            # Xác định nội dung cần ghi vào file
            filename = path.name
            if filename in DEFAULT_FILE_CONTENTS:
                content = DEFAULT_FILE_CONTENTS[filename]
            elif path.suffix == ".ipynb":
                content = EMPTY_NOTEBOOK_JSON
            else:
                content = ""  # File rỗng

            # Tạo và ghi file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            created_files_count += 1
            print(f"[FILE MỚI]     Tạo file:    {path}")
        else:
            skipped_files_count += 1
            print(f"[ĐÃ TỒN TẠI]   Bỏ qua file: {path} (không ghi đè)")

    print("-" * 70)
    print(f"Hoàn thành! Tạo mới {created_files_count} file, bỏ qua {skipped_files_count} file đã tồn tại.")
    print("=" * 70)


if __name__ == "__main__":
    create_project_template()
