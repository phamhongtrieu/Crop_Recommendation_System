# Frontend Streamlit – Crop Recommendation

Giao diện gồm form 7 thông số và vùng kết quả dự đoán, dùng toàn bộ chiều rộng
trang. Không có sidebar cấu hình. Ô nhập số ẩn nút xóa nhanh **×**, giữ nút tăng/giảm
và slider bên dưới.
Backend mặc định là `http://127.0.0.1:8000`, timeout dự đoán là 10 giây.
Nhấn **Dự đoán** để gọi `POST /predict` bằng Requests và hiển thị kết quả thực
từ Backend. Chạy từ thư mục gốc với môi trường đã có dependency:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Mở <http://localhost:8501>. Backend mặc định là <http://127.0.0.1:8000>.

Nếu Backend dùng địa chỉ khác, đặt biến môi trường trước khi chạy Frontend:

```powershell
$env:CROP_API_URL = "http://127.0.0.1:9000"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

## Dependencies và môi trường

Xem [README gốc](../README.md) để clone, tạo môi trường mới và cài
`requirements-artifact.txt` cho cả hệ thống. Không tạo lại hoặc cài đè `.venv`
đang hoạt động. Terminal Frontend phải kích hoạt cùng môi trường với Backend.
`frontend/requirements_frontend.txt` khai báo Streamlit và Requests cho giao diện;
NumPy/pandas là dependency của Streamlit, còn Backend khai báo chúng trực tiếp.

Streamlit được ghim **1.67.0**, là bản đã chạy bộ AppTest hiện có. Mức 1.25 cũ
không hỗ trợ đủ `st.container(key=..., gap=..., border=...)`. Việc toàn bộ UI/CSS
và test chạy đúng trên các bản cũ hơn 1.67.0 **CHƯA XÁC MINH**. Không nâng cấp
thư viện tự động khi khởi động giao diện.

Giao diện dùng theme sáng, chữ tối để dễ đọc. Mỗi ô nhập có slider và dòng khoảng
giá trị bên dưới; nhập số hoặc kéo slider đều cập nhật giá trị còn lại. Khi nhập
ngoài khoảng, ô báo lỗi và nút Dự đoán bị khóa; kéo slider sẽ đưa ô về giá trị hợp lệ.
Mỗi lần thay đổi thông số sẽ xóa kết quả dự đoán cũ.
Các ô nhập, slider, khoảng giá trị bên dưới và thông báo lỗi đều hiển thị 2 chữ số
sau dấu phẩy; giá trị gốc vẫn được giữ nguyên khi kiểm tra phạm vi và gửi dự đoán.

Phạm vi nhập liệu khớp Backend, gồm cả hai đầu:

| Input | Min | Max | Step |
| --- | ---: | ---: | ---: |
| Nitrogen (N) | 0 | 140 | 1 |
| Phosphorus (P) | 5 | 145 | 1 |
| Potassium (K) | 5 | 205 | 1 |
| Temperature (°C) | 8.825674745 | 43.67549305 | 0.01 |
| Humidity (%) | 14.25803981 | 99.98187601 | 0.01 |
| pH | 3.504752314 | 9.93509073 | 0.01 |
| Rainfall (mm) | 20.21126747 | 298.5601175 | 0.01 |

Ô số và slider dùng chung Min/Max/Step. Range được kiểm tra tại cả Frontend và
Backend. Random lọc các dòng CSV nằm ngoài range này trước khi chọn mẫu.
Min/Max lấy trực tiếp từ CSV raw qua `src/input_config.py`, giữ nguyên các chữ số
thập phân. Cả 2.200 dòng raw đều nằm trong range này. Khởi động lại ứng dụng
sau khi đổi CSV raw để nạp lại giới hạn.

Giá trị phải là số hữu hạn. Chưa xác minh đơn vị N/P/K và chu kỳ tổng hợp rainfall từ nguồn dataset;
không mặc định đơn vị hoặc tự đổi giá trị. Đây là phạm vi quan sát trong dữ liệu raw.

Chạy test từ thư mục gốc sau khi cài dependencies đầy đủ:

```powershell
python -B -m pytest -p no:cacheprovider frontend/test_app.py -q
```

Các test hiện có kiểm tra random chỉ điền dữ liệu, chặn dự đoán khi nhập sai,
xóa kết quả cũ khi sửa thông số và xử lý mất kết nối. Cài đặt toàn hệ thống trên
môi trường sạch theo cấu hình artifact **CHƯA XÁC MINH**; AppTest không thay thế
kiểm tra trực quan giao diện trong trình duyệt.

## Hiển thị kết quả

- Thẻ xanh hiển thị tên cây tiếng Việt, tiếng Anh và emoji. Nhãn chưa biết giữ
  tên gốc; dữ liệu API được escape trước khi đưa vào thẻ HTML.
- Ưu tiên `recommended_crop`; vẫn hỗ trợ các tên trường cũ của giao diện.
  Xác suất dự đoán của cây chính chỉ được hiển thị nếu tìm thấy đúng nhãn trong Top 3 hợp lệ.
- `top_3` phải là danh sách các đối tượng có `crop` là chuỗi không rỗng và
  `probability` là số hữu hạn trong [0, 1]. Không nhận boolean, chuỗi số hoặc null.
  Mục lỗi/trùng nhãn được bỏ qua kèm cảnh báo; không tự sửa xác suất dự đoán.
- Sắp xếp giảm dần, giữ thứ tự gốc khi đồng hạng, hiển thị tối đa 3 mục.
  Xác suất dự đoán lấy từ `probability` gốc 0–1. Nội dung phần trăm dùng hai chữ số
  thập phân, trừ giá trị dương dưới 0.01% hiển thị `<0.01%` và giá trị rất
  gần 100% nhưng chưa bằng 100% hiển thị `>99.99%`.
- Thiếu/rỗng Top 3 vẫn hiển thị cây chính và thông báo chưa có xác suất dự đoán.
  Không tạo danh sách hoặc xác suất thay thế; không chuẩn hóa lại tổng Top 3.
- Nhãn hiển thị là **Xác suất dự đoán** trên thẻ kết quả, từng cây trong Top 3 và nhãn hỗ trợ đọc màn hình.
  API vẫn dùng trường `probability`. Chú thích: “Giá trị do mô hình ước lượng,
  không phải tỷ lệ trồng thành công.”

Bảng tên trong `CROP_NAMES` khớp 22 nhãn của Random Forest trong
[`models/random_forest/model_metadata.json`](../models/random_forest/model_metadata.json),
bao gồm `mango` → Xoài. Hai nhãn dễ nhầm được đối chiếu theo Feedipedia:
[black gram → Đậu mười](https://www.feedipedia.org/node/236),
[moth bean → Đậu bướm](https://www.feedipedia.org/node/237).

## Kiểm tra thủ công

1. Chạy Backend và Frontend; nhập đủ 7 thông số. Dùng mẫu:
   N=90, P=42, K=43, temperature=20.87974371, humidity=82.00274423,
   ph=6.502985292, rainfall=202.9355362.
2. Nhấn dự đoán: Random Forest hiện tại dự đoán Lúa (Rice). Với mẫu trên, Top 3 là
   Rice (96%), Jute (4%), Apple (0%). Các thanh tiến trình và phần trăm lấy từ API.
3. Kiểm tra không có sidebar và nút xóa nhanh **×** trong ô số. Các nút tăng/giảm
   và slider vẫn cập nhật giá trị, xóa kết quả dự đoán cũ khi thông số thay đổi.
4. Cấu hình `CROP_API_URL` tới cổng không có Backend, chạy lại Frontend và nhấn
   Dự đoán: giao diện phải báo không thể kết nối.
5. Với API trả response cũ chỉ có `recommended_crop`, vẫn thấy tên cây chính,
   không thấy thanh xác suất giả. Có thể kiểm tra các response lỗi bằng
   Streamlit AppTest và mock `requests.post`, không cần đổi Backend thật.

Đã kiểm tra bằng `py_compile`, Streamlit AppTest cho response mới/cũ, dữ liệu
Top 3 lỗi, lỗi HTTP, lỗi JSON, timeout và mất kết nối. Đã kiểm tra luồng gọi
HTTP tới Backend với mô hình thật và khởi động server Streamlit. AppTest kiểm
tra phần tử giao diện; chưa đánh giá trực quan bố cục trong trình duyệt.
