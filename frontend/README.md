# Frontend Streamlit – Crop Recommendation

Giao diện giữ nguyên form 7 thông số. Sidebar hiển thị tên hệ thống, trạng thái
Backend và mục **⚙️ Cài đặt nâng cao** đóng mặc định, chứa URL/timeout.
Mặc định URL là `http://127.0.0.1:8000`, timeout dự đoán là 10 giây.
Trạng thái lấy từ `/health` với timeout 2 giây, cache 30 giây theo URL để tránh
gọi lại mỗi lần render. Trạng thái có thể chậm tối đa khoảng 30 giây so với thực tế;
việc dự đoán vẫn gọi API trực tiếp với cấu hình hiện tại.
Nhấn **Dự đoán** để gọi `POST /predict` bằng Requests và hiển thị kết quả thực
từ Backend. Chạy từ thư mục gốc với môi trường đã có dependency:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Mở <http://localhost:8501>. Backend mặc định là <http://127.0.0.1:8000>.

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

Phạm vi nhập liệu khớp Backend: N 0–140, P 0–145, K 0–205, temperature 0–50 °C,
humidity 0–100%, ph 0–14, rainfall 0–300 mm; đều gồm hai đầu. Giá trị phải là số
hữu hạn. Chưa xác minh đơn vị N/P/K và chu kỳ tổng hợp rainfall từ nguồn dataset;
không mặc định đơn vị hoặc tự đổi giá trị. Đây là phạm vi demo.

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
  Xác suất cây chính chỉ được hiển thị nếu tìm thấy đúng nhãn trong Top 3 hợp lệ.
- `top_3` phải là danh sách các đối tượng có `crop` là chuỗi không rỗng và
  `probability` là số hữu hạn trong [0, 1]. Không nhận boolean, chuỗi số hoặc null.
  Mục lỗi/trùng nhãn được bỏ qua kèm cảnh báo; không tự sửa giá trị xác suất.
- Sắp xếp giảm dần, giữ thứ tự gốc khi đồng hạng, hiển thị tối đa 3 mục.
  Thanh tiến trình nhận xác suất gốc 0–1. Nội dung phần trăm dùng hai chữ số
  thập phân, trừ xác suất dương dưới 0.01% hiển thị `<0.01%` và xác suất rất
  gần 100% nhưng chưa bằng 100% hiển thị `>99.99%`.
- Thiếu/rỗng Top 3 vẫn hiển thị cây chính và thông báo chưa có xác suất.
  Không tạo danh sách hoặc xác suất thay thế; không chuẩn hóa lại tổng Top 3.
- Có ghi chú: xác suất là ước lượng từ mô hình, không bảo đảm cây phát triển tốt.

Bảng tên trong `CROP_NAMES` khớp 22 nhãn của `models/model_metadata.json`,
bao gồm `mango` → Xoài. Hai nhãn dễ nhầm được đối chiếu theo Feedipedia:
[black gram → Đậu mười](https://www.feedipedia.org/node/236),
[moth bean → Đậu bướm](https://www.feedipedia.org/node/237).

## Kiểm tra thủ công

1. Chạy Backend và Frontend; nhập đủ 7 thông số. Dùng mẫu:
   N=90, P=42, K=43, temperature=20.87974371, humidity=82.00274423,
   ph=6.502985292, rainfall=202.9355362.
2. Nhấn dự đoán: với artifact hiện tại kết quả chính là Lúa (Rice), Top 3 là
   Rice, Jute, Coffee. Các thanh tiến trình và phần trăm lấy từ API.
3. Mở **⚙️ Cài đặt nâng cao** trong sidebar, đổi URL hoặc timeout rồi đóng lại.
   Mở lại để kiểm tra giá trị được giữ và thử dự đoán bằng cấu hình mới.
4. Nhập URL của cổng không có Backend: giao diện phải báo không thể kết nối.
5. Với API trả response cũ chỉ có `recommended_crop`, vẫn thấy tên cây chính,
   không thấy thanh xác suất giả. Có thể kiểm tra các response lỗi bằng
   Streamlit AppTest và mock `requests.post`, không cần đổi Backend thật.

Đã kiểm tra bằng `py_compile`, Streamlit AppTest cho response mới/cũ, dữ liệu
Top 3 lỗi, lỗi HTTP, lỗi JSON, timeout và mất kết nối. Đã kiểm tra luồng gọi
HTTP tới Backend với mô hình thật và khởi động server Streamlit. AppTest kiểm
tra phần tử giao diện; chưa đánh giá trực quan bố cục trong trình duyệt.
