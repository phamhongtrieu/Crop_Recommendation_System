"""Giao diện Streamlit gọi REST API gợi ý cây trồng."""

import requests
import streamlit as st


st.set_page_config(
    page_title="Gợi ý cây trồng",
    page_icon="🌱",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { background: linear-gradient(180deg, #f4fbf5 0%, #ffffff 35%); }
    .hero { padding: 1.2rem 0 .5rem; }
    .hero h1 { color: #14532d; margin-bottom: .25rem; }
    .result-card { background: #ecfdf3; border: 1px solid #bbf7d0;
        border-radius: 16px; padding: 1.25rem; text-align: center; }
    .result-label { color: #166534; font-size: .95rem; }
    .result-crop { color: #14532d; font-size: 2rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Cấu hình API")
    api_url = st.text_input(
        "URL backend",
        value="http://127.0.0.1:8000",
        help="Nhập địa chỉ gốc của backend FastAPI; ứng dụng sẽ gọi /predict.",
    ).strip().rstrip("/")
    timeout_seconds = st.number_input(
        "Thời gian chờ tối đa (giây)", min_value=1, max_value=120, value=10
    )
    st.caption("Backend cần đang chạy và cung cấp endpoint POST /predict.")

st.markdown(
    '<div class="hero"><h1>🌱 Hệ thống gợi ý cây trồng</h1>'
    '<p>Nhập điều kiện đất và thời tiết để tìm loại cây phù hợp.</p></div>',
    unsafe_allow_html=True,
)
st.info("Điền đủ 7 thông số nông nghiệp bên dưới, sau đó nhấn **Dự đoán**.")

with st.form("crop_recommendation_form"):
    st.subheader("Thông số nông nghiệp")
    col1, col2 = st.columns(2)
    with col1:
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0, step=1.0)
        potassium = st.number_input("Potassium (K)", min_value=0.0, value=43.0, step=1.0)
        temperature = st.number_input("Temperature (°C)", value=20.0, step=0.1, format="%.2f")
    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, format="%.2f")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0, step=1.0)
    submitted = st.form_submit_button("🔎 Dự đoán / Recommend Crop", type="primary", use_container_width=True)


def crop_from_response(data):
    """Lấy tên cây từ các tên trường response phổ biến của backend."""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("prediction", "predicted_crop", "crop", "recommended_crop", "label"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


if submitted:
    # Soát các giá trị trước khi gửi để lỗi nhập liệu được báo rõ ràng.
    values = {
        "Nitrogen (N)": nitrogen,
        "Phosphorus (P)": phosphorus,
        "Potassium (K)": potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "pH": ph,
        "Rainfall": rainfall,
    }
    invalid = [name for name, value in values.items() if value != value]
    if invalid:
        st.error("Dữ liệu không hợp lệ. Vui lòng kiểm tra: " + ", ".join(invalid) + ".")
    elif not api_url:
        st.error("Vui lòng nhập URL backend trong thanh cấu hình bên trái.")
    else:
        payload = {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }
        try:
            with st.spinner("Đang gửi thông số và chờ kết quả dự đoán..."):
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=float(timeout_seconds),
                )
            if not response.ok:
                detail = ""
                try:
                    error_body = response.json()
                    detail = error_body.get("detail", error_body) if isinstance(error_body, dict) else error_body
                except ValueError:
                    detail = response.text.strip()
                st.error(f"API trả về lỗi HTTP {response.status_code}: {detail or 'Không có mô tả lỗi.'}")
            else:
                try:
                    result = response.json()
                except ValueError:
                    st.error("API trả về dữ liệu không đúng định dạng JSON.")
                else:
                    crop = crop_from_response(result)
                    if crop:
                        st.markdown(
                            '<div class="result-card"><div class="result-label">Cây trồng được đề xuất</div>'
                            f'<div class="result-crop">🌾 {crop}</div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("API phản hồi thành công nhưng không tìm thấy tên cây trồng. Hãy kiểm tra định dạng response của backend.")
                        st.caption(f"Phản hồi nhận được: {result}")
        except requests.exceptions.Timeout:
            st.error(f"API phản hồi quá thời gian chờ ({timeout_seconds} giây). Vui lòng thử lại.")
        except requests.exceptions.ConnectionError:
            st.error(f"Không thể kết nối tới API tại {api_url}. Hãy kiểm tra backend và địa chỉ API.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Không thể gửi yêu cầu tới API: {exc}")
