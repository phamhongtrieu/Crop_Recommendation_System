"""Giao diện Streamlit gọi REST API gợi ý cây trồng."""

import csv
from html import escape
import math
from numbers import Real
from pathlib import Path
import random

import requests
import streamlit as st


# Supported demo ranges; these are not absolute physical limits.
INPUT_LIMITS = {
    "N": (0.0, 140.0), "P": (0.0, 145.0), "K": (0.0, 205.0),
    "temperature": (0.0, 50.0), "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0), "rainfall": (0.0, 300.0),
}
INPUT_LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "pH", "rainfall": "Rainfall (mm)",
}


def validate_inputs(values):
    """Validate all seven raw inputs without coercing, clamping or dropping errors."""
    errors = {}
    for field, (lower, upper) in INPUT_LIMITS.items():
        value = values.get(field)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not lower <= value <= upper or not math.isfinite(value)):
            errors[field] = (f"Giá trị {INPUT_LABELS[field]} không hợp lệ. "
                             f"Vui lòng nhập số hữu hạn từ {lower:g} đến {upper:g}.")
    return errors


# Keys match the 22 classes in models/model_metadata.json (including mango).
CROP_NAMES = {
    "apple": ("Táo", "Apple", "🍎"),
    "banana": ("Chuối", "Banana", "🍌"),
    "blackgram": ("Đậu mười", "Black gram", "🫘"),
    "chickpea": ("Đậu gà", "Chickpea", "🫘"),
    "coconut": ("Dừa", "Coconut", "🥥"),
    "coffee": ("Cà phê", "Coffee", "☕"),
    "cotton": ("Bông", "Cotton", "🌿"),
    "grapes": ("Nho", "Grapes", "🍇"),
    "jute": ("Đay", "Jute", "🌿"),
    "kidneybeans": ("Đậu thận", "Kidney beans", "🫘"),
    "lentil": ("Đậu lăng", "Lentil", "🫘"),
    "maize": ("Ngô", "Maize", "🌽"),
    "mango": ("Xoài", "Mango", "🥭"),
    "mothbeans": ("Đậu bướm", "Moth beans", "🫘"),
    "mungbean": ("Đậu xanh", "Mung bean", "🫘"),
    "muskmelon": ("Dưa lưới", "Muskmelon", "🍈"),
    "orange": ("Cam", "Orange", "🍊"),
    "papaya": ("Đu đủ", "Papaya", "🌱"),
    "pigeonpeas": ("Đậu triều", "Pigeon peas", "🫘"),
    "pomegranate": ("Lựu", "Pomegranate", "🌱"),
    "rice": ("Lúa", "Rice", "🌾"),
    "watermelon": ("Dưa hấu", "Watermelon", "🍉"),
}


def crop_display_name(crop):
    names = CROP_NAMES.get(crop.strip().lower())
    if names:
        vietnamese, english, emoji = names
        return f"{emoji} {vietnamese} ({english})"
    return f"🌱 {crop}"


def probability_text(probability):
    if 0 < probability < 0.0001:
        return "<0.01%"
    if probability < 1 and probability * 100 >= 99.995:
        return ">99.99%"
    return f"{probability * 100:.2f}%"


def top_crops_from_response(data):
    """Validate API values without inventing, clamping or renormalizing probabilities."""
    raw = data.get("top_3") if isinstance(data, dict) else None
    if raw is None or raw == []:
        return [], False
    if not isinstance(raw, list):
        return [], True
    valid, seen = [], set()
    invalid = False
    for item in raw:
        if not isinstance(item, dict):
            invalid = True
            continue
        crop, probability = item.get("crop"), item.get("probability")
        if (not isinstance(crop, str) or not crop.strip()
                or isinstance(probability, bool) or not isinstance(probability, Real)
                or not 0 <= probability <= 1 or not math.isfinite(probability)):
            invalid = True
            continue
        key = crop.strip().lower()
        if key in seen:
            invalid = True
            continue
        seen.add(key)
        valid.append({"crop": crop.strip(), "probability": float(probability)})
    return sorted(valid, key=lambda item: -item["probability"])[:3], invalid


def show_prediction(crop, result):
    top_crops, invalid = top_crops_from_response(result)
    probability = next((item["probability"] for item in top_crops
                        if item["crop"].lower() == crop.strip().lower()), None)
    probability_label = (
        f"Xác suất ước lượng của mô hình: {probability_text(probability)}"
        if probability is not None else "Chưa có xác suất hợp lệ cho cây được đề xuất."
    )
    st.markdown(
        '<div class="result-card"><div class="result-label">Cây trồng được đề xuất</div>'
        f'<div class="result-crop">{escape(crop_display_name(crop))}</div>'
        f'<div class="result-label">{escape(probability_label)}</div></div>',
        unsafe_allow_html=True,
    )
    st.subheader("🌿 Top 3 cây trồng")
    if invalid:
        st.warning("Một số dữ liệu Top 3 không hợp lệ đã được bỏ qua. Không thể hiển thị xác suất cho các mục này.")
    if top_crops:
        for rank, item in enumerate(top_crops, start=1):
            st.text(f"{rank}. {crop_display_name(item['crop'])} — {probability_text(item['probability'])}")
            # st.progress truncates to whole percentages in Streamlit 1.67.
            # Keep sub-percent widths; only the fill, never the track, is green.
            percentage = item["probability"] * 100
            st.markdown(
                '<div class="crop-probability-track" role="progressbar" '
                f'aria-label="Xác suất {escape(item["crop"], quote=True)}" '
                f'aria-valuenow="{percentage:.15g}" aria-valuemin="0" aria-valuemax="100">'
                f'<div class="crop-probability-fill" style="width: {percentage:.15g}%"></div>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Backend chưa cung cấp danh sách Top 3 hợp lệ cho kết quả này.")
    st.caption("Xác suất trên là ước lượng từ mô hình Machine Learning, không phải tỷ lệ đảm bảo cây trồng sẽ phát triển tốt trong thực tế.")


@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
def backend_status(api_url):
    """Cache each URL's health briefly to avoid a request on every rerun."""
    if not api_url:
        return "⚪ Chưa cấu hình kết nối Backend."
    try:
        response = requests.get(f"{api_url}/health", timeout=2.0)
        response.raise_for_status()
        health = response.json()
        if not isinstance(health, dict) or health.get("status") != "ok":
            return "🟠 Chưa xác nhận được trạng thái Backend."
        if health.get("model_loaded") is True:
            return "🟢 Backend đã kết nối · Mô hình sẵn sàng."
        return "🟠 Backend đã kết nối · Mô hình chưa sẵn sàng."
    except (requests.exceptions.RequestException, ValueError):
        return "🔴 Chưa kết nối được Backend."


def clear_prediction():
    st.session_state.pop("last_prediction", None)


def agricultural_input(field, help_text, error, step=0.1, number_format=None):
    """Scope visual validation to a stable container, keeping native controls."""
    with st.container(key=f"field_{field}", gap="small"):
        options = {"format": number_format} if number_format else {}
        value = st.number_input(
            INPUT_LABELS[field], value=None, step=step, key=f"crop_input_{field}",
            help=help_text, on_change=clear_prediction, **options,
        )
        style, message = "", ""
        if error:
            lower, upper = INPUT_LIMITS[field]
            message = f"{INPUT_LABELS[field]} phải nằm trong khoảng {lower:g}–{upper:g}."
            # Keep the existing border width/geometry. The inset adds a second
            # red pixel without resizing the input or styling stepper buttons.
            style = (
                f'<style>.st-key-field_{field} [data-testid="stNumberInputContainer"] '
                '{border-color: #dc2626; box-shadow: inset 0 0 0 1px #dc2626;}</style>'
            )
        st.markdown(
            style + '<div class="crop-field-error" aria-live="polite">'
            + (escape(message) if message else "&nbsp;") + '</div>',
            unsafe_allow_html=True,
        )
        return value


def load_sample_inputs():
    """Choose a label uniformly, then a valid raw row, before widgets render."""
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Crop_recommendation.csv"
    fields = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not set((*fields, "label")).issubset(reader.fieldnames or []):
                raise ValueError("Missing required sample columns")
            groups = {}
            for row in reader:
                label = (row.get("label") or "").strip()
                try:
                    values = {field: float(row[field]) for field in fields}
                except (ValueError, TypeError, KeyError):
                    continue
                if label and not validate_inputs(values):
                    groups.setdefault(label, []).append(values)
        if not groups:
            raise ValueError("No valid sample rows")
        previous = st.session_state.get("previous_sample_label")
        labels = [label for label in groups if label != previous] or list(groups)
        label = random.choice(labels)
        previous_values = st.session_state.get("selected_sample", {}).get("values")
        candidates = [row for row in groups[label] if row != previous_values] or groups[label]
        values = random.choice(candidates)
    except FileNotFoundError:
        st.session_state["sample_error"] = "Không tìm thấy file CSV dữ liệu mẫu. Bạn vẫn có thể nhập thông số thủ công."
        return
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, csv.Error):
        st.session_state["sample_error"] = "Không đọc được CSV hoặc không có dòng hợp lệ trong phạm vi demo. Bạn vẫn có thể nhập thông số thủ công."
        return
    # Update only after the complete row is validated; do not call /predict.
    st.session_state.update({f"crop_input_{field}": value for field, value in values.items()})
    st.session_state["previous_sample_label"] = label
    st.session_state["selected_sample"] = {"label": label, "values": values.copy()}
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("sample_error", None)


st.set_page_config(
    page_title="Gợi ý cây trồng",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMain"] { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f8f8f8; }
    [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }
    .hero { padding: .25rem 0 .5rem; }
    .hero h1 { color: #14532d; margin-bottom: .25rem; }
    .result-card { background: #EAF8EF; border: 1px solid #bbf7d0;
        border-radius: 16px; padding: 1.25rem; text-align: center; }
    .result-label { color: #166534; font-size: .95rem; }
    .result-crop { color: #14532d; font-size: clamp(1.25rem, 2.5vw, 2rem);
        font-weight: 700; overflow-wrap: anywhere; }
    .st-key-crop_inputs { background-color: #ffffff; }
    .st-key-crop_inputs [data-testid="stNumberInputContainer"] {
        background-color: #f8f8f8; border: 1px solid #d9d9d9; border-radius: 10px; }
    .st-key-crop_inputs [data-testid="stNumberInputContainer"] input {
        background-color: transparent; color: #262626; }
    .st-key-crop_inputs [data-testid="stButton"] button {
        background-color: #237A57; border-color: #237A57; color: #ffffff; }
    .st-key-crop_inputs [data-testid="stButton"] button:not(:disabled):hover {
        background-color: #14532D; border-color: #14532D; color: #ffffff; }
    .crop-probability-track { width: 100%; height: .5rem; background-color: #E5E7EB;
        border-radius: .5rem; overflow: hidden; }
    .crop-probability-fill { height: 100%; background-color: #237A57; }
    .crop-field-error { min-height: 2rem; line-height: 1rem; font-size: .75rem;
        color: #b91c1c; }
    .sample-reference { height: 5rem; overflow: auto; overflow-wrap: anywhere;
        color: #166534; font-size: .85rem; line-height: 1.4; }
    @media (max-width: 850px) {
        [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
            flex-direction: column; }
        [data-testid="stMainBlockContainer"] [data-testid="stColumn"] {
            width: 100%; flex: 1 1 100%; min-width: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🌱 Crop Recommendation System")
    connection_status = st.empty()
    with st.expander("⚙️ Cài đặt nâng cao", expanded=False):
        api_url = st.text_input(
            "URL backend",
            value="http://127.0.0.1:8000",
            help="Nhập địa chỉ gốc của backend FastAPI; ứng dụng sẽ gọi /predict.",
        ).strip().rstrip("/")
        timeout_seconds = st.number_input(
            "Thời gian chờ tối đa (giây)", min_value=1, max_value=120, value=10
        )
        st.caption("Backend cần đang chạy và cung cấp endpoint POST /predict.")
    connection_status.caption(backend_status(api_url))

st.markdown(
    '<div class="hero"><h1>🌱 Hệ thống gợi ý cây trồng</h1>'
    '<p>Nhập điều kiện đất và thời tiết để tìm loại cây phù hợp.</p></div>',
    unsafe_allow_html=True,
)
input_column, result_column = st.columns([1.1, 1], gap="large")

# Session state is the single source of initial and sample widget values.
for field, default in {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 20.0,
                       "humidity": 80.0, "ph": 6.5, "rainfall": 200.0}.items():
    st.session_state.setdefault(f"crop_input_{field}", default)
input_errors = validate_inputs({field: st.session_state[f"crop_input_{field}"] for field in INPUT_LIMITS})

with input_column:
    if st.button("🎲 Random dữ liệu mẫu", key="load_sample", on_click=load_sample_inputs):
        if st.session_state.get("sample_error"):
            st.toast(st.session_state["sample_error"], icon="⚠️")
        else:
            st.toast("Đã random dữ liệu mẫu. Nhấn Dự đoán để xem kết quả.", icon="✅")

with input_column, st.container(border=True, key="crop_inputs"):
    st.subheader("Thông số nông nghiệp")
    st.caption("Giới hạn nhập là phạm vi hỗ trợ của bản demo, không phải giới hạn vật lý tuyệt đối.")
    col1, col2 = st.columns(2)
    with col1:
        # Native min/max rejects out-of-range edits in the browser without a
        # rerun. Validate in Python instead, so those edits invalidate results.
        nitrogen = agricultural_input("N", "Hàm lượng đạm trong đất.", input_errors.get("N"), step=1.0)
        phosphorus = agricultural_input("P", "Hàm lượng lân trong đất.", input_errors.get("P"), step=1.0)
        potassium = agricultural_input("K", "Hàm lượng kali trong đất.", input_errors.get("K"), step=1.0)
        temperature = agricultural_input("temperature", "Nhiệt độ môi trường (°C).",
                                        input_errors.get("temperature"), number_format="%.2f")
    with col2:
        humidity = agricultural_input("humidity", "Độ ẩm không khí (%).", input_errors.get("humidity"))
        ph = agricultural_input("ph", "Độ chua/kiềm của đất.", input_errors.get("ph"), number_format="%.2f")
        rainfall = agricultural_input("rainfall", "Lượng mưa (mm).", input_errors.get("rainfall"), step=1.0)
    current_payload = dict(N=nitrogen, P=phosphorus, K=potassium, temperature=temperature,
                           humidity=humidity, ph=ph, rainfall=rainfall)
    validation_errors = validate_inputs(current_payload)
    submitted = st.button("🔎 Dự đoán cây trồng", type="primary", use_container_width=True,
                          key="predict_crop", disabled=bool(validation_errors))


def crop_from_response(data):
    """Lấy tên cây từ các tên trường response phổ biến của backend."""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("recommended_crop", "prediction", "predicted_crop", "crop", "label"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


with result_column:
    st.subheader("Kết quả dự đoán")
    previous_prediction = st.session_state.get("last_prediction")
    # Reject legacy records and results that belong to different input values.
    if previous_prediction and (
        validation_errors
        or not isinstance(previous_prediction, dict)
        or not previous_prediction.get("explicit_submit")
        or previous_prediction.get("payload") != current_payload
    ):
        clear_prediction()
    if validation_errors:
        st.warning("⚠️ Vui lòng kiểm tra và sửa các thông số không hợp lệ ở bên trái.")
    if submitted:
        # A failed attempt must never leave an older successful result on display.
        st.session_state.pop("last_prediction", None)
        # Soát các giá trị trước khi gửi để lỗi nhập liệu được báo rõ ràng.
        if validate_inputs(current_payload):
            # Defensive guard even if a disabled button is triggered externally.
            pass
        elif not api_url:
            st.error("Vui lòng nhập URL backend trong thanh cấu hình bên trái.")
        else:
            payload = current_payload.copy()
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
                            st.session_state["last_prediction"] = {
                                "crop": crop, "result": result, "payload": payload.copy(),
                                "explicit_submit": True,
                            }
                        else:
                            st.error("API phản hồi thành công nhưng không tìm thấy tên cây trồng. Hãy kiểm tra định dạng response của backend.")
                            st.caption(f"Phản hồi nhận được: {result}")
            except requests.exceptions.Timeout:
                st.error(f"API phản hồi quá thời gian chờ ({timeout_seconds} giây). Vui lòng thử lại.")
            except requests.exceptions.ConnectionError:
                st.error(f"Không thể kết nối tới API tại {api_url}. Hãy kiểm tra backend và địa chỉ API.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Không thể gửi yêu cầu tới API: {exc}")

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction:
        st.caption("Kết quả của lần dự đoán gần nhất. Nếu đã thay đổi thông số, hãy nhấn Dự đoán cây trồng để cập nhật kết quả.")
        show_prediction(last_prediction["crop"], last_prediction["result"])
    elif not submitted and not validation_errors:
        st.info("Nhấn Dự đoán cây trồng để xem kết quả")
