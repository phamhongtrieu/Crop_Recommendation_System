"""Frontend regressions: run python -m pytest frontend/test_app.py -q."""

import ast
import csv
import math
import re
from contextlib import contextmanager
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from numbers import Real
from unittest.mock import Mock, mock_open, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

from src.input_config import format_bound
APP_PATH = Path(__file__).with_name("app.py")


class ProbabilityBars(HTMLParser):
    def __init__(self):
        super().__init__()
        self.widths = []
        self.values = []
        self.labels = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("class") == "crop-probability-track":
            self.values.append(float(attrs["aria-valuenow"]))
            self.labels.append(attrs["aria-label"])
        if attrs.get("class") == "crop-probability-fill":
            self.widths.append(float(attrs["style"].removeprefix("width: ").removesuffix("%")))


def parse_bars(app):
    parser = ProbabilityBars()
    for element in app.markdown:
        parser.feed(element.value)
    return parser


def highlighted_fields(app):
    return set(re.findall(
        r'\.st-key-field_(\w+) \[data-testid="stNumberInputContainer"\]',
        "\n".join(element.value for element in app.markdown),
    ))


def assert_validation_warning(app):
    right = app.main.columns[-1]
    assert len(right.warning) == 1
    assert right.warning[0].value == "Vui lòng kiểm tra và sửa các thông số không hợp lệ ở bên trái."
    assert right.warning[0].proto.icon == "⚠️"
    assert not right.error
    assert not right.info


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("CROP_API_URL", raising=False)
    response = Mock(ok=True)
    response.json.return_value = {
        "recommended_crop": "rice",
        "top_3": [
            {"crop": "rice", "probability": 0.9948},
            {"crop": "jute", "probability": 0.0052},
            {"crop": "coffee", "probability": 0.00001},
        ],
    }
    with patch("requests.post", return_value=response) as post:
        yield response, post


def test_fractional_probability_widths_and_appearance(api):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
    assert not app.exception
    assert not parse_bars(app).widths
    assert len(app.main.number_input) == 7
    assert not app.sidebar.expander and not app.sidebar.number_input
    assert not app.sidebar.header and not app.sidebar.text_input
    app.button(key="predict_crop").click().run()
    assert not app.exception
    bars = parse_bars(app)
    # Regression: jute must be 0.52% wide, never a full-width green track.
    assert bars.widths == pytest.approx([99.48, 0.52, 0.001])
    assert bars.values == pytest.approx(bars.widths)
    assert all(label.startswith("Xác suất dự đoán ") for label in bars.labels)
    assert bars.widths[1] < 1
    assert any("<0.01%" in element.value for element in app.text)
    assert app.main.columns[-1].subheader[1].value == "🌿 Top 3 cây trồng"
    css = app.markdown[0].value
    assert 'button[kind="primary"]' not in css
    assert '[data-testid="stProgress"]' not in css
    assert "background-color: #E5E7EB" in css
    assert css.count("background-color: #237A57") == 2
    # Only the prediction button is green; steppers and help buttons retain
    # Streamlit's native colors even though they share the same input container.
    assert ".st-key-crop_inputs button" not in css
    assert '.st-key-crop_inputs [data-testid="stButton"] button {' in css
    assert '[data-testid="stMain"] { background-color: #ffffff; }' in css
    assert any("Rice" in element.value for element in app.main.columns[-1].markdown)
    app.run()
    assert parse_bars(app).widths == pytest.approx([99.48, 0.52, 0.001])
    assert post.call_count == 1


def test_backend_url_can_be_configured_without_sidebar(api, monkeypatch):
    _, post = api
    monkeypatch.setenv("CROP_API_URL", " http://127.0.0.1:9000/ ")
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception and not app.sidebar.expander
    assert not app.text_input
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert post.call_args.args == ("http://127.0.0.1:9000/predict",)
    assert post.call_args.kwargs["timeout"] == 10.0


@pytest.mark.parametrize("probability", [0.0, 0.0052, 0.00001, 0.9948, 1.0])
def test_bar_uses_absolute_probability_without_rescaling(api, probability):
    response, _ = api
    response.json.return_value = {
        "recommended_crop": "rice", "top_3": [{"crop": "rice", "probability": probability}],
    }
    app = AppTest.from_file(str(APP_PATH)).run()
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert parse_bars(app).widths == pytest.approx([probability * 100])


def test_prediction_request_and_error_handling_unchanged(api):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    app.button(key="predict_crop").click().run()
    post.assert_called_once_with(
        "http://127.0.0.1:8000/predict",
        json=dict(N=90., P=42., K=43., temperature=20., humidity=80., ph=6.5, rainfall=200.),
        timeout=10.,
    )
    post.side_effect = requests.exceptions.ConnectionError()
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert app.error
    assert not parse_bars(app).widths


def test_all_input_tooltips(api):
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    expected = {
        "N": "Hàm lượng đạm trong đất.", "P": "Hàm lượng lân trong đất.",
        "K": "Hàm lượng kali trong đất.", "temperature": "Nhiệt độ môi trường (°C).",
        "humidity": "Độ ẩm không khí (%).", "ph": "Độ chua/kiềm của đất.",
        "rainfall": "Lượng mưa (mm).",
    }
    for field, help_text in expected.items():
        assert app.number_input(key=f"crop_input_{field}").proto.help == help_text
        lower, upper = DEMO_LIMITS[field]
        assert any(f"Khoảng giá trị: {format_bound(lower)}–{format_bound(upper)}" in caption.value for caption in app.caption)


def test_real_csv_sample_populates_form_without_predicting(api, monkeypatch, caplog):
    _, post = api
    csv_path = APP_PATH.parent.parent / "data" / "raw" / "Crop_recommendation.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    fields = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
    real_rows = {(row["label"], tuple(float(row[field]) for field in fields)) for row in rows}
    # Absolute dataset path must work even when launched outside project root.
    monkeypatch.chdir(APP_PATH.parent)
    app = AppTest.from_file(str(APP_PATH)).run()
    initial_layout = [element.type for element in app.main]
    previous_label = None
    for _ in range(20):
        app.button(key="load_sample").click().run()
        assert not app.exception
        assert not app.warning
        assert not app.success
        assert app.get("toast")
        assert [element.type for element in app.main] == initial_layout
        post.assert_not_called()
        sample = app.session_state["selected_sample"]
        label = sample["label"]
        assert label != previous_label
        previous_label = label
        expected = sample["values"]
        assert (label, tuple(expected[field] for field in fields)) in real_rows
        for field, value in expected.items():
            assert app.number_input(key=f"crop_input_{field}").value == value
            assert app.slider(key=f"crop_slider_{field}").value == value
        assert not any('class="sample-reference"' in item.value for item in app.markdown)
        assert not any('class="result-card"' in item.value for item in app.markdown)
        assert "Đã random dữ liệu mẫu. Nhấn Dự đoán để xem kết quả." in app.get("toast")[-1].proto.body
    assert not parse_bars(app).widths
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert post.call_args.kwargs["json"] == expected
    assert len(parse_bars(app).widths) == 3
    app.button(key="load_sample").click().run()
    assert not app.exception
    assert post.call_count == 1
    assert not parse_bars(app).widths
    assert not app.success
    assert "default value but also had its value set" not in caplog.text


@pytest.mark.parametrize("csv_data", [None, "", "N,P,K,temperature,humidity,ph,rainfall,label\n",
    "N,P,K\n", "N,P,K\n90,42,43\n",
    "N,P,K,temperature,humidity,ph,rainfall\n90,42,43,20,101,6.5,200\n",
    "N,P,K,temperature,humidity,ph,rainfall\n90,42,43,nan,80,6.5,200\n"])
def test_sample_read_error_preserves_existing_inputs(api, csv_data):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    previous = [item.value for item in app.main.number_input]
    original_open = Path.open

    def sample_open(path, *args, **kwargs):
        if path.name == "Crop_recommendation.csv":
            if csv_data is None:
                raise FileNotFoundError("Sample CSV is missing")
            return mock_open(read_data=csv_data)()
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", sample_open):
        app.button(key="load_sample").click().run()
    assert not app.exception
    assert app.get("toast")
    assert [item.value for item in app.main.number_input] == previous
    post.assert_not_called()


with (APP_PATH.parent.parent / "data/raw/Crop_recommendation.csv").open(encoding="utf-8-sig") as source:
    RAW_ROWS = list(csv.DictReader(source))
DEMO_LIMITS = {
    field: (min(float(row[field]) for row in RAW_ROWS), max(float(row[field]) for row in RAW_ROWS))
    for field in ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
}
EXPECTED_STEPS = {"N": 1., "P": 1., "K": 1., "temperature": 0.01,
                  "humidity": 0.01, "ph": 0.01, "rainfall": 0.01}


@pytest.mark.parametrize("field", DEMO_LIMITS)
def test_slider_sync_ranges_and_recovery_from_invalid_input(api, field):
    _, post = api
    lower, upper = DEMO_LIMITS[field]
    app = AppTest.from_file(str(APP_PATH)).run()
    assert len(app.main.slider) == 7
    slider = app.slider(key=f"crop_slider_{field}")
    assert slider.proto.min == lower and slider.proto.max == upper
    assert slider.proto.step == EXPECTED_STEPS[field]
    assert app.number_input(key=f"crop_input_{field}").proto.step == EXPECTED_STEPS[field]
    assert slider.value == app.number_input(key=f"crop_input_{field}").value

    app.button(key="predict_crop").click().run()
    app.slider(key=f"crop_slider_{field}").set_value(upper).run()
    assert not app.exception
    assert app.number_input(key=f"crop_input_{field}").value == upper
    assert "last_prediction" not in app.session_state
    assert post.call_count == 1

    app.number_input(key=f"crop_input_{field}").set_value(lower).run()
    assert app.slider(key=f"crop_slider_{field}").value == lower
    app.button(key="predict_crop").click().run()
    assert post.call_args.kwargs["json"][field] == lower
    assert post.call_count == 2

    app.number_input(key=f"crop_input_{field}").set_value(upper + 1).run()
    assert app.button(key="predict_crop").disabled
    assert app.slider(key=f"crop_slider_{field}").value == lower
    assert "last_prediction" not in app.session_state
    app.slider(key=f"crop_slider_{field}").set_value(upper).run()
    assert not app.exception and not app.warning
    assert app.number_input(key=f"crop_input_{field}").value == upper
    assert not app.button(key="predict_crop").disabled
    assert post.call_count == 2


@pytest.mark.parametrize("probability, expected_label", [
    (0.999999, "Xác suất dự đoán: >99.99%"),
    (None, "Chưa có xác suất dự đoán hợp lệ cho cây được đề xuất."),
])
def test_probability_labels_in_result_card_and_top_crops(api, probability, expected_label):
    response, _ = api
    response.json.return_value = {"recommended_crop": "rice"}
    if probability is not None:
        response.json.return_value["top_3"] = [{"crop": "rice", "probability": probability}]
    app = AppTest.from_file(str(APP_PATH)).run()
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert any(expected_label in unescape(element.value) for element in app.markdown)
    if probability is not None:
        assert any(expected_label in element.value for element in app.text)
        assert parse_bars(app).widths == pytest.approx([probability * 100])
    visible_text = "\n".join(element.value for element in [*app.markdown, *app.caption, *app.text])
    assert "Độ tin cậy" not in visible_text and "độ tin cậy" not in visible_text
    assert any("Giá trị do mô hình ước lượng, không phải tỷ lệ trồng thành công." == caption.value for caption in app.caption)


@pytest.mark.parametrize("position", ["lower", "middle", "upper"])
def test_frontend_bounds_and_valid_payload(api, position):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    expected = {}
    for field, (lower, upper) in DEMO_LIMITS.items():
        widget = app.number_input(key=f"crop_input_{field}")
        # Browser range fences must not swallow invalid edits before Python
        # can clear a previous prediction. Demo ranges are validated server-side.
        assert not widget.proto.has_min
        assert not widget.proto.has_max
        value = {"lower": lower, "middle": (lower + upper) / 2, "upper": upper}[position]
        expected[field] = value
        widget.set_value(value)
    app.button(key="predict_crop").click().run()
    assert not app.exception and not app.error
    assert post.call_args.kwargs["json"] == expected
    assert len(parse_bars(app).widths) == 3
    assert not highlighted_fields(app)


@pytest.mark.parametrize("field", DEMO_LIMITS)
@pytest.mark.parametrize("side", ["below", "above"])
def test_stale_out_of_range_state_is_never_silently_clamped(api, field, side):
    _, post = api
    lower, upper = DEMO_LIMITS[field]
    app = AppTest.from_file(str(APP_PATH))
    app.session_state[f"crop_input_{field}"] = lower - 0.01 if side == "below" else upper + 0.01
    app.session_state[f"crop_slider_{field}"] = app.session_state[f"crop_input_{field}"]
    app.run()
    assert not app.exception
    assert_validation_warning(app)
    assert highlighted_fields(app) == {field}
    assert app.number_input(key=f"crop_input_{field}").value == (lower - 0.01 if side == "below" else upper + 0.01)
    assert lower <= app.slider(key=f"crop_slider_{field}").value <= upper
    assert app.button(key="predict_crop").disabled
    post.assert_not_called()


@pytest.mark.parametrize("field", DEMO_LIMITS)
def test_out_of_range_csv_sample_is_rejected_without_clamping(api, field):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    previous = [item.value for item in app.main.number_input]
    row = dict(N=90., P=42., K=43., temperature=20., humidity=80., ph=6.5, rainfall=200.)
    row[field] = DEMO_LIMITS[field][1] + 0.01
    row["label"] = "rice"
    data = ",".join(row) + "\n" + ",".join(str(value) for value in row.values()) + "\n"
    original_open = Path.open

    def sample_open(path, *args, **kwargs):
        if path.name == "Crop_recommendation.csv":
            return mock_open(read_data=data)()
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", sample_open):
        app.button(key="load_sample").click().run()
    assert not app.exception and app.get("toast")
    assert [item.value for item in app.main.number_input] == previous
    post.assert_not_called()


@contextmanager
def csv_source(data):
    original_open = Path.open

    def sample_open(path, *args, **kwargs):
        if path.name == "Crop_recommendation.csv":
            return mock_open(read_data=data)()
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", sample_open):
        yield


def test_uniform_label_selection_filters_invalid_rows_and_keeps_labels_hidden(api):
    _, post = api
    data = ("N,P,K,temperature,humidity,ph,rainfall,label\n"
            "90,42,43,20,80,6.5,200,rice\n"
            "91,42,43,20,80,6.5,200,rice\n"
            "50,30,25,25,70,6,100,maize\n"
            "141,30,25,25,70,6,100,invalid_range\n"
            "nan,30,25,25,70,6,100,invalid_number\n"
            ",30,25,25,70,6,100,missing_number\n"
            "50,30,25,25,70,6,100,\n")
    app = AppTest.from_file(str(APP_PATH)).run()
    choices = []

    def choose_first(options):
        choices.append(options.copy())
        return options[0]

    with csv_source(data), patch("random.choice", side_effect=choose_first):
        app.button(key="load_sample").click().run()
        assert app.session_state["selected_sample"]["label"] == "rice"
        app.button(key="load_sample").click().run()
        assert not app.exception
        assert app.session_state["selected_sample"]["label"] == "maize"
    # Class candidates are chosen before rows, regardless of per-class row count.
    assert choices[0] == ["rice", "maize"]
    assert len(choices[1]) == 2
    assert choices[2] == ["maize"]
    assert len(choices[3]) == 1
    post.assert_not_called()
    app.number_input(key="crop_input_N").set_value(51.)
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert post.call_args.kwargs["json"]["N"] == 51.
    assert "label" not in post.call_args.kwargs["json"]
    # Dataset maize must never replace the API's (mock) rice prediction.
    assert app.session_state["last_prediction"]["crop"] == "rice"
    assert not any('class="sample-reference"' in item.value for item in app.markdown)
    assert len(parse_bars(app).widths) == 3


def test_single_label_sample_works_while_backend_is_offline(api):
    _, post = api
    data = "N,P,K,temperature,humidity,ph,rainfall,label\n50,30,25,25,70,6,100,new_crop\n"
    post.side_effect = requests.exceptions.ConnectionError()
    app = AppTest.from_file(str(APP_PATH)).run()
    with csv_source(data):
        for _ in range(3):
            app.button(key="load_sample").click().run()
            assert not app.exception
            assert app.session_state["selected_sample"]["label"] == "new_crop"
            assert not any("new_crop" in item.value for item in app.markdown)
            post.assert_not_called()


@pytest.mark.parametrize("missing", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])
def test_missing_required_csv_column_is_safe(api, missing):
    _, post = api
    fields = [field for field in (*DEMO_LIMITS, "label") if field != missing]
    app = AppTest.from_file(str(APP_PATH)).run()
    with csv_source(",".join(fields) + "\n"):
        app.button(key="load_sample").click().run()
    assert not app.exception
    assert app.get("toast")
    assert "selected_sample" not in app.session_state
    post.assert_not_called()


@pytest.mark.parametrize("field", DEMO_LIMITS)
def test_editing_any_input_clears_prediction_without_api_call(api, field):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    app.button(key="load_sample").click().run()
    selected_sample = app.session_state["selected_sample"]
    post.assert_not_called()
    assert app.main.columns[-1].info[0].value == "Nhấn Dự đoán cây trồng để xem kết quả"
    app.button(key="predict_crop").click().run()
    assert len(parse_bars(app).widths) == 3
    app.run()
    assert len(parse_bars(app).widths) == 3
    assert post.call_count == 1
    widget = app.number_input(key=f"crop_input_{field}")
    lower, upper = DEMO_LIMITS[field]
    widget.set_value(lower if widget.value != lower else upper).run()
    assert not app.exception
    assert not parse_bars(app).widths
    assert "last_prediction" not in app.session_state
    assert app.session_state["selected_sample"] == selected_sample
    assert not any('class="sample-reference"' in item.value for item in app.markdown)
    assert app.main.columns[-1].info[0].value == "Nhấn Dự đoán cây trồng để xem kết quả"
    app.run()
    assert not parse_bars(app).widths
    assert post.call_count == 1
    app.button(key="predict_crop").click().run()
    assert post.call_count == 2
    assert len(parse_bars(app).widths) == 3


@pytest.mark.parametrize("record", [
    {"crop": "rice", "result": {"recommended_crop": "rice"}},
    {"crop": "rice", "explicit_submit": True, "payload": {"N": 10}},
    "legacy-result",
])
def test_stale_session_result_never_reappears_on_rerun(api, record):
    _, post = api
    app = AppTest.from_file(str(APP_PATH))
    app.session_state["last_prediction"] = record
    app.run()
    assert not app.exception
    assert "last_prediction" not in app.session_state
    assert not parse_bars(app).widths
    assert app.main.columns[-1].info[0].value == "Nhấn Dự đoán cây trồng để xem kết quả"
    app.button(key="load_sample").click().run()
    app.run()
    assert not app.exception
    assert "last_prediction" not in app.session_state
    assert not parse_bars(app).widths
    post.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("N", 90000.), ("P", -1.), ("K", 206.), ("temperature", 51.),
    ("humidity", 101.), ("ph", 15.), ("rainfall", 301.),
    ("N", None), ("temperature", float("nan")), ("rainfall", float("inf")),
])
def test_invalid_edit_hides_all_results_and_requires_new_prediction(api, field, value):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    app.button(key="predict_crop").click().run()
    assert len(parse_bars(app).widths) == 3
    widget = app.number_input(key=f"crop_input_{field}")
    original = widget.value
    widget.set_value(value).run()
    assert not app.exception
    assert_validation_warning(app)
    assert highlighted_fields(app) == {field}
    lower, upper = DEMO_LIMITS[field]
    assert any(f"{format_bound(lower)}–{format_bound(upper)}" in element.value for element in app.markdown)
    assert not parse_bars(app).widths
    assert not any('class="result-card"' in item.value for item in app.markdown)
    assert "last_prediction" not in app.session_state
    assert app.button(key="predict_crop").disabled
    assert post.call_count == 1
    with pytest.raises(AppTestError, match="disabled"):
        app.button(key="predict_crop").click()
    assert post.call_count == 1
    app.number_input(key=f"crop_input_{field}").set_value(original).run()
    assert not app.exception and not app.error
    assert not highlighted_fields(app)
    assert not app.button(key="predict_crop").disabled
    assert not parse_bars(app).widths
    assert post.call_count == 1
    app.run()
    assert not parse_bars(app).widths
    app.button(key="predict_crop").click().run()
    assert post.call_count == 2
    assert len(parse_bars(app).widths) == 3


def test_multiple_invalid_inputs_report_all_errors(api):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    app.number_input(key="crop_input_N").set_value(90000.)
    app.number_input(key="crop_input_P").set_value(-1.)
    app.number_input(key="crop_input_K").set_value(206.).run()
    assert not app.exception
    assert_validation_warning(app)
    assert highlighted_fields(app) == {"N", "P", "K"}
    for label in ("Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"):
        assert any(f"{label} phải nằm trong khoảng" in element.value for element in app.markdown)
    assert not parse_bars(app).widths
    post.assert_not_called()


def test_invalid_input_invalidates_even_matching_legacy_payload(api):
    _, post = api
    app = AppTest.from_file(str(APP_PATH))
    payload = dict(N=90000., P=42., K=43., temperature=20., humidity=80., ph=6.5, rainfall=200.)
    for field, value in payload.items():
        app.session_state[f"crop_input_{field}"] = value
    app.session_state["last_prediction"] = {
        "explicit_submit": True, "payload": payload,
        "crop": "rice", "result": {"recommended_crop": "rice"},
    }
    app.run()
    assert not app.exception
    assert_validation_warning(app)
    assert "last_prediction" not in app.session_state
    assert not parse_bars(app).widths
    post.assert_not_called()


@pytest.fixture(scope="module")
def input_validator():
    # Load the actual pure validator without executing Streamlit's UI/network
    # statements. Its constants and function body come directly from app.py.
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "validate_inputs"
    ) or (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"INPUT_LIMITS", "INPUT_LABELS"}
            for target in node.targets
        )
    )]
    namespace = {"math": math, "Real": Real, "INPUT_LIMITS": DEMO_LIMITS, "format_bound": format_bound}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace["validate_inputs"]


@pytest.mark.parametrize("missing", DEMO_LIMITS)
def test_validator_rejects_each_missing_field(input_validator, missing):
    values = dict(N=90., P=42., K=43., temperature=20., humidity=80., ph=6.5, rainfall=200.)
    del values[missing]
    assert set(input_validator(values)) == {missing}


@pytest.mark.parametrize("invalid", [None, "90", "abc", True, float("nan"), float("inf"), float("-inf"), 10**400])
def test_validator_rejects_non_numeric_or_nonfinite_values(input_validator, invalid):
    values = dict(N=invalid, P=42., K=43., temperature=20., humidity=80., ph=6.5, rainfall=200.)
    assert set(input_validator(values)) == {"N"}


def test_field_error_highlights_are_scoped_reversible_and_cleared_by_random(api):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    initial_input_layout = [element.type for element in app.main.columns[0]]
    assert not highlighted_fields(app)
    assert sum('class="crop-field-error"' in element.value for element in app.markdown) == 7
    app.button(key="predict_crop").click().run()
    assert len(parse_bars(app).widths) == 3

    app.number_input(key="crop_input_ph").set_value(100.).run()
    assert not app.exception
    assert highlighted_fields(app) == {"ph"}
    assert any("pH phải nằm trong khoảng 3.50–9.94." in element.value for element in app.markdown)
    assert not parse_bars(app).widths
    assert [element.type for element in app.main.columns[0]] == initial_input_layout
    app.number_input(key="crop_input_ph").set_value(6.5).run()
    assert not highlighted_fields(app)
    app.number_input(key="crop_input_N").set_value(90000.).run()
    assert highlighted_fields(app) == {"N"}
    app.number_input(key="crop_input_ph").set_value(100.).run()
    assert highlighted_fields(app) == {"N", "ph"}
    for element in app.markdown:
        if ".st-key-field_" in element.value:
            assert "border-color: #dc2626" in element.value
            assert "box-shadow: inset 0 0 0 1px #dc2626" in element.value
            assert "button" not in element.value
            assert "nth-child" not in element.value
    app.number_input(key="crop_input_N").set_value(90.)
    app.number_input(key="crop_input_ph").set_value(6.5).run()
    assert not highlighted_fields(app)
    assert not parse_bars(app).widths
    assert post.call_count == 1
    app.number_input(key="crop_input_N").set_value(90000.)
    app.number_input(key="crop_input_ph").set_value(100.).run()
    app.button(key="load_sample").click().run()
    assert not app.exception and not app.error
    assert not highlighted_fields(app)
    assert all(element.value is not None for element in app.main.number_input)
    assert post.call_count == 1
    assert [element.type for element in app.main.columns[0]] == initial_input_layout
    app.button(key="predict_crop").click().run()
    assert not app.exception
    assert post.call_count == 2
    assert len(parse_bars(app).widths) == 3


@pytest.mark.parametrize("fields", [("ph",), ("N", "ph"), tuple(DEMO_LIMITS)])
def test_one_warning_for_one_multiple_or_all_invalid_fields(api, fields):
    _, post = api
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.warning
    app.button(key="predict_crop").click().run()
    assert len(parse_bars(app).widths) == 3
    for field in fields:
        value = {"N": 90000., "ph": 100.}.get(field, DEMO_LIMITS[field][1] + 1)
        app.number_input(key=f"crop_input_{field}").set_value(value)
    app.run()
    assert not app.exception
    assert_validation_warning(app)
    assert highlighted_fields(app) == set(fields)
    assert not parse_bars(app).widths
    assert app.button(key="predict_crop").disabled
    assert post.call_count == 1
    app.button(key="load_sample").click().run()
    assert not app.exception and not app.warning and not app.error
    assert not highlighted_fields(app)
    assert not parse_bars(app).widths
    assert post.call_count == 1
    app.button(key="predict_crop").click().run()
    assert len(parse_bars(app).widths) == 3
    assert post.call_count == 2
