"""Verify the exported summary keeps its data table and model-selection notes."""

import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import load_workbook
import pandas as pd
import pytest

from models.reporting import SUMMARY_COLUMNS, export_report

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def report_models():
    with TemporaryDirectory(prefix="report-", dir=PROJECT_DIR / "tests") as directory:
        destination = Path(directory).resolve()
        assert destination.is_relative_to(PROJECT_DIR / "tests")
        for source in sorted((PROJECT_DIR / "models").glob("*/evaluation_results.json")):
            model_dir = destination / source.parent.name
            model_dir.mkdir()
            (model_dir / source.name).write_bytes(source.read_bytes())
        yield destination


def test_report_preserves_table_and_explains_random_forest_selection(report_models):
    rows = []
    for path in sorted(report_models.glob("*/evaluation_results.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["tables"]["Tong_hop"])
    path = export_report(report_models)
    saved = pd.read_excel(path, sheet_name="Tong_hop", nrows=len(rows))
    pd.testing.assert_frame_equal(saved, pd.DataFrame(rows)[SUMMARY_COLUMNS], check_dtype=False, atol=1e-12)
    forest = next(row for row in rows if row["Model"] == "Random Forest")
    with closing(load_workbook(path)) as workbook:
        sheet = workbook["Tong_hop"]
        assert sheet.auto_filter.ref == f"A1:M{len(rows) + 1}"
        assert sheet.cell(len(rows) + 3, 1).value == "Mô hình được chọn cho dự đoán: Random Forest"
        notes = {sheet.cell(row, 1).value: sheet.cell(row, 4).value for row in range(len(rows) + 4, sheet.max_row + 1)}
        assert f"{forest['CV_f1_macro_mean']:.2%}".replace(".", ",") in notes["Kết quả CV"]
        assert f"{forest['CV_train_validation_gap_F1'] * 100:.2f}".replace(".", ",") in notes["Khoảng cách train/validation"]
        assert "predict_proba" in notes["Phù hợp với ứng dụng"]
        assert "Feature Importance" in notes["Phù hợp với ứng dụng"]


def test_selection_notes_refresh_when_evaluation_results_change(report_models):
    result_path = report_models / "random_forest/evaluation_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["tables"]["Tong_hop"][0]["CV_f1_macro_mean"] = 0.875
    result_path.write_text(json.dumps(result), encoding="utf-8")
    path = export_report(report_models)
    with closing(load_workbook(path)) as workbook:
        sheet = workbook["Tong_hop"]
        cv_note = next(sheet.cell(row, 4).value for row in range(1, sheet.max_row + 1)
                       if sheet.cell(row, 1).value == "Kết quả CV")
        assert "87,50%" in cv_note
        assert "xếp hạng 5/5" in cv_note
