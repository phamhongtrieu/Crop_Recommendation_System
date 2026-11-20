"""Tổng hợp kết quả đã đánh giá của các thuật toán: python -m models.reporting."""

import json
from pathlib import Path
import sys

import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Thêm thư mục gốc để chạy trực tiếp file bằng nút Run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.common import MODELS_DIR

CRITERIA = [
    ("Model", "Tên thuật toán"),
    ("Configuration", "Cấu hình tốt nhất được GridSearch chọn theo CV F1-macro"),
    ("CV_accuracy_mean", "Accuracy trung bình trên validation"),
    ("CV_f1_macro_mean", "F1-macro validation trung bình; tiêu chí chọn tham số"),
    ("CV_f1_macro_std", "Độ lệch chuẩn F1-macro giữa các fold; ddof=0"),
    ("Test_accuracy", "Accuracy trên tập test độc lập"),
    ("Test_precision_macro", "Precision trung bình không trọng số qua các lớp"),
    ("Test_recall_macro", "Recall trung bình không trọng số qua các lớp"),
    ("Test_f1_macro", "F1 trung bình không trọng số qua các lớp"),
    ("Train_f1_macro", "F1-macro trên toàn bộ train sau khi fit"),
    ("CV_train_validation_gap_F1", "F1 train trung bình trong CV trừ F1 validation trung bình"),
    ("Preprocessing", "Tiền xử lý bên trong Pipeline; fit riêng từng fold"),
    ("Best_grid_parameters", "Siêu tham số tốt nhất được GridSearch chọn"),
]
SUMMARY_COLUMNS = [name for name, _ in CRITERIA]
TABLE_NAMES = ("Tong_hop", "GridSearch")


def export_report(models_dir: Path = MODELS_DIR) -> Path:
    """Chỉ tổng hợp thuật toán có evaluation_results.json, bỏ qua thư mục trống."""
    models_dir = Path(models_dir)
    frames = {name: [] for name in TABLE_NAMES}
    for result_path in sorted(models_dir.glob("*/evaluation_results.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for name in TABLE_NAMES:
            frames[name].append(pd.DataFrame(result["tables"][name]))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    tables["Tong_hop"] = tables["Tong_hop"][SUMMARY_COLUMNS]
    tables = {
        "Tong_hop": tables.pop("Tong_hop"),
        "Tieu_chi": pd.DataFrame(CRITERIA, columns=["Tiêu chí", "Ý nghĩa"]),
        **tables,
    }
    report_path = models_dir / "model_evaluation_results.xlsx"
    temporary_path = report_path.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            sheet.row_dimensions[1].height = 36
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="174B46")
                cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for index, column in enumerate(frame.columns, 1):
                letter = get_column_letter(index)
                length = max([len(str(column))] + frame[column].head(100).fillna("").astype(str).str.len().tolist())
                sheet.column_dimensions[letter].width = min(65, max(15, length + 2))
                is_metric = (
                    column.startswith(("CV_accuracy", "CV_f1", "Test_accuracy", "Test_precision",
                                       "Test_recall", "Test_f1", "Train_f1"))
                    or column == "CV_train_validation_gap_F1"
                )
                for cells in sheet.iter_rows(min_row=2, min_col=index, max_col=index):
                    cell = cells[0]
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    if cell.row % 2 == 0:
                        cell.fill = PatternFill("solid", fgColor="F0F6F4")
                    if is_metric and isinstance(cell.value, (float, int)):
                        cell.number_format = "0.00%"
                if column in {"CV_f1_macro_mean", "Test_f1_macro"} and sheet.max_row > 1:
                    sheet.conditional_formatting.add(
                        f"{letter}2:{letter}{sheet.max_row}",
                        ColorScaleRule(start_type="min", start_color="FDE4DB",
                                       end_type="max", end_color="B7DDCD"),
                    )
        writer.book.properties.title = "Kết quả đánh giá mô hình khuyến nghị cây trồng"
    temporary_path.replace(report_path)
    return report_path


if __name__ == "__main__":
    print(export_report())
