
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.reporting import export_report


report_path = export_report(Path(__file__).resolve().parents[1])
print(f"Đã xuất Excel: {report_path}")
