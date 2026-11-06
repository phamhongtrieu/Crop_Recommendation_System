"""Kiểm tra model đã xuất cho backend: python -m src.test_model.

Cần chạy python -m src.export_model trước. Test không huấn luyện lại model.
"""

import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TestCropModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_dir = PROJECT_DIR / "models"
        model_path = model_dir / "crop_model.joblib"
        metadata_path = model_dir / "model_metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Chạy python -m src.export_model trước khi test.")

        cls.model = joblib.load(model_path)
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_dir = PROJECT_DIR / "data" / "processed"
        cls.X_test = pd.read_csv(data_dir / "X_test.csv")
        cls.y_test = pd.read_csv(data_dir / "y_test.csv")["label"]
        cls.predictions = cls.model.predict(cls.X_test)

    def test_feature_and_label_contract(self):
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.assertEqual(self.metadata["features"], features)
        self.assertEqual(self.X_test.columns.tolist(), features)
        self.assertEqual(self.model.classes_.tolist(), self.metadata["classes"])
        self.assertEqual(set(self.y_test), set(self.metadata["classes"]))
        self.assertEqual(len(self.predictions), len(self.y_test))
        self.assertTrue(set(self.predictions).issubset(self.metadata["classes"]))

    def test_export_matches_best_cv_model(self):
        artifact = joblib.load(
            PROJECT_DIR / "notebooks" / "model_training_results.joblib"
        )
        summary = artifact["training_summary"]
        candidates = summary.loc[summary["Configuration"] == "GridSearch"]
        self.assertAlmostEqual(
            self.metadata["cv_f1_macro_mean"], candidates["CV_f1_macro_mean"].max()
        )
        self.assertEqual(self.metadata["training_run_id"], artifact["run_id"])
        original = artifact["fitted_models"][self.metadata["model_name"]]["GridSearch"]
        np.testing.assert_array_equal(
            self.predictions, original.predict(self.X_test)
        )

    def test_holdout_metrics(self):
        options = {
            "labels": self.metadata["classes"], "average": "macro", "zero_division": 0
        }
        actual = {
            "accuracy": accuracy_score(self.y_test, self.predictions),
            "precision_macro": precision_score(self.y_test, self.predictions, **options),
            "recall_macro": recall_score(self.y_test, self.predictions, **options),
            "f1_macro": f1_score(self.y_test, self.predictions, **options),
        }
        self.assertEqual(len(self.y_test), self.metadata["test_samples"])
        for metric, value in actual.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(value, self.metadata["test_metrics"][metric])
        self.assertGreaterEqual(actual["f1_macro"], 0.95, "F1-macro test dưới 95%.")
        print(
            f"\nTest Accuracy: {actual['accuracy']:.4%}; "
            f"F1-macro: {actual['f1_macro']:.4%} "
            f"({int((self.predictions == self.y_test).sum())}/{len(self.y_test)} mẫu đúng)"
        )

    def test_backend_json_sample(self):
        # Dữ liệu gốc như JSON backend nhận; chủ động đảo thứ tự khóa đầu vào.
        payload = json.loads(
            '{"rainfall": 202.9355362, "ph": 6.502985292, '
            '"humidity": 82.00274423, "temperature": 20.87974371, '
            '"K": 43, "P": 42, "N": 90}'
        )
        sample = pd.DataFrame([payload], columns=self.metadata["features"])
        self.assertEqual(self.model.predict(sample).tolist(), ["rice"])

    def test_prediction_probabilities(self):
        probabilities = self.model.predict_proba(self.X_test)
        self.assertEqual(
            probabilities.shape, (len(self.X_test), len(self.metadata["classes"]))
        )
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_array_equal(
            self.model.classes_[probabilities.argmax(axis=1)], self.predictions
        )

    def test_batch_matches_single_prediction(self):
        for index in (0, 1, len(self.X_test) - 1):
            with self.subTest(index=index):
                single = self.model.predict(self.X_test.iloc[[index]])[0]
                self.assertEqual(single, self.predictions[index])


if __name__ == "__main__":
    unittest.main(verbosity=2)
