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
        model_dir = PROJECT_DIR / "models" / "softmax"
        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "model_metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Chạy train.py và evaluate.ipynb trước khi test.")

        cls.model = joblib.load(model_path)
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cls.artifact = joblib.load(model_dir / "training_results.joblib")
        cls.search = cls.artifact["search"]
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

    def test_saved_model_matches_best_cv_model(self):
        artifact = self.artifact
        self.assertAlmostEqual(
            self.metadata["cv_f1_macro_mean"],
            self.search.cv_results_["mean_test_f1_macro"].max(),
        )
        self.assertEqual(self.metadata["training_finished_at"], artifact["finished_at"])
        self.assertEqual(self.metadata["parameters"], self.search.best_params_)
        original = self.search.best_estimator_
        np.testing.assert_array_equal(
            self.predictions, original.predict(self.X_test)
        )

    def test_scaler_is_fit_on_training_data(self):
        X_train = pd.read_csv(PROJECT_DIR / "data" / "processed" / "X_train.csv")
        scaler = self.model.named_steps["scaler"]
        self.assertEqual(scaler.n_samples_seen_, len(X_train))
        np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
        self.assertEqual(self.model.named_steps["model"].solver, "lbfgs")

    def test_excel_matches_summary_and_grid_search(self):
        from models.reporting import SUMMARY_COLUMNS, TABLE_NAMES

        result = json.loads(
            (PROJECT_DIR / "models" / "softmax" / "evaluation_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(result["tables"]), {"Tong_hop", "GridSearch"})
        self.assertEqual(len(result["tables"]["Tong_hop"]), 1)
        grid_results = self.search.cv_results_
        grid_rows = result["tables"]["GridSearch"]
        self.assertEqual(len(grid_rows), len(grid_results["params"]))
        for i, candidate in enumerate(grid_rows):
            self.assertEqual(candidate["Candidate"], i + 1)
            self.assertEqual(json.loads(candidate["Parameters"]), grid_results["params"][i])
            self.assertAlmostEqual(candidate["CV_accuracy_mean"], grid_results["mean_test_accuracy"][i])
            self.assertAlmostEqual(candidate["CV_f1_macro_mean"], grid_results["mean_test_f1_macro"][i])
            self.assertAlmostEqual(candidate["CV_f1_macro_std"], grid_results["std_test_f1_macro"][i])
            self.assertEqual(candidate["Rank_CV_F1_macro"], grid_results["rank_test_f1_macro"][i])
        with pd.ExcelFile(PROJECT_DIR / "models" / "model_evaluation_results.xlsx") as workbook:
            self.assertEqual(workbook.sheet_names, ["Tong_hop", "Tieu_chi", "GridSearch"])
            summary = pd.read_excel(workbook, sheet_name="Tong_hop")
            self.assertEqual(summary.columns.tolist(), SUMMARY_COLUMNS)
            for name in TABLE_NAMES:
                expected = pd.DataFrame(result["tables"][name])
                saved = pd.read_excel(workbook, sheet_name=name)
                saved = saved.loc[saved["Model"] == "Softmax"].reset_index(drop=True)
                pd.testing.assert_frame_equal(saved, expected, check_dtype=False, atol=1e-12)
        for row in result["tables"]["Tong_hop"]:
            self.assertEqual(row["Configuration"], "GridSearch")
            expected_f1 = [
                self.search.cv_results_[f"split{fold}_test_f1_macro"][self.search.best_index_]
                for fold in range(self.search.n_splits_)
            ]
            self.assertAlmostEqual(np.mean(expected_f1), row["CV_f1_macro_mean"])
            self.assertAlmostEqual(np.std(expected_f1, ddof=0), row["CV_f1_macro_std"])
            model = self.search.best_estimator_
            self.assertEqual(json.loads(row["Best_grid_parameters"]), self.search.best_params_)
            self.assertAlmostEqual(accuracy_score(self.y_test, model.predict(self.X_test)), row["Test_accuracy"])
        selected_candidates = [row for row in grid_rows if row["Selected"]]
        self.assertEqual(len(selected_candidates), 1)
        best_grid = selected_candidates[0]
        self.assertTrue(best_grid["Selected"])
        self.assertEqual(best_grid["Candidate"], self.search.best_index_ + 1)
        self.assertEqual(json.loads(best_grid["Parameters"]), self.search.best_params_)
        self.assertAlmostEqual(best_grid["CV_f1_macro_mean"], self.search.best_score_)

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
