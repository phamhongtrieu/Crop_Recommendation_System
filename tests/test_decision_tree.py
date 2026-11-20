"""Validate the trained Decision Tree, its holdout results and shared report."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid

from models.common import read_split
from models.reporting import SUMMARY_COLUMNS, TABLE_NAMES


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "decision_tree"


@pytest.fixture(scope="module")
def artifacts():
    model = joblib.load(MODEL_DIR / "model.joblib")
    artifact = joblib.load(MODEL_DIR / "training_results.joblib")
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    evaluation = json.loads((MODEL_DIR / "evaluation_results.json").read_text(encoding="utf-8"))
    return model, artifact, metadata, evaluation


def test_requested_grid_was_evaluated(artifacts):
    _, artifact, _, evaluation = artifacts
    search = artifact["search"]
    expected = list(ParameterGrid({
        "criterion": ["gini", "entropy"],
        "max_depth": [8, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 3],
        "ccp_alpha": [0.0],
    }))
    assert len(expected) == 16
    assert search.cv_results_["params"] == expected
    assert search.n_splits_ == 5
    assert search.refit == "f1_macro"
    assert search.best_score_ == pytest.approx(search.cv_results_["mean_test_f1_macro"].max())
    for fold in range(5):
        for metric in ("accuracy", "f1_macro"):
            scores = search.cv_results_[f"split{fold}_test_{metric}"]
            assert scores.shape == (16,)
            assert np.isfinite(scores).all()
    selected = [row for row in evaluation["tables"]["GridSearch"] if row["Selected"]]
    assert len(selected) == 1
    assert selected[0]["Candidate"] == search.best_index_ + 1
    assert json.loads(selected[0]["Parameters"]) == search.best_params_


def test_saved_tree_is_refit_on_training_data(artifacts):
    model, artifact, metadata, _ = artifacts
    X_train, y_train = read_split("train")
    refitted = clone(model).fit(X_train, y_train)
    assert model.random_state == 42
    assert model.tree_.n_node_samples[0] == len(X_train)
    assert metadata["features"] == X_train.columns.tolist() == model.feature_names_in_.tolist()
    assert metadata["classes"] == model.classes_.tolist()
    assert metadata["parameters"] == artifact["search"].best_params_
    assert metadata["training_finished_at"] == artifact["finished_at"]
    for name in ("feature", "threshold", "value"):
        np.testing.assert_array_equal(getattr(model.tree_, name), getattr(refitted.tree_, name))
        np.testing.assert_array_equal(
            getattr(model.tree_, name), getattr(artifact["search"].best_estimator_.tree_, name)
        )


def test_evaluation_matches_holdout_predictions(artifacts):
    model, _, metadata, evaluation = artifacts
    X_test, y_test = read_split("test")
    predictions = model.predict(X_test)
    options = {"labels": model.classes_, "average": "macro", "zero_division": 0}
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, **options),
        "recall_macro": recall_score(y_test, predictions, **options),
        "f1_macro": f1_score(y_test, predictions, **options),
    }
    summary = evaluation["tables"]["Tong_hop"][0]
    assert list(summary) == SUMMARY_COLUMNS
    assert metadata["test_samples"] == len(X_test)
    for name, value in metrics.items():
        assert metadata["test_metrics"][name] == pytest.approx(value)
        assert summary[f"Test_{name}"] == pytest.approx(value)
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), len(model.classes_))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.classes_[probabilities.argmax(axis=1)], predictions)


def test_full_tree_visualizations_are_saved(artifacts):
    model, _, metadata, _ = artifacts
    assert metadata["tree"]["depth"] == model.get_depth()
    assert metadata["tree"]["leaves"] == model.get_n_leaves()
    assert metadata["tree"]["node_count"] == model.tree_.node_count
    png = MODEL_DIR / metadata["tree"]["visualization_png"]
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (MODEL_DIR / metadata["tree"]["visualization_svg"]).read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Decision Tree - best GridSearchCV model" in svg
    for node in range(model.tree_.node_count):
        assert f"node #{node} " in svg or f"node #{node}<" in svg


def test_shared_excel_contains_decision_tree_results(artifacts):
    _, _, _, evaluation = artifacts
    with pd.ExcelFile(MODEL_DIR.parent / "model_evaluation_results.xlsx") as workbook:
        assert workbook.sheet_names == ["Tong_hop", "Tieu_chi", "GridSearch"]
        for name in TABLE_NAMES:
            saved = pd.read_excel(workbook, sheet_name=name)
            saved = saved.loc[saved["Model"] == "Decision Tree"].reset_index(drop=True)
            expected = pd.DataFrame(evaluation["tables"][name])
            pd.testing.assert_frame_equal(saved, expected, check_dtype=False, atol=1e-12)
