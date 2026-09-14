import numpy as np
import pandas as pd
import pytest

from ml.model.evaluate import evaluate_predictions, evaluate_xgboost, run_baselines


def test_metrics_keep_absent_classes_and_are_structured():
    metrics = evaluate_predictions(pd.Series([0, 0]), pd.Series([0, 0]))
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"] == [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    assert metrics["per_class"]["DOWN"]["support"] == 0
    assert metrics["per_class"]["FLAT"]["f1-score"] == 1
    assert metrics["per_class"]["UP"]["recall"] == 0
    assert metrics["log_loss"] is None


def test_xgboost_decodes_classes_and_probability_column_order():
    class Model:
        classes_ = np.array([2, 0, 1])

        def predict(self, X):
            return np.array([0, 1, 2])

        def predict_proba(self, X):
            return np.array([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.7, 0.1, 0.2]])

    metrics = evaluate_xgboost(Model(), np.zeros((3, 12)), np.array([0, 1, 2]))
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["log_loss"] == pytest.approx(-np.log([0.8, 0.6, 0.7]).mean())
    assert metrics["macro_f1"] == 1


def test_momentum_uses_contiguous_history_despite_overlap_future_and_row_order():
    dates = pd.bdate_range("2024-01-01", periods=9)
    history = pd.DataFrame({"ticker": "INFY", "date": dates, "close": [50, 50, 50, 100, 110, 120, 90, 100, 130]})
    other = history.assign(ticker="TCS", close=200 / history["close"])
    history = pd.concat([history, other], ignore_index=True)
    evaluation = history.groupby("ticker").tail(3).copy()
    evaluation["target"] = [-1, -1, 1, 1, 1, -1]
    evaluation = evaluation.sample(frac=1, random_state=42)
    history = pd.concat([history, pd.DataFrame({"ticker": ["INFY", "TCS"], "date": [dates[-1] + pd.Timedelta(days=1)] * 2, "close": [1, 1]})])
    training = pd.DataFrame({"target": [-1, -1, 0, 1]})
    metrics = run_baselines(training, evaluation, history)
    assert metrics["momentum"]["accuracy"] == 1
    assert metrics["momentum"]["confusion_matrix"] == [[3, 0, 0], [0, 0, 0], [0, 0, 3]]
    assert metrics["majority"]["accuracy"] == 0.5


def test_momentum_rejects_missing_history_instead_of_inventing_neutral():
    frame = pd.DataFrame({"ticker": ["INFY"], "date": pd.to_datetime(["2024-01-01"]), "close": [100], "target": [0]})
    with pytest.raises(ValueError, match="history"):
        run_baselines(frame, frame, frame.iloc[:0])


@pytest.mark.parametrize("lookback", [0, -1])
def test_momentum_requires_positive_past_lookback(lookback):
    frame = pd.DataFrame({"ticker": ["INFY"] * 3, "date": pd.bdate_range("2024-01-01", periods=3), "close": [100, 110, 90], "target": [-1, 0, 1]})
    with pytest.raises(ValueError, match="lookback_days"):
        run_baselines(frame, frame.iloc[[1]], frame, lookback_days=lookback)
