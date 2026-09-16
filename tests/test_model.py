import hashlib

import numpy as np
import pytest

from ml.features.indicators import generate_inference_features
from ml.model import predict
from ml.model.train import FEATURE_COLS


@pytest.mark.parametrize("date", ["2026/09/11", "2026-9-11", "20260911", "2026-09-11T12:00:00", "2026-02-30", "", "NaT", " 2026-09-11", "2026-09-11\n", None, 20260911])
def test_strict_date_rejected_before_fetch(date, monkeypatch):
    def fetch(*args):
        pytest.fail("Invalid date reached market data fetch")

    monkeypatch.setattr(predict, "pick_OHLCV", fetch)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        predict.predict_trend("INFY", date)


def test_real_artifacts_dimensions_and_fixture_inference(prices, monkeypatch):
    model, scaler, names = predict._load_artifacts()
    assert model.n_features_in_ == scaler.n_features_in_ == len(names) == 12
    assert names == FEATURE_COLS
    np.testing.assert_array_equal(scaler.feature_names_in_, names)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    target = prices.iloc[-6]["date"]
    monkeypatch.setattr(predict, "pick_OHLCV", lambda *args: prices.copy())
    result = predict.predict_trend("INFY", target.strftime("%Y-%m-%d"))
    features = generate_inference_features(prices.iloc[:-5])
    probabilities = model.predict_proba(scaler.transform(features.iloc[[-1]][names]))[0]
    assert set(result) == {"ml_trend", "ml_confidence"}
    assert result["ml_trend"] == ["bearish", "neutral", "bullish"][int(np.argmax(probabilities))]
    assert result["ml_confidence"] == pytest.approx(float(probabilities.max()))
    assert 0 <= result["ml_confidence"] <= 1


@pytest.mark.parametrize("filename", ["xgb_model.joblib", "scaler.joblib", "feature_names.joblib"])
def test_missing_artifacts_are_clear(filename, tmp_path, monkeypatch):
    for name in ("xgb_model.joblib", "scaler.joblib", "feature_names.joblib"):
        if name != filename:
            (tmp_path / name).touch()
    predict._load_artifacts.cache_clear()
    monkeypatch.setattr(predict, "ARTIFACT_DIR", tmp_path)
    try:
        with pytest.raises(FileNotFoundError, match=filename):
            predict._load_artifacts()
    finally:
        predict._load_artifacts.cache_clear()


def test_empty_ticker_is_rejected():
    with pytest.raises(ValueError, match="ticker"):
        predict.predict_trend("  ", "2023-11-01")


def test_data_fetch_error_propagates(monkeypatch):
    def fetch(*args):
        raise ValueError("No data returned for INVALID.NS")

    monkeypatch.setattr(predict, "pick_OHLCV", fetch)
    with pytest.raises(ValueError, match="INVALID.NS"):
        predict.predict_trend("INVALID", "2023-11-01")


def test_empty_history_is_rejected(prices, monkeypatch):
    monkeypatch.setattr(predict, "pick_OHLCV", lambda *args: prices.iloc[:0].copy())
    with pytest.raises(ValueError, match="No market data"):
        predict.predict_trend("INFY", "2023-11-01")


def test_pipeline_trains_once_evaluates_and_saves_same_real_model(prices, monkeypatch, tmp_path):
    import json
    import runpy
    import joblib
    from pathlib import Path
    from ml.dataProcessing import data_fetch
    from ml.dataProcessing.data_preprocessing import chronological_split, purge_split_boundary
    from ml.features.indicators import generate_feature_dataset
    from ml.model import evaluate, train

    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in predict.ARTIFACT_DIR.glob("*.joblib")}
    train_df, val_df, test_df = chronological_split(generate_feature_dataset(prices))
    train_df, val_df = purge_split_boundary(train_df), purge_split_boundary(val_df)
    expected_train, expected_val, _, _ = train.prepare_tensors(train_df, val_df, test_df)
    def fetch(tickers, start, end):
        assert start == "2020-01-01"
        return prices.copy(), [{"ticker": "INFY", "ok": True}]

    monkeypatch.setattr(data_fetch, "fetch_grp", fetch)
    trained, evaluated, saved = [], [], []
    real_train, real_evaluate, real_save = train.train_xgboost, evaluate.evaluate_xgboost, train.save_artifacts

    def fit(*args):
        for actual, expected in zip(args, (*expected_train, *expected_val), strict=True):
            np.testing.assert_array_equal(actual, expected)
        model = real_train(*args)
        trained.append(model)
        return model

    def assess(model, *args):
        evaluated.append(model)
        return real_evaluate(model, *args)

    def save(model, scaler, **kwargs):
        assert Path(kwargs.get("output_dir", "ml/saved_models")).resolve() == predict.ARTIFACT_DIR.parent / "candidate_models"
        saved.append(model)
        kwargs["output_dir"] = str(tmp_path)
        return real_save(model, scaler, **kwargs)

    monkeypatch.setattr(train, "train_xgboost", fit)
    monkeypatch.setattr(evaluate, "evaluate_xgboost", assess)
    monkeypatch.setattr(train, "save_artifacts", save)
    runpy.run_module("ml.run_pipeline", run_name="__main__")
    assert len(trained) == 1
    assert evaluated == [trained[0], trained[0]]
    assert saved == trained
    reloaded = joblib.load(tmp_path / "xgb_model.joblib")
    assert reloaded.get_booster().save_raw() == trained[0].get_booster().save_raw()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(metrics) == {"validation", "test"}
    assert metrics["test"]["xgboost"]["log_loss"] >= 0
    for name, frame in (("validation", val_df), ("test", test_df)):
        momentum = np.select([frame["return_3d"] > 0.0075, frame["return_3d"] < -0.0075], [1, -1], default=0)
        expected_confusion = [[int(((frame["target"] == truth) & (momentum == prediction)).sum()) for prediction in (-1, 0, 1)] for truth in (-1, 0, 1)]
        assert metrics[name]["momentum"]["confusion_matrix"] == expected_confusion
        for result in metrics[name].values():
            assert sum(sum(row) for row in result["confusion_matrix"]) == len(frame)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in predict.ARTIFACT_DIR.glob("*.joblib")} == before


@pytest.mark.parametrize("encoded,trend", [(0, "bearish"), (1, "neutral"), (2, "bullish")])
def test_prediction_uses_model_class_order(encoded, trend, prices, monkeypatch):
    _, scaler, names = predict._load_artifacts()

    class Model:
        classes_ = np.array([2, 0, 1])

        def predict_proba(self, X):
            result = np.full((1, 3), 0.1)
            result[0, list(self.classes_).index(encoded)] = 0.8
            return result

    monkeypatch.setattr(predict, "_load_artifacts", lambda: (Model(), scaler, names))
    monkeypatch.setattr(predict, "pick_OHLCV", lambda *args: prices.copy())
    assert predict.predict_trend("INFY", "2023-12-02") == {"ml_trend": trend, "ml_confidence": 0.8}


def test_v1_history_origin_and_training_parity_with_range_respecting_fetch(monkeypatch):
    import pandas as pd
    from ml.features.indicators import build_indicators_for_ticker

    dates = pd.bdate_range("2019-01-01", "2024-01-01")
    day = np.arange(len(dates))
    close = 100 + day * 0.02 + 3 * np.sin(day / 5)
    history = pd.DataFrame({"ticker": "INFY.NS", "date": dates, "open": close - 0.2, "high": close + 1, "low": close - 1, "close": close, "volume": 10000 + day % 17 * 300})
    target = pd.Timestamp("2023-10-06")
    training = build_indicators_for_ticker(history[history["date"] >= "2020-01-01"])
    expected = training.loc[training["date"] == target, FEATURE_COLS].reset_index(drop=True)
    requested, observed = [], []
    model, scaler, names = predict._load_artifacts()
    original_transform = scaler.transform

    def fetch(ticker, start, end):
        requested.append((ticker, start, end))
        return history[(history["date"] >= start) & (history["date"] < end)].copy()

    def transform(frame):
        observed.append(frame.reset_index(drop=True))
        return original_transform(frame)

    monkeypatch.setattr(predict, "pick_OHLCV", fetch)
    monkeypatch.setattr(scaler, "transform", transform)
    result = predict.predict_trend("INFY", "2023-10-06")
    assert requested == [("INFY", "2020-01-01", "2023-10-07")]
    pd.testing.assert_frame_equal(observed[0], expected)
    probabilities = model.predict_proba(original_transform(expected[names]))[0]
    assert result["ml_confidence"] == pytest.approx(float(probabilities.max()))
