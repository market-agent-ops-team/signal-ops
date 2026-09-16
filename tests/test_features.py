import numpy as np
import pandas as pd
import pytest

from ml.features import indicators
from ml.model.train import FEATURE_COLS


def test_features_are_causal_and_keep_latest_row(prices):
    prefix = indicators.generate_inference_features(prices.iloc[:160])
    changed = prices.copy()
    changed.loc[160:, ["close", "volume"]] *= 7
    complete = indicators.generate_inference_features(changed)
    pd.testing.assert_frame_equal(prefix, complete.iloc[:len(prefix)])
    assert prefix.iloc[-1]["date"] == prices.iloc[159]["date"]
    assert not {"target", "target_return"} & set(prefix.columns)
    assert np.isfinite(prefix[FEATURE_COLS].to_numpy()).all()


def test_training_matches_inference_and_forward_labels(prices):
    training = indicators.build_indicators_for_ticker(prices)
    inference = indicators.generate_inference_features(prices).set_index("date")
    pd.testing.assert_frame_equal(
        training.set_index("date")[FEATURE_COLS],
        inference.loc[training["date"], FEATURE_COLS],
    )
    assert training.iloc[-1]["date"] == prices.iloc[-4]["date"]
    expected_return = prices.iloc[-1]["close"] / prices.iloc[-4]["close"] - 1
    assert training.iloc[-1]["target_return"] == pytest.approx(expected_return)


def test_training_uses_shared_feature_values(prices, monkeypatch):
    original = indicators.add_technical_indicators

    def adjusted(frame):
        result = original(frame)
        result[FEATURE_COLS] += 0.125
        return result

    monkeypatch.setattr(indicators, "add_technical_indicators", adjusted)
    training = indicators.build_indicators_for_ticker(prices).set_index("date")
    inference = indicators.generate_inference_features(prices).set_index("date")
    pd.testing.assert_frame_equal(training[FEATURE_COLS], inference.loc[training.index, FEATURE_COLS])


def test_multiple_tickers_do_not_share_indicator_history(prices):
    other = prices.assign(ticker="TCS.NS", close=prices["close"] * 3)
    combined = indicators.generate_feature_dataset(pd.concat([other, prices]).sample(frac=1, random_state=42))
    expected = indicators.build_indicators_for_ticker(other)
    pd.testing.assert_frame_equal(combined[combined["ticker"] == "TCS.NS"].reset_index(drop=True), expected)


def test_insufficient_inference_history(prices):
    with pytest.raises(ValueError, match="60"):
        indicators.generate_inference_features(prices.iloc[:59])
