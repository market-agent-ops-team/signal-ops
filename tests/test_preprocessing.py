import numpy as np
import pandas as pd
import pytest

from ml.dataProcessing.data_preprocessing import chronological_split, purge_split_boundary
from ml.model.train import FEATURE_COLS, prepare_tensors


@pytest.mark.parametrize("length,horizon,remaining", [(2, 3, 0), (3, 3, 0), (4, 3, 1), (4, 0, 4), (0, 3, 0)])
def test_purge_short_zero_and_empty_boundaries(prices, length, horizon, remaining):
    result = purge_split_boundary(prices.iloc[:length], horizon)
    assert len(result) == remaining
    assert list(result.columns) == list(prices.columns)
    if remaining:
        assert result.iloc[-1]["date"] == prices.iloc[remaining - 1]["date"]


def test_negative_purge_is_rejected(prices):
    with pytest.raises(ValueError, match="horizon"):
        purge_split_boundary(prices, -1)


def test_chronology_and_label_endpoints_per_ticker(prices):
    frame = pd.concat([prices, prices.assign(ticker="TCS.NS")]).sample(frac=1, random_state=42)
    train, val, test = chronological_split(frame)
    purged_train = purge_split_boundary(train)
    purged_val = purge_split_boundary(val)
    for ticker in frame["ticker"].unique():
        dates = prices["date"].tolist()
        for earlier, later in [(purged_train, val), (purged_val, test)]:
            last = earlier.loc[earlier["ticker"] == ticker, "date"].max()
            first = later.loc[later["ticker"] == ticker, "date"].min()
            assert dates[dates.index(last) + 3] < first


def test_scaler_is_fit_only_on_training_and_targets_are_encoded():
    def frame(values):
        result = pd.DataFrame({name: values for name in FEATURE_COLS})
        result["target"] = [-1, 0, 1]
        return result

    train, val, test = frame([1.0, 2.0, 3.0]), frame([100.0, 200.0, 300.0]), frame([-300.0, -200.0, -100.0])
    training, validation, testing, scaler = prepare_tensors(train, val, test)
    assert scaler.mean_ is not None
    assert scaler.var_ is not None
    np.testing.assert_allclose(scaler.mean_, 2.0)
    np.testing.assert_allclose(scaler.var_, 2 / 3)
    assert scaler.n_samples_seen_ == 3
    np.testing.assert_allclose(training[0].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(validation[0][0], 98 / np.sqrt(2 / 3))
    np.testing.assert_allclose(testing[0][0], -302 / np.sqrt(2 / 3))
    for _, labels in (training, validation, testing):
        np.testing.assert_array_equal(labels, [0, 1, 2])


@pytest.mark.parametrize("missing_sessions", [False, True])
def test_pooled_date_cutoffs_and_label_purge_with_unequal_histories(prices, missing_sessions):
    from ml.features.indicators import generate_feature_dataset

    groups = [prices.copy(), prices.iloc[:150].assign(ticker="EARLY.NS"), prices.iloc[100:].assign(ticker="LATE.NS")]
    if missing_sessions:
        groups[1] = groups[1].iloc[::2].copy()
        groups[2] = groups[2].drop(groups[2].index[::4]).copy()
    for group in groups:
        group["label_end"] = group["date"].shift(-3)
    features = generate_feature_dataset(pd.concat(groups, ignore_index=True))
    train, val, test = chronological_split(features.sample(frac=1, random_state=42))
    assert train["date"].max() < val["date"].min() < test["date"].min()
    assert val["date"].max() < test["date"].min()
    assert val["date"].min() == prices.iloc[180]["date"]
    assert test["date"].min() == prices.iloc[217]["date"]
    assert len(train) + len(val) + len(test) == len(features)
    for split, cutoff in ((train, val["date"].min()), (val, test["date"].min())):
        purged = purge_split_boundary(split, horizon_days=3)
        assert (purged["label_end"] < cutoff).all()
        for ticker, group in split.groupby("ticker"):
            kept = purged[purged["ticker"] == ticker]
            assert len(kept) == max(0, len(group) - 3)
