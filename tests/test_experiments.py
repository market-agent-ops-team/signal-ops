import numpy as np
import pandas as pd

from ml.experiments.config import HORIZON_THRESHOLD_GRID, SEED, TUNING_CANDIDATE_COUNT
from ml.experiments.core import (
    ExperimentRecord,
    ablation_feature_sets,
    add_extended_features,
    add_nifty_context,
    build_tuning_candidates,
    compare_calibration,
    horizon_threshold_grid,
    reserve_final_holdout,
    walk_forward_folds,
)
from ml.features.indicators import add_technical_indicators
from ml.model.train import FEATURE_COLS


def _frame():
    dates = pd.bdate_range("2023-01-02", periods=60)
    rows = []
    for ticker in ["INFY", "TCS"]:
        close = 100 + np.arange(60) * 0.1
        rows.append(pd.DataFrame({
            "ticker": ticker,
            "date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10000,
        }))
    return pd.concat(rows, ignore_index=True)


def test_holdout_reserved_before_selection_and_is_untouched():
    df = _frame()
    develop, holdout = reserve_final_holdout(df, holdout_frac=0.1)
    assert develop["date"].max() < holdout["date"].min()
    assert len(holdout) > 0
    assert len(develop) + len(holdout) == len(df)


def test_walk_forward_is_chronological():
    dates = list(pd.bdate_range("2023-01-02", periods=12))
    folds = walk_forward_folds(dates, n_folds=3)
    assert len(folds) == 3
    for train, val in folds:
        assert max(train) < min(val)


def test_nifty_and_extended_features():
    base = add_technical_indicators(_frame().query("ticker == 'INFY'").copy()).dropna().reset_index(drop=True)
    nifty = pd.DataFrame({"date": base["date"], "close": base["close"] * 1.01 + 5})
    with_nifty = add_nifty_context(base, nifty)
    for col in ["nifty_return_1d", "relative_return_3d", "relative_return_5d"]:
        assert col in with_nifty.columns
    extended = add_extended_features(base.assign(ema_9=base["close"], ema_21=base["close"]))
    for col in ["return_10d", "volatility_10d", "ema_9_vs_21", "volume_change_1d", "high_low_range", "close_open_return"]:
        assert col in extended.columns


def test_ablations_cover_required_groups():
    sets = ablation_feature_sets(FEATURE_COLS + ["nifty_return_1d"])
    for key in ["all", "without_rsi", "without_volume", "without_bollinger", "without_returns", "without_market"]:
        assert key in sets
    assert "rsi_14" not in sets["without_rsi"]
    assert "rsi_14" in sets["all"]


def test_tuning_is_bounded_and_reproducible():
    first = build_tuning_candidates(seed=SEED)
    second = build_tuning_candidates(seed=SEED)
    assert len(first) == TUNING_CANDIDATE_COUNT == 12
    assert first == second
    assert len(horizon_threshold_grid()) == len(HORIZON_THRESHOLD_GRID) == 7


def test_calibration_compares_three_methods():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(42)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    estimator = LogisticRegression(max_iter=200)
    estimator.fit(X[:60], y[:60])
    result = compare_calibration(estimator, X[:60], y[:60], X[60:], y[60:])
    assert set(result) == {"none", "sigmoid", "isotonic"}
    assert result["none"] is not None


def test_record_round_trip():
    record = ExperimentRecord(name="v1", dataset="2020-2024", features=list(FEATURE_COLS), horizon_days=3, flat_threshold=0.0075, params={"seed": SEED}, accuracy=0.38, macro_f1=0.37, log_loss=1.09, status="done")
    assert ExperimentRecord.from_dict(record.to_dict()) == record
