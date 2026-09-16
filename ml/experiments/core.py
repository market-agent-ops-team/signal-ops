import dataclasses
import math
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from ml.experiments.config import (
    CALIBRATION_METHODS,
    HORIZON_THRESHOLD_GRID,
    SEED,
    TUNING_CANDIDATE_COUNT,
)
from ml.model.train import FEATURE_COLS


@dataclasses.dataclass
class ExperimentRecord:
    name: str
    dataset: str
    features: list[str]
    horizon_days: int
    flat_threshold: float
    params: dict
    accuracy: float | None = None
    macro_f1: float | None = None
    log_loss: float | None = None
    status: str = "pending"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRecord":
        return cls(**data)


def reserve_final_holdout(df: pd.DataFrame, holdout_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be between 0 and 1")
    dates = sorted(df["date"].unique())
    cutoff_index = math.ceil(len(dates) * (1 - holdout_frac))
    if cutoff_index >= len(dates):
        raise ValueError("Insufficient dates for holdout reservation")
    cutoff = dates[cutoff_index]
    develop = df[df["date"] < cutoff].copy()
    holdout = df[df["date"] >= cutoff].copy()
    if develop.empty or holdout.empty:
        raise ValueError("Holdout reservation produced an empty split")
    return develop.reset_index(drop=True), holdout.reset_index(drop=True)


def walk_forward_folds(dates: list, n_folds: int = 3) -> list[tuple[list, list]]:
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    ordered = sorted(dates)
    if len(ordered) < n_folds + 1:
        raise ValueError("Insufficient dates for walk-forward folds")
    chunk = len(ordered) // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = chunk * (i + 1)
        val_end = chunk * (i + 2)
        folds.append((ordered[:train_end], ordered[train_end:val_end]))
    return folds


def add_nifty_context(stock: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    frame = stock.sort_values("date").copy()
    market = nifty.sort_values("date")[["date", "close"]].copy()
    market["nifty_return_1d"] = market["close"].pct_change(1)
    market["nifty_return_3d"] = market["close"].pct_change(3)
    market["nifty_return_5d"] = market["close"].pct_change(5)
    market["nifty_volatility_5d"] = market["nifty_return_1d"].rolling(5).std()
    frame = frame.merge(market.drop(columns=["close"]), on="date", how="left")
    frame["relative_return_3d"] = frame["return_3d"] - frame["nifty_return_3d"]
    frame["relative_return_5d"] = frame["return_5d"] - frame["nifty_return_5d"]
    return frame


def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.sort_values("date").copy()
    frame["return_10d"] = frame["close"].pct_change(10)
    frame["volatility_10d"] = frame["return_1d"].rolling(10).std()
    frame["ema_9_vs_21"] = (frame["ema_9"] - frame["ema_21"]) / frame["ema_21"]
    frame["volume_change_1d"] = frame["volume"].pct_change(1)
    frame["volume_change_5d"] = frame["volume"].pct_change(5)
    frame["high_low_range"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["close_open_return"] = (frame["close"] - frame["open"]) / frame["open"]
    return frame


def ablation_feature_sets(all_features: list[str] | None = None) -> dict[str, list[str]]:
    base = list(all_features) if all_features else list(FEATURE_COLS)
    groups = {
        "rsi": ["rsi_14"],
        "volume": ["vol_ratio", "obv_divergence"],
        "bollinger": ["bb_width", "bb_pct"],
        "returns": ["return_1d", "return_3d", "return_5d"],
        "market": ["nifty_return_1d", "nifty_return_3d", "nifty_return_5d", "nifty_volatility_5d", "relative_return_3d", "relative_return_5d"],
    }
    sets = {"all": base}
    for name, drop in groups.items():
        sets[f"without_{name}"] = [c for c in base if c not in drop]
    return sets


def build_tuning_candidates(seed: int = SEED, count: int = TUNING_CANDIDATE_COUNT) -> list[dict]:
    rng = random.Random(seed)
    depths = [2, 3, 4, 5, 6]
    rates = [0.01, 0.03, 0.05, 0.1]
    child_weights = [1, 3, 5, 10]
    subsamples = [0.6, 0.8, 1.0]
    colsample = [0.6, 0.8, 1.0]
    alphas = [0, 0.1, 0.5, 1.0]
    lambdas = [1, 2, 5, 10]
    candidates = []
    for _ in range(count):
        candidates.append({
            "max_depth": rng.choice(depths),
            "learning_rate": rng.choice(rates),
            "min_child_weight": rng.choice(child_weights),
            "subsample": rng.choice(subsamples),
            "colsample_bytree": rng.choice(colsample),
            "reg_alpha": rng.choice(alphas),
            "reg_lambda": rng.choice(lambdas),
            "n_estimators": 300,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": seed,
        })
    return candidates


def horizon_threshold_grid() -> list[tuple[int, float]]:
    return list(HORIZON_THRESHOLD_GRID)


def compare_calibration(estimator: Any, X_cal: np.ndarray, y_cal: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for method in CALIBRATION_METHODS:
        if method == "none":
            proba = estimator.predict_proba(X_val)
        elif method == "sigmoid":
            calibrator = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
            calibrator.fit(X_cal, y_cal)
            proba = calibrator.predict_proba(X_val)
        elif method == "isotonic":
            calibrator = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
            calibrator.fit(X_cal, y_cal)
            proba = calibrator.predict_proba(X_val)
        else:
            continue
        try:
            results[method] = float(log_loss(y_val, proba, labels=[0, 1, 2]))
        except ValueError:
            results[method] = None
    return results
