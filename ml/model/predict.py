from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd

from ml.dataProcessing.data_fetch import pick_OHLCV
from ml.features.indicators import generate_inference_features


class TrendPrediction(TypedDict):
    ml_trend: str
    ml_confidence: float


ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "saved_models"
)


@lru_cache(maxsize=1)
def _load_artifacts():
    model_path = ARTIFACT_DIR / "xgb_model.joblib"
    scaler_path = ARTIFACT_DIR / "scaler.joblib"
    features_path = ARTIFACT_DIR / "feature_names.joblib"

    for path in (model_path,scaler_path,features_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}"
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    return model, scaler, feature_names


def predict_trend(
    ticker: str,
    target_date: str,
) -> TrendPrediction:

    ticker = ticker.strip()

    if not ticker:
        raise ValueError("ticker must not be empty")

    try:
        target = pd.Timestamp(target_date)
    except Exception as exc:
        raise ValueError(
            "target_date must be a valid date"
        ) from exc

    # Fetch enough history for 50-day SMA and other indicators.
    start = target - pd.Timedelta(days=180)

    # yfinance treats end date as exclusive.
    end = target + pd.Timedelta(days=1)

    raw = pick_OHLCV(
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    raw["date"] = pd.to_datetime(raw["date"])

    # Handles weekends and market holidays by using the
    # latest trading session on or before target_date.
    raw = raw[
        raw["date"] <= target
    ].copy()

    if raw.empty:
        raise ValueError(
            f"No market data available for {ticker} "
            f"on or before {target_date}"
        )

    features = generate_inference_features(raw)

    if features.empty:
        raise ValueError(
            "Insufficient valid feature rows for prediction"
        )

    model, scaler, feature_names = _load_artifacts()

    missing = [
        col
        for col in feature_names
        if col not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing inference features: {missing}"
        )

    latest = features.iloc[[-1]]

    X = scaler.transform(
        latest[feature_names]
    )

    probabilities = model.predict_proba(X)[0]

    best_position = int(
        np.argmax(probabilities)
    )

    encoded_class = int(
        model.classes_[best_position]
    )

    class_to_trend = {
        0: "bearish",
        1: "neutral",
        2: "bullish",
    }

    if encoded_class not in class_to_trend:
        raise ValueError(
            f"Unexpected model class: {encoded_class}"
        )

    return {
        "ml_trend": class_to_trend[encoded_class],
        "ml_confidence": float(
            probabilities[best_position]
        ),
    }