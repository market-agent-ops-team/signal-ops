from functools import lru_cache
from datetime import date
from pathlib import Path
import re
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd

from ml.dataProcessing.data_fetch import pick_OHLCV
from ml.features.indicators import V1_HISTORY_START, generate_inference_features


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

    if not isinstance(target_date, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", target_date):
        raise ValueError("target_date must be a valid YYYY-MM-DD date")
    try:
        target = pd.Timestamp(date.fromisoformat(target_date))
    except ValueError as exc:
        raise ValueError(
            "target_date must be a valid YYYY-MM-DD date"
        ) from exc

    # yfinance treats end date as exclusive.
    end = target + pd.Timedelta(days=1)

    raw = pick_OHLCV(
        ticker,
        V1_HISTORY_START,
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
