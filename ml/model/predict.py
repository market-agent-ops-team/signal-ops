from typing import TypedDict


class TrendPrediction(TypedDict):
    ml_trend: str
    ml_confidence: float


def predict_trend(ticker: str, target_date: str) -> TrendPrediction:
    if not ticker.strip():
        raise ValueError("ticker must not be empty")
    if not target_date.strip():
        raise ValueError("target_date must not be empty")
    return {
        "ml_trend": "bullish",
        "ml_confidence": 0.9,
    }
