import logging
from collections.abc import Mapping
from typing import Any


logger = logging.getLogger(__name__)

_SIGNAL_ALIASES = {
    "bullish": "bullish",
    "positive": "bullish",
    "up": "bullish",
    "upward": "bullish",
    "buy": "bullish",
    "bearish": "bearish",
    "negative": "bearish",
    "down": "bearish",
    "downward": "bearish",
    "sell": "bearish",
    "neutral": "neutral",
    "flat": "neutral",
    "sideways": "neutral",
    "hold": "neutral",
}


def normalize_signal(signal: Any) -> str:
    value = str(signal or "").strip().lower()
    return _SIGNAL_ALIASES.get(value, value or "unknown")


def detect_divergence(ml_trend: Any, news_sentiment: Any) -> bool:
    model_signal = normalize_signal(ml_trend)
    news_signal = normalize_signal(news_sentiment)
    return {model_signal, news_signal} == {"bullish", "bearish"}


def format_confidence(confidence: Any) -> str:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "unknown confidence"
    return f"{min(max(value, 0.0), 1.0) * 100:.1f}% confidence"


def generate_analyst_reasoning(
    ml_trend: Any,
    ml_confidence: Any,
    news_sentiment: Any,
    divergence_flag: bool,
) -> str:
    model_signal = normalize_signal(ml_trend)
    news_signal = normalize_signal(news_sentiment)
    confidence = format_confidence(ml_confidence)

    if divergence_flag:
        return (
            f"The model is {model_signal} ({confidence}), while aggregated news "
            f"sentiment is {news_signal}. Technical and news signals conflict, so "
            "the position warrants additional review before acting on either signal."
        )

    return (
        f"The model is {model_signal} ({confidence}) and aggregated news sentiment "
        f"is {news_signal}. No direct bullish-versus-bearish divergence is present."
    )


def analyst_node(state: Mapping[str, Any]) -> dict[str, Any]:
    ml_trend = state.get("ml_trend")
    ml_confidence = state.get("ml_confidence")
    news_sentiment = state.get("news_sentiments")
    divergence_flag = detect_divergence(ml_trend, news_sentiment)
    analyst_reasoning = generate_analyst_reasoning(
        ml_trend,
        ml_confidence,
        news_sentiment,
        divergence_flag,
    )

    logger.info(
        "[Analyst] Model=%s, news=%s, divergence=%s",
        normalize_signal(ml_trend),
        normalize_signal(news_sentiment),
        divergence_flag,
    )
    return {
        "divergence_flag": divergence_flag,
        "analyst_reasoning": analyst_reasoning,
    }
