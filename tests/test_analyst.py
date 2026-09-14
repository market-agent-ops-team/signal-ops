import pytest

from agents.analyst_agent import analyst_node, detect_divergence, format_confidence, normalize_signal


@pytest.mark.parametrize("model,news,expected", [
    ("bullish", "bullish", False), ("bullish", "bearish", True),
    ("bearish", "bullish", True), ("bearish", "bearish", False),
    ("neutral", "bullish", False), ("neutral", "bearish", False),
])
def test_six_divergence_cases(model, news, expected):
    assert detect_divergence(model, news) is expected


@pytest.mark.parametrize("alias,expected", [
    (" positive ", "bullish"), ("NEGATIVE", "bearish"),
    ("up", "bullish"), ("down", "bearish"), ("flat", "neutral"),
])
def test_signal_aliases(alias, expected):
    assert normalize_signal(alias) == expected


@pytest.mark.parametrize("value,expected", [
    (0, "0.0% confidence"), (0.3483, "34.8% confidence"),
    (1, "100.0% confidence"), ("bad", "unknown confidence"),
    (None, "unknown confidence"), (float("nan"), "unknown confidence"),
    (float("inf"), "unknown confidence"), (-0.1, "unknown confidence"),
    (1.1, "unknown confidence"),
])
def test_confidence(value, expected):
    assert format_confidence(value) == expected


@pytest.mark.parametrize("status", ["unavailable", "empty"])
def test_missing_news_cannot_diverge_or_claim_observed_sentiment(status):
    result = analyst_node({"ml_trend": "bearish", "ml_confidence": 0.35,
                           "news_sentiments": "bullish", "news_status": status})
    assert result["divergence_flag"] is False
    assert status in result["analyst_reasoning"].lower()
    assert "sentiment is bullish" not in result["analyst_reasoning"]
