import pytest

from agents.editor_agent import build_report


@pytest.mark.parametrize("model,news,status,heading", [
    ("bullish", "bullish", "available", "Signals Aligned"),
    ("bearish", "bearish", "available", "Signals Aligned"),
    ("neutral", "bullish", "available", "No Directional Alignment"),
    ("bullish", "neutral", "available", "No Directional Alignment"),
    ("neutral", "neutral", "available", "No Directional Alignment"),
    ("bullish", None, "unavailable", "News Unavailable"),
    ("bullish", None, "empty", "No Relevant News"),
])
def test_report_semantics(model, news, status, heading):
    report = build_report({"ticker": "INFY", "target_date": "2026-09-11",
        "ml_trend": model, "ml_confidence": 0.3483, "news_sentiments": news,
        "news_status": status, "divergence_flag": False})
    assert heading in report
    assert "34.8%" in report
    assert "INFY" in report
    if heading != "Signals Aligned":
        assert "Signals Aligned" not in report


def test_partial_state_does_not_claim_alignment():
    report = build_report({})
    assert "Signals Aligned" not in report
    assert "unknown confidence" in report


def test_divergence_report():
    report = build_report({"ml_trend": "bearish", "ml_confidence": 0.35,
        "news_sentiments": "bullish", "news_status": "available", "divergence_flag": True,
        "analyst_reasoning": "Conflicting evidence", "news_items": [
            {"title": "Infosys growth", "source": "Example", "date": "2026-09-10", "sentiment": "positive"}]})
    for text in ("Signal Divergence", "BEARISH", "BULLISH", "Conflicting evidence", "Infosys growth", "Example"):
        assert text in report
