from unittest.mock import Mock

import pytest

from agents import graph, researcher_agent
from tests.test_api import client


@pytest.mark.parametrize("news_status", ["available", "empty", "unavailable"])
def test_real_langgraph_through_api(monkeypatch, news_status, client):
    from api import cache

    store = Mock()
    store.get.return_value = None
    monkeypatch.setattr(cache, "get_redis_client", lambda: store)
    predict = Mock(return_value={"ml_trend": "bearish", "ml_confidence": 0.35})
    monkeypatch.setattr(graph, "predict_trend", predict)
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    http = Mock()
    http.json.return_value = {"status": "ok", "articles": [
        {"title": "Infosys excellent growth and record success", "description": "Great profits",
         "source": {"name": "Example"}, "publishedAt": "2026-09-10T10:00:00Z"}
    ] if news_status == "available" else []}
    get = Mock(return_value=http)
    monkeypatch.setattr(researcher_agent.requests, "get", get)
    if news_status == "unavailable":
        monkeypatch.delenv("NEWS_API_KEY")
    response = client.post("/analyze", json={"ticker": " infy.ns ", "target_date": "2026-09-11"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "INFY"
    assert payload["ml_trend"] == "bearish"
    assert payload["ml_confidence"] == 0.35
    assert payload["news_status"] == news_status
    assert payload["divergence_flag"] is (news_status == "available")
    assert payload["news_sentiment"] == ("bullish" if news_status == "available" else None)
    assert "BEARISH" in payload["final_report"]
    if news_status == "available":
        assert "BULLISH" in payload["final_report"]
        assert "Signal Divergence" in payload["final_report"]
        assert "conflict" in payload["analyst_reasoning"]
    else:
        assert "Signals Aligned" not in payload["final_report"]
    predict.assert_called_once_with("INFY", "2026-09-11")
    store.get.return_value = store.setex.call_args.args[2]
    predict.reset_mock()
    get.reset_mock()
    assert client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"}).json() == payload
    predict.assert_not_called()
    get.assert_not_called()
