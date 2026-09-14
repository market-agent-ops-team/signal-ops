import asyncio
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api import routes


@pytest.fixture
def api_setup(monkeypatch, client):
    from api import cache

    redis = Mock()
    redis.get.return_value = None
    monkeypatch.setattr(cache, "get_redis_client", lambda: redis)
    result = {"ticker": "INFY", "target_date": "2026-09-11", "ml_trend": "bearish",
              "ml_confidence": 0.3483, "news_sentiments": "bullish", "news_status": "available",
              "divergence_flag": True, "analyst_reasoning": "Signals conflict", "final_report": "# Report"}
    graph = Mock()
    graph.invoke.return_value = result
    monkeypatch.setattr(routes, "graph_app", graph)
    return client, graph, redis, result


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_does_no_work(api_setup):
    client, graph, redis, _ = api_setup
    assert client.get("/health").json() == {"status": "ok"}
    graph.invoke.assert_not_called()
    redis.get.assert_not_called()


@pytest.mark.parametrize("ticker", ["INFY", "infy", " infy.ns "])
def test_valid_normalized_request(api_setup, ticker):
    client, graph, redis, _ = api_setup
    response = client.post("/analyze", json={"ticker": ticker, "target_date": "2026-09-11"})
    assert response.status_code == 200
    assert response.json() == {"ticker": "INFY", "target_date": "2026-09-11", "ml_trend": "bearish",
        "ml_confidence": 0.3483, "news_sentiment": "bullish", "news_status": "available",
        "divergence_flag": True, "analyst_reasoning": "Signals conflict", "final_report": "# Report"}
    graph.invoke.assert_called_once_with({"ticker": "INFY", "target_date": "2026-09-11"})
    redis.get.assert_called_once_with("analysis:INFY:2026-09-11")


@pytest.mark.parametrize("payload", [{}, {"target_date": "2026-09-11"},
    *[{"ticker": t, "target_date": "2026-09-11"} for t in ["", " ", "INFY.BO", "INFY.L", "INFY.NS.NS", "A:B", None, 123]],
    *[{"ticker": "INFY", "target_date": d} for d in ["2026-9-11", "2026-09-1", "20260911", "2026-02-30", "2026-09-11T00:00:00", None, 123]]])
def test_bad_request_never_reaches_cache_or_graph(api_setup, payload):
    client, graph, redis, _ = api_setup
    assert client.post("/analyze", json=payload).status_code == 422
    graph.invoke.assert_not_called()
    redis.get.assert_not_called()


@pytest.mark.parametrize("error,status,detail", [
    (ValueError("secret-url"), 422, "Analysis data is invalid or unavailable"),
    (FileNotFoundError("secret-path"), 503, "Model artifacts are unavailable"),
    (RuntimeError("secret-traceback"), 500, "Analysis failed"),
])
def test_controlled_graph_errors(api_setup, caplog, error, status, detail):
    client, graph, redis, _ = api_setup
    graph.invoke.side_effect = error
    response = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert "secret" not in caplog.text
    redis.setex.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("ml_trend", "up"), ("ml_confidence", float("nan")), ("ml_confidence", float("inf")),
    ("ml_confidence", -0.1), ("ml_confidence", 1.1), ("ml_confidence", "0.3"),
    ("news_sentiments", "positive"), ("news_sentiments", None), ("news_status", "broken"),
    ("news_status", "unavailable"), ("divergence_flag", False), ("divergence_flag", "true"),
    ("final_report", ""), ("analyst_reasoning", " "), ("ticker", "TCS"), ("target_date", "2026-09-10"),
])
def test_invalid_final_response_not_cached(api_setup, field, value):
    client, _, redis, result = api_setup
    result[field] = value
    response = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis failed"}
    redis.setex.assert_not_called()


@pytest.mark.parametrize("trend", ["bullish", "neutral", "bearish"])
@pytest.mark.parametrize("status,sentiment", [("available", "neutral"), ("empty", None), ("unavailable", None)])
def test_three_trends_and_news_availability(api_setup, trend, status, sentiment):
    client, _, _, result = api_setup
    result.update(ml_trend=trend, news_status=status, news_sentiments=sentiment, divergence_flag=False)
    response = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert response.status_code == 200
    assert response.json()["news_sentiment"] == sentiment
    assert response.json()["news_status"] == status


def test_graph_runs_outside_async_loop(api_setup):
    client, graph, _, result = api_setup

    def invoke(state):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return result

    graph.invoke.side_effect = invoke
    assert client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"}).status_code == 200
