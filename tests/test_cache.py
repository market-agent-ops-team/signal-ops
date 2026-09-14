import json
from unittest.mock import Mock

import pytest
import redis

from tests.test_api import api_setup, client


@pytest.mark.parametrize("ticker", ["INFY", "infy", " INFY.NS "])
def test_key_normalization(ticker):
    from api.cache import build_cache_key

    assert build_cache_key(ticker, "2026-09-11") == "analysis:INFY:2026-09-11"


@pytest.mark.parametrize("ticker,target", [("INFY.BO", "2026-09-11"), ("INFY.NS.NS", "2026-09-11"),
    ("INFY", "2026-9-11"), ("INFY", "2026-02-30"), ("", "2026-09-11")])
def test_key_rejects_invalid_inputs(ticker, target):
    from api.cache import build_cache_key

    with pytest.raises(ValueError):
        build_cache_key(ticker, target)


@pytest.mark.parametrize("ttl", [None, "37"])
def test_cache_miss_write_ttl_then_hit_exact_equality(api_setup, monkeypatch, ttl):
    client, graph, store, _ = api_setup
    if ttl is None:
        monkeypatch.delenv("CACHE_TTL_SECONDS", raising=False)
    else:
        monkeypatch.setenv("CACHE_TTL_SECONDS", ttl)
    response = client.post("/analyze", json={"ticker": "infy.ns", "target_date": "2026-09-11"})
    assert response.status_code == 200
    key, seconds, encoded = store.setex.call_args.args
    assert key == "analysis:INFY:2026-09-11"
    assert seconds == (900 if ttl is None else 37)
    assert json.loads(encoded) == response.json()
    assert "news_sentiments" not in json.loads(encoded)
    store.get.return_value = encoded
    graph.invoke.reset_mock()
    store.setex.reset_mock()
    cached = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert cached.json() == response.json()
    graph.invoke.assert_not_called()
    store.setex.assert_not_called()


@pytest.mark.parametrize("encoded", ["not-json", "null", "[]", "{}", "true", '"text"', b"\xff"])
def test_corrupt_cache_recomputed(api_setup, encoded):
    client, graph, store, _ = api_setup
    store.get.return_value = encoded
    response = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert response.status_code == 200
    assert response.json()["ml_confidence"] == 0.3483
    graph.invoke.assert_called_once()
    store.setex.assert_called_once()


@pytest.mark.parametrize("field,value", [
    ("ticker", "TCS"), ("ticker", "INFY.BO"), ("ticker", " infy.ns "),
    ("target_date", "2026-09-10"), ("target_date", "2026-9-11"),
    ("ml_trend", "up"), ("ml_confidence", float("nan")), ("ml_confidence", float("inf")),
    ("ml_confidence", -1), ("ml_confidence", 2), ("ml_confidence", True),
    ("news_sentiment", None), ("news_sentiment", "positive"),
    ("news_status", "empty"), ("news_status", "bad"),
    ("divergence_flag", False), ("divergence_flag", "true"),
    ("final_report", ""), ("analyst_reasoning", None), ("extra", "unexpected"),
])
def test_invalid_cached_payload_recomputed(api_setup, field, value):
    client, graph, store, result = api_setup
    payload = dict(result)
    payload["news_sentiment"] = payload.pop("news_sentiments")
    payload[field] = value
    store.get.return_value = json.dumps(payload)
    response = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert response.status_code == 200
    assert response.json()["ticker"] == "INFY"
    assert response.json()["ml_trend"] == "bearish"
    graph.invoke.assert_called_once()
    store.setex.assert_called_once()


@pytest.mark.parametrize("operation", ["get", "setex"])
def test_redis_outage_safe_fallback(api_setup, caplog, operation):
    client, graph, store, _ = api_setup
    getattr(store, operation).side_effect = redis.ConnectionError("redis://secret@host")
    response = client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"})
    assert response.status_code == 200
    assert response.json()["divergence_flag"] is True
    graph.invoke.assert_called_once()
    assert "secret" not in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_redis_configuration_short_timeouts(monkeypatch):
    from api import cache

    constructor = Mock()
    monkeypatch.setattr(cache.redis.Redis, "from_url", constructor)
    monkeypatch.setenv("REDIS_URL", "redis://example:6379/2")
    cache.get_redis_client.cache_clear()
    try:
        cache.get_redis_client()
        cache.get_redis_client()
        constructor.assert_called_once()
        assert constructor.call_args.args == ("redis://example:6379/2",)
        kwargs = constructor.call_args.kwargs
        assert 0 < kwargs["socket_timeout"] <= 1
        assert 0 < kwargs["socket_connect_timeout"] <= 1
        assert kwargs["decode_responses"] is True
    finally:
        cache.get_redis_client.cache_clear()


@pytest.mark.parametrize("ttl", ["bad", "0", "-1"])
def test_invalid_cache_configuration_falls_back(api_setup, monkeypatch, caplog, ttl):
    client, _, store, _ = api_setup
    monkeypatch.setenv("CACHE_TTL_SECONDS", ttl)
    assert client.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"}).status_code == 200
    store.setex.assert_not_called()
    assert caplog.records
