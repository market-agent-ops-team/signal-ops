import os
import time

import pytest
import redis

pytestmark = pytest.mark.live


def _redis_client():
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
    )


def test_redis_expiry_contract():
    client = _redis_client()
    client.ping()
    client.setex("analysis:TEST:2000-01-01", 1, '{"probe": true}')
    assert client.ttl("analysis:TEST:2000-01-01") <= 1
    time.sleep(1.2)
    assert client.get("analysis:TEST:2000-01-01") is None


def test_redis_outage_does_not_break_imports():
    from api import cache

    client = _redis_client()
    try:
        client.ping()
    except redis.RedisError:
        pytest.skip("Redis unavailable for live outage check")
    assert callable(cache.get_cached_analysis)
    assert callable(cache.set_cached_analysis)


def test_live_api_health():
    import urllib.request

    base = os.getenv("LIVE_API_URL", "http://localhost:8000")
    with urllib.request.urlopen(base + "/health", timeout=5) as response:
        assert response.status == 200
