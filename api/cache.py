import json
import logging
import os
from functools import lru_cache
from typing import Any

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from agents.researcher_agent import normalize_ticker, validate_target_date


logger = logging.getLogger(__name__)


def build_cache_key(ticker: str, target_date: str) -> str:
    return f"analysis:{normalize_ticker(ticker)}:{validate_target_date(target_date)}"


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
        retry=Retry(NoBackoff(), 0),
    )


def get_cached_analysis(ticker: str, target_date: str) -> Any:
    key = build_cache_key(ticker, target_date)
    try:
        encoded = get_redis_client().get(key)
        return json.loads(encoded) if isinstance(encoded, (str, bytes)) else None
    except (redis.RedisError, ValueError, UnicodeError):
        logger.warning("Analysis cache read failed; recomputing.")
        return None


def set_cached_analysis(ticker: str, target_date: str, payload: dict) -> None:
    key = build_cache_key(ticker, target_date)
    try:
        ttl = int(os.getenv("CACHE_TTL_SECONDS", "900"))
        if ttl <= 0:
            raise ValueError("Cache TTL must be positive")
        encoded = json.dumps(payload, allow_nan=False)
        get_redis_client().setex(key, ttl, encoded)
    except (redis.RedisError, ValueError, UnicodeError):
        logger.warning("Analysis cache write failed; returning uncached response.")
