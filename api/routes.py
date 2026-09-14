import asyncio
import logging
import math
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from agents.analyst_agent import detect_divergence
from agents.graph import app as graph_app
from agents.researcher_agent import normalize_ticker, validate_target_date
from api.cache import get_cached_analysis, set_cached_analysis


logger = logging.getLogger(__name__)
router = APIRouter()

VALID_TRENDS = {"bullish", "bearish", "neutral"}
VALID_STATUSES = {"available", "empty", "unavailable"}
RESPONSE_KEYS = {"ticker", "target_date", "ml_trend", "ml_confidence", "news_sentiment", "news_status", "divergence_flag", "analyst_reasoning", "final_report"}


class AnalyzeRequest(BaseModel):
    ticker: str
    target_date: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Ticker must be a string")
        return normalize_ticker(v)

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Target date must be a string")
        return validate_target_date(v)


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    target_date: str
    ml_trend: str
    ml_confidence: float
    news_sentiment: str | None
    news_status: str
    divergence_flag: bool
    analyst_reasoning: str
    final_report: str


def _valid_confidence(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    as_float = float(value)
    return math.isfinite(as_float) and 0.0 <= as_float <= 1.0


def _valid_sentiment_for_status(sentiment: Any, status: Any) -> bool:
    if status == "available":
        return sentiment in VALID_TRENDS
    if status in VALID_STATUSES:
        return sentiment is None
    return False


def _expected_divergence(ml_trend: Any, sentiment: Any, status: Any) -> bool:
    return status == "available" and detect_divergence(ml_trend, sentiment)


def _validate_common(ticker: Any, target_date: Any, ml_trend: Any, confidence: Any, status: Any, divergence: Any, reasoning: Any, report: Any, expected_ticker: str, expected_date: str) -> bool:
    if ticker != expected_ticker or target_date != expected_date:
        return False
    if ml_trend not in VALID_TRENDS:
        return False
    if not _valid_confidence(confidence):
        return False
    if status not in VALID_STATUSES:
        return False
    if type(divergence) is not bool:
        return False
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False
    if not isinstance(report, str) or not report.strip():
        return False
    return True


def _validate_graph_output(data: Any, expected_ticker: str, expected_date: str) -> bool:
    if not isinstance(data, dict):
        return False
    sentiment = data.get("news_sentiments")
    status = data.get("news_status")
    if not _validate_common(data.get("ticker"), data.get("target_date"), data.get("ml_trend"), data.get("ml_confidence"), status, data.get("divergence_flag"), data.get("analyst_reasoning"), data.get("final_report"), expected_ticker, expected_date):
        return False
    if not _valid_sentiment_for_status(sentiment, status):
        return False
    return data.get("divergence_flag") is _expected_divergence(data.get("ml_trend"), sentiment, status)


def _validate_cached_payload(data: Any, expected_ticker: str, expected_date: str) -> bool:
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != RESPONSE_KEYS:
        return False
    sentiment = data.get("news_sentiment")
    status = data.get("news_status")
    if not _validate_common(data.get("ticker"), data.get("target_date"), data.get("ml_trend"), data.get("ml_confidence"), status, data.get("divergence_flag"), data.get("analyst_reasoning"), data.get("final_report"), expected_ticker, expected_date):
        return False
    if not _valid_sentiment_for_status(sentiment, status):
        return False
    return data.get("divergence_flag") is _expected_divergence(data.get("ml_trend"), sentiment, status)


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    cached = get_cached_analysis(request.ticker, request.target_date)
    if _validate_cached_payload(cached, request.ticker, request.target_date):
        return AnalyzeResponse(**cached)

    try:
        initial_state = {
            "ticker": request.ticker,
            "target_date": request.target_date
        }
        final_state = await asyncio.to_thread(graph_app.invoke, initial_state)
    except FileNotFoundError:
        logger.warning("Analysis failed: model artifacts unavailable.")
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable")
    except ValueError:
        logger.warning("Analysis failed: invalid analysis data.")
        raise HTTPException(status_code=422, detail="Analysis data is invalid or unavailable")
    except Exception:
        logger.warning("Analysis failed during graph execution.")
        raise HTTPException(status_code=500, detail="Analysis failed")

    if not _validate_graph_output(final_state, request.ticker, request.target_date):
        logger.warning("Analysis failed: invalid pipeline response.")
        raise HTTPException(status_code=500, detail="Analysis failed")

    response_payload = {
        "ticker": final_state["ticker"],
        "target_date": final_state["target_date"],
        "ml_trend": final_state["ml_trend"],
        "ml_confidence": float(final_state["ml_confidence"]),
        "news_sentiment": final_state["news_sentiments"],
        "news_status": final_state["news_status"],
        "divergence_flag": final_state["divergence_flag"],
        "analyst_reasoning": final_state["analyst_reasoning"],
        "final_report": final_state["final_report"],
    }

    if not _validate_cached_payload(response_payload, request.ticker, request.target_date):
        logger.warning("Analysis failed: invalid pipeline response.")
        raise HTTPException(status_code=500, detail="Analysis failed")

    set_cached_analysis(request.ticker, request.target_date, response_payload)
    return AnalyzeResponse(**response_payload)
