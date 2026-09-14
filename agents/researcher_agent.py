import os
import requests
import logging
import re

from datetime import datetime, timedelta
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TICKER_MAP = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ITC.NS": "ITC Limited",
}


def normalize_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper().removesuffix(".NS")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9&-]{0,29}", normalized):
        raise ValueError("Ticker must be an NSE symbol, optionally ending in .NS")
    return normalized


def validate_target_date(target_date: str) -> str:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", target_date):
        raise ValueError("target_date must use YYYY-MM-DD format")
    datetime.strptime(target_date, "%Y-%m-%d")
    return target_date


def get_company_name(ticker: str) -> str:
    normalized = normalize_ticker(ticker)
    return TICKER_MAP.get(normalized + ".NS", normalized)


def fetch_news(
    company_name: str,
    target_date: str,
    max_articles: int = 10,
) -> list[dict] | None:

    validate_target_date(target_date)
    if not company_name or not company_name.strip():
        raise ValueError("Company name cannot be empty")

    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        logger.warning(
            "News unavailable: missing API key."
        )
        return None

    try:
        analysis_date = datetime.strptime(
            target_date,
            "%Y-%m-%d",
        )
    except ValueError as exc:
        raise ValueError(
            "target_date must use YYYY-MM-DD format"
        ) from exc

    from_date = (
        analysis_date - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    to_date = analysis_date.strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": f'"{company_name}"',
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "pageSize": 50,
        "language": "en",
    }

    headers = {
        "X-Api-Key": api_key,
    }

    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10,
        )

        resp.raise_for_status()

        payload = resp.json()
        if not isinstance(payload, dict) or payload.get("status") != "ok" or not isinstance(payload.get("articles"), list):
            logger.warning("News unavailable: invalid response schema.")
            return None
        raw_articles = payload["articles"]

        company_lower = company_name.lower()
        articles = []
        valid_items = 0

        for article in raw_articles:
            if not isinstance(article, dict):
                continue
            if any(article.get(key) is not None and not isinstance(article[key], str)
                   for key in ("title", "description", "publishedAt")):
                continue
            title = article.get("title") or ""
            description = article.get("description") or ""
            if not (title.strip() or description.strip()):
                continue
            valid_items += 1

            searchable_text = (
                title + " " + description
            ).lower()

            if company_lower not in searchable_text:
                continue

            source = article.get("source")
            source_name = source.get("name") if isinstance(source, dict) else None

            articles.append(
                {
                    "title": title or "No title",
                    "content": description[:200],
                    "source": source_name if isinstance(source_name, str) and source_name.strip() else "Unknown",
                    "date": (
                        article.get("publishedAt")
                        or ""
                    )[:10],
                }
            )

            if len(articles) >= max_articles:
                break

        if raw_articles and not valid_items:
            logger.warning("News unavailable: no valid article records.")
            return None
        return articles

    except (requests.exceptions.RequestException, ValueError):
        logger.warning("News unavailable: request or JSON decoding failed.")
        return None


def classify_sentiment(
    articles: list[dict],
) -> list[dict]:

    if not articles:
        return articles

    analyzer = SentimentIntensityAnalyzer()

    for article in articles:
        text = (
            article.get("title", "")
            + " "
            + article.get("content", "")
        )

        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            article["sentiment"] = "positive"

        elif compound <= -0.05:
            article["sentiment"] = "negative"

        else:
            article["sentiment"] = "neutral"

    return articles


def aggregate_sentiment(
    articles: list[dict],
) -> str:

    if not articles:
        return "neutral"

    sentiments = [
        article.get("sentiment", "neutral")
        for article in articles
    ]

    positive = sentiments.count("positive")
    negative = sentiments.count("negative")

    if positive > negative:
        return "bullish"

    if negative > positive:
        return "bearish"

    return "neutral"


def news_research_node(state: dict) -> dict:

    ticker = state.get("ticker", "")
    target_date = state.get("target_date", "")

    if not target_date:
        raise ValueError(
            "target_date is required for news research"
        )

    company_name = get_company_name(ticker)

    logger.info(
        f"[Researcher] Fetching news for "
        f"{company_name} ({ticker}) "
        f"around {target_date}"
    )

    articles = fetch_news(
        company_name,
        target_date,
    )

    if articles is None:
        return {"news_items": [], "news_sentiments": None, "news_status": "unavailable"}
    if not articles:
        return {"news_items": [], "news_sentiments": None, "news_status": "empty"}

    articles = classify_sentiment(articles)

    overall = aggregate_sentiment(articles)

    logger.info(
        f"[Researcher] Got {len(articles)} articles. "
        f"Overall sentiment: {overall}"
    )

    return {
        "news_items": articles,
        "news_sentiments": overall,
        "news_status": "available",
    }


if __name__ == "__main__":
    test_state = {
        "ticker": "INFY",
        "target_date": "2026-09-11",
    }

    result = news_research_node(test_state)

    print(
        f"\nOverall sentiment: "
        f"{result['news_sentiments']}"
    )

    print(
        f"Articles found: "
        f"{len(result['news_items'])}\n"
    )

    for item in result["news_items"]:
        label = {
            "positive": "[+]",
            "negative": "[-]",
            "neutral": "[.]",
        }.get(
            item["sentiment"],
            "[.]",
        )

        print(
            f"  {label} "
            f"[{item['sentiment']:>8}] "
            f"{item['title']}"
        )

        print(
            f"    source: {item['source']} | "
            f"date: {item['date']}"
        )

        print()
