import os
import requests
import logging

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


def get_company_name(ticker: str) -> str:
    ticker = ticker.strip().upper()

    if not ticker.endswith((".NS", ".BO")):
        ticker += ".NS"

    return TICKER_MAP.get(
        ticker,
        ticker.replace(".NS", "").replace(".BO", "")
    )


def fetch_news(
    company_name: str,
    target_date: str,
    max_articles: int = 10,
) -> list[dict]:

    if not company_name or not company_name.strip():
        logger.warning(
            "Empty company name provided. Returning empty news."
        )
        return []

    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        logger.warning(
            "NEWS_API_KEY not found in .env. Returning empty news."
        )
        return []

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

        raw_articles = (
            resp.json().get("articles") or []
        )

        company_lower = company_name.lower()
        articles = []

        for article in raw_articles:
            title = article.get("title") or ""
            description = article.get("description") or ""

            searchable_text = (
                title + " " + description
            ).lower()

            # Keep only articles that directly mention the company
            if company_lower not in searchable_text:
                continue

            articles.append(
                {
                    "title": title or "No title",
                    "content": description[:200],
                    "source": (
                        article.get("source", {})
                        .get("name", "Unknown")
                    ),
                    "date": (
                        article.get("publishedAt")
                        or ""
                    )[:10],
                }
            )

            if len(articles) >= max_articles:
                break

        return articles

    except requests.exceptions.Timeout:
        logger.warning(
            f"NewsAPI request timed out for "
            f"'{company_name}'"
        )
        return []

    except requests.exceptions.HTTPError as exc:
        logger.warning(
            f"NewsAPI HTTP error for "
            f"'{company_name}': {exc}"
        )
        return []

    except Exception as exc:
        logger.warning(
            f"NewsAPI unexpected error for "
            f"'{company_name}': {exc}"
        )
        return []

    except requests.exceptions.Timeout:
        logger.warning(
            f"NewsAPI request timed out for "
            f"'{company_name}'"
        )
        return []

    except requests.exceptions.HTTPError as exc:
        logger.warning(
            f"NewsAPI HTTP error for "
            f"'{company_name}': {exc}"
        )
        return []

    except Exception as exc:
        logger.warning(
            f"NewsAPI unexpected error for "
            f"'{company_name}': {exc}"
        )
        return []


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

    if not ticker or not ticker.strip():
        logger.error(
            "[Researcher] Empty ticker received. "
            "Defaulting to neutral state."
        )

        return {
            "news_items": [],
            "news_sentiments": "neutral",
        }

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

    articles = classify_sentiment(articles)

    overall = aggregate_sentiment(articles)

    logger.info(
        f"[Researcher] Got {len(articles)} articles. "
        f"Overall sentiment: {overall}"
    )

    return {
        "news_items": articles,
        "news_sentiments": overall,
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