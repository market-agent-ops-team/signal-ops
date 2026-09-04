import os
import requests
import logging
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
   return TICKER_MAP.get(ticker, ticker.replace(".NS", "").replace(".BO", ""))
def fetch_news(company_name: str, max_articles: int = 10) -> list[dict]:
  api_key = os.getenv("NEWS_API_KEY")
  if not api_key:
    logger.warning("NEWS_API_KEY not found in .env — returning empty news")
    return []
  url = "https://newsapi.org/v2/everything"
  params = {
    "q": company_name,
    "sortBy": "publishedAt",
    "pageSize": max_articles,
    "language": "en",
    "apiKey": api_key,
  }
  try:
      resp = requests.get(url, params=params, timeout=10)
      resp.raise_for_status()
      raw_articles = resp.json().get("articles", [])
      articles = []
      for a in raw_articles:
           articles.append({
                "title": a.get("title") or "No title",
                "content": (a.get("description") or "")[:200],
                "source": a.get("source", {}).get("name", "Unknown"),
                "date": (a.get("publishedAt") or "")[:10],
                })
      return articles
  except requests.exceptions.Timeout:   
    logger.warning(f"NewsAPI request timed out for '{company_name}'")
    return []
  except requests.exceptions.HTTPError as e:
    logger.warning(f"NewsAPI HTTP error for '{company_name}': {e}")
    return []
  except Exception as e:
    logger.warning(f"NewsAPI unexpected error for '{company_name}': {e}")
    return []

def classify_sentiment(articles: list[dict]) -> list[dict]:
  if not articles:
    return articles
  analyzer = SentimentIntensityAnalyzer()
  for article in articles:
    text = article.get("title", "") + " " + article.get("content", "")
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
      article["sentiment"] = "positive"
    elif compound <= -0.05:
      article["sentiment"] = "negative"
    else:
      article["sentiment"] = "neutral"

  return articles
def aggregate_sentiment(articles: list[dict]) -> str:
    if not articles:
        return "neutral"
    sentiments = [a.get("sentiment", "neutral") for a in articles]
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    if pos > neg:
        return "bullish"
    elif neg > pos:
        return "bearish"
    return "neutral"
def news_research_node(state: dict) -> dict:
    ticker = state["ticker"]
    company_name = get_company_name(ticker)
    logger.info(f"[Researcher] Fetching news for {company_name} ({ticker})")
    articles = fetch_news(company_name)
    articles = classify_sentiment(articles)
    overall = aggregate_sentiment(articles)
    logger.info(f"[Researcher] Got {len(articles)} articles — overall: {overall}")
    return {
        "news_items": articles,
        "news_sentiments": overall,
    }
if __name__ == "__main__":
    test_state = {"ticker": "RELIANCE.NS", "target_date": "2026-08-11"}
    result = news_research_node(test_state)
    print(f"\nOverall sentiment: {result['news_sentiments']}")
    print(f"Articles found: {len(result['news_items'])}\n")
    for item in result["news_items"]:
        label = {"positive": "[+]", "negative": "[-]", "neutral": "[.]"}.get(item["sentiment"], "[.]")
        print(f"  {label} [{item['sentiment']:>8}] {item['title']}")
        print(f"    source: {item['source']}  |  date: {item['date']}")
        print()
  