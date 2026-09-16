from unittest.mock import Mock

import pytest
import requests

from agents import researcher_agent as researcher


@pytest.fixture
def news_http(monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", "test-secret")
    response = Mock()
    response.json.return_value = {"status": "ok", "articles": []}
    get = Mock(return_value=response)
    monkeypatch.setattr(researcher.requests, "get", get)
    return get, response


@pytest.mark.parametrize("ticker", ["INFY", "infy.ns", " INFY "])
def test_company_aliases(ticker):
    assert researcher.get_company_name(ticker) == "Infosys"


@pytest.mark.parametrize("ticker", ["", " ", "INFY.BO", "INFY.L", "INFY.NS.NS", "NSE:INFY"])
def test_bad_tickers_rejected(ticker):
    with pytest.raises(ValueError):
        researcher.get_company_name(ticker)


@pytest.mark.parametrize("target", ["2026-9-01", "2026-09-1", "2026/09/01", "2026-02-30", " 2026-09-01", ""])
def test_strict_date_even_without_key(target, monkeypatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        researcher.news_research_node({"ticker": "INFY", "target_date": target})


def test_missing_key_is_unavailable(monkeypatch, news_http):
    monkeypatch.delenv("NEWS_API_KEY")
    result = researcher.news_research_node({"ticker": "INFY", "target_date": "2026-09-11"})
    assert result == {"news_items": [], "news_sentiments": None, "news_status": "unavailable"}
    news_http[0].assert_not_called()


@pytest.mark.parametrize("failure", [requests.Timeout("secret-url"), requests.HTTPError("secret-url"),
                                      requests.ConnectionError("secret-url"), ValueError("secret-url")])
def test_external_failure_safe(news_http, caplog, failure):
    news_http[0].side_effect = failure
    result = researcher.news_research_node({"ticker": "INFY", "target_date": "2026-09-11"})
    assert result["news_status"] == "unavailable"
    assert result["news_sentiments"] is None
    assert "secret-url" not in caplog.text
    assert caplog.records


@pytest.mark.parametrize("payload", [None, [], {}, {"status": "error", "articles": []},
    {"status": "ok", "articles": None}, {"status": "ok", "articles": {}},
    {"status": "ok", "articles": [None, 1, {"title": 5}]}])
def test_malformed_news_is_unavailable(news_http, payload):
    news_http[1].json.return_value = payload
    result = researcher.news_research_node({"ticker": "INFY", "target_date": "2026-09-11"})
    assert result["news_status"] == "unavailable"
    assert result["news_sentiments"] is None


def test_malformed_json(news_http):
    news_http[1].json.side_effect = ValueError("secret-url")
    assert researcher.news_research_node({"ticker": "INFY", "target_date": "2026-09-11"})["news_status"] == "unavailable"


def test_http_status_failure(news_http, caplog):
    news_http[1].raise_for_status.side_effect = requests.HTTPError("secret-url")
    assert researcher.news_research_node({"ticker": "INFY", "target_date": "2026-09-11"})["news_status"] == "unavailable"
    assert "secret-url" not in caplog.text


@pytest.mark.parametrize("articles", [[], [{"title": "Another company", "description": None, "source": None}]])
def test_empty_relevant_news(news_http, articles):
    news_http[1].json.return_value = {"status": "ok", "articles": articles}
    result = researcher.news_research_node({"ticker": "INFY", "target_date": "2026-09-11"})
    assert result == {"news_items": [], "news_sentiments": None, "news_status": "empty"}


def test_valid_neutral_and_null_source_with_invalid_items(news_http):
    news_http[1].json.return_value = {"status": "ok", "articles": [None, {"title": 12},
        {"title": "Infosys meeting on Tuesday", "description": None, "source": None, "publishedAt": None}]}
    result = researcher.news_research_node({"ticker": "infy.ns", "target_date": "2026-09-11"})
    assert result["news_status"] == "available"
    assert result["news_sentiments"] == "neutral"
    assert len(result["news_items"]) == 1
    assert result["news_items"][0]["source"] == "Unknown"
    kwargs = news_http[0].call_args.kwargs
    assert kwargs["params"]["q"] == '"Infosys"'
    assert kwargs["params"]["from"] == "2026-09-04"
    assert kwargs["params"]["to"] == "2026-09-11"
    assert kwargs["headers"] == {"X-Api-Key": "test-secret"}
    assert 0 < kwargs["timeout"] <= 10


@pytest.mark.parametrize("labels,expected", [
    (["positive", "positive", "negative"], "bullish"),
    (["negative", "negative", "positive"], "bearish"),
    (["positive", "negative"], "neutral"),
])
def test_aggregation(labels, expected):
    assert researcher.aggregate_sentiment([{"sentiment": label} for label in labels]) == expected
