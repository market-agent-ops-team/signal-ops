import json
from html.parser import HTMLParser

import pytest

from tests.test_api import api_setup, client


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.mark.parametrize("path,title", [("/", "Signals."), ("/examples", "Example reports"),
    ("/methodology", "Methodology"), ("/examples/divergence", "Signal Divergence"),
    ("/examples/bullish-alignment", "Signals Aligned"), ("/examples/bearish-alignment", "Signals Aligned")])
def test_editorial_pages_do_not_run_analysis(api_setup, path, title):
    browser, graph, store, _ = api_setup
    response = browser.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert title in response.text
    assert not any(tag == "script" for tag, _ in Document(response.text).elements)
    graph.invoke.assert_not_called()
    store.get.assert_not_called()


@pytest.mark.parametrize("query,field,value", [({}, "ticker", ""),
    ({"ticker": '><script>alert(1)</script>', "target_date": "2026-09-11"}, "ticker", '><script>alert(1)</script>'),
    ({"ticker": "INFY", "target_date": "2026-02-30"}, "target_date", "2026-02-30"),
    ({"ticker": "INFY.BO", "target_date": "2026-09-11"}, "ticker", "INFY.BO")])
def test_report_validation_retains_values_with_linked_errors(api_setup, query, field, value):
    browser, graph, store, _ = api_setup
    response = browser.get("/report", params=query)
    assert response.status_code == 422
    elements = Document(response.text).elements
    assert any(tag == "input" and attrs.get("name") == field and attrs.get("value") == value
               and attrs.get("aria-invalid") == "true" for tag, attrs in elements)
    assert any(tag == "a" and attrs.get("href") == f"#{field}" for tag, attrs in elements)
    assert not any(tag == "script" for tag, _ in elements)
    graph.invoke.assert_not_called()
    store.get.assert_not_called()


def test_html_and_json_share_normalized_cache_and_report(api_setup):
    browser, graph, store, result = api_setup
    first = browser.get("/report", params={"ticker": " infy.ns ", "target_date": "2026-09-11"})
    assert first.status_code == 200
    assert "34.8%" in first.text and "Signals conflict" in first.text
    assert 'value="INFY"' in first.text
    encoded = store.setex.call_args.args[2]
    store.get.return_value = encoded
    graph.invoke.reset_mock()
    payload = browser.post("/analyze", json={"ticker": "INFY", "target_date": "2026-09-11"}).json()
    assert payload == json.loads(encoded)
    second = browser.get("/report?ticker=INFY&target_date=2026-09-11")
    assert second.text == first.text
    graph.invoke.assert_not_called()


@pytest.mark.parametrize("error,status", [(ValueError("secret"), 422),
    (FileNotFoundError("secret"), 503), (RuntimeError("secret"), 500)])
def test_report_errors_offer_retained_retry_form(api_setup, error, status):
    browser, graph, _, _ = api_setup
    graph.invoke.side_effect = error
    response = browser.get("/report?ticker=INFY&target_date=2026-09-11")
    assert response.status_code == status
    assert "secret" not in response.text
    assert 'value="INFY"' in response.text and 'action="/report"' in response.text


def test_untrusted_report_and_reasoning_are_safe(api_setup):
    browser, _, _, result = api_setup
    result["final_report"] = '# News\n<script>alert(1)</script>\n\n[unsafe](javascript:alert(1))\n\n[safe](https://example.com)\n\n![pixel](https://example.com/pixel)'
    result["analyst_reasoning"] = '<img src=x onerror="alert(1)">'
    response = browser.get("/report?ticker=INFY&target_date=2026-09-11")
    assert response.status_code == 200
    elements = Document(response.text).elements
    assert not any(tag in {"script", "img"} for tag, _ in elements)
    assert not any(attrs.get("href", "").startswith("javascript:") for _, attrs in elements)
    assert any(attrs.get("href") == "https://example.com" for _, attrs in elements)
    assert "&lt;script&gt;" in response.text and "&lt;img" in response.text


@pytest.mark.parametrize("preference", ["light", "dark", "system"])
def test_theme_cookie_and_return(client, preference):
    client.cookies.clear()
    response = client.post("/theme", data={"theme": preference, "return_to": "/report?ticker=INFY&target_date=2026-09-11"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/report?ticker=INFY&target_date=2026-09-11"
    cookie = response.headers["set-cookie"]
    assert f"theme={preference}" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    html = client.get("/").text
    assert (f'data-theme="{preference}"' in html) is (preference != "system")
    client.cookies.clear()


@pytest.mark.parametrize("destination", ["https://evil.test", "//evil.test", "/\\evil.test", "%2f%2fevil.test", "/%5cevil.test", "/%0aevil", "javascript:alert(1)"])
def test_theme_rejects_external_or_ambiguous_returns(client, destination):
    response = client.post("/theme", data={"theme": "light", "return_to": destination}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    client.cookies.clear()


def test_invalid_theme_is_html_error_without_cookie(client):
    response = client.post("/theme", data={"theme": "invalid"})
    assert response.status_code == 422
    assert "text/html" in response.headers["content-type"]
    assert "set-cookie" not in response.headers


def test_unknown_example_is_html_404(client):
    response = client.get("/examples/README")
    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "Back to workspace" in response.text


@pytest.mark.parametrize("status,sentiment,trend,relationship", [
    ("available", "bullish", "bullish", "Aligned"),
    ("available", "neutral", "neutral", "No directional alignment"),
    ("available", "neutral", "bearish", "No directional alignment"),
    ("empty", None, "bullish", "No directional alignment"),
    ("unavailable", None, "bearish", "No directional alignment"),
])
def test_report_does_not_invent_news_or_directional_alignment(api_setup, status, sentiment, trend, relationship):
    browser, _, _, result = api_setup
    result.update(news_status=status, news_sentiments=sentiment, ml_trend=trend, divergence_flag=False)
    response = browser.get("/report?ticker=INFY&target_date=2026-09-11")
    assert response.status_code == 200
    assert f"<strong>{relationship}</strong>" in response.text
    if sentiment is None:
        assert "<strong>Not observed</strong>" in response.text
    assert f"Status: {status}" in response.text


def test_real_graph_news_scripts_cannot_become_html(monkeypatch, client):
    from unittest.mock import Mock
    from agents import graph, researcher_agent
    from api import cache

    store = Mock()
    store.get.return_value = None
    monkeypatch.setattr(cache, "get_redis_client", lambda: store)
    monkeypatch.setattr(graph, "predict_trend", lambda *args: {"ml_trend": "bearish", "ml_confidence": 0.35})
    monkeypatch.setenv("NEWS_API_KEY", "test-key")
    news = Mock()
    news.json.return_value = {"status": "ok", "articles": [{
        "title": 'Infosys <script>alert(1)</script> excellent growth',
        "description": "Strong profits", "source": {"name": '<img src=x onerror="alert(1)">'},
        "publishedAt": "2026-09-10T10:00:00Z"}]}
    monkeypatch.setattr(researcher_agent.requests, "get", lambda *args, **kwargs: news)
    response = client.get("/report?ticker=INFY&target_date=2026-09-11")
    assert response.status_code == 200
    assert "&lt;script&gt;" in response.text and "&lt;img" in response.text
    assert not any(tag in {"script", "img"} for tag, _ in Document(response.text).elements)


@pytest.mark.parametrize("path,content_type", [("/static/css/app.css", "text/css"),
    ("/static/fonts/playfair-display-latin-wght-normal.woff2", "font/woff2"),
    ("/static/fonts/source-serif-4-latin-wght-normal.woff2", "font/woff2"),
    ("/static/fonts/jetbrains-mono-latin-wght-normal.woff2", "font/woff2")])
def test_compiled_assets_are_served(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_type)
    assert len(response.content) > 1000


def test_report_metadata_stays_separate_from_section_headings(client):
    response = client.get("/examples/divergence")
    assert '<p><strong>Ticker:</strong> <code>INFY</code><br' in response.text
    assert '<strong>Date:</strong> 2026-09-11</p>' in response.text
    assert '<h3>Technical Signal</h3>' in response.text


def test_workspace_ticker_input_offers_symbol_suggestions(client):
    elements = Document(client.get("/").text).elements
    ticker = next(attrs for tag, attrs in elements if tag == "input" and attrs.get("name") == "ticker")
    assert ticker.get("list") == "ticker-suggestions"
    assert any(tag == "datalist" and attrs.get("id") == "ticker-suggestions" for tag, attrs in elements)
    assert '<option value="INFY">Infosys</option>' in client.get("/").text
    assert '<option value="TECHM">' in client.get("/").text


def test_every_suggestion_is_a_valid_ticker(client):
    from agents.researcher_agent import normalize_ticker

    text = client.get("/").text
    for value in ("INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "RELIANCE"):
        assert f'<option value="{value}">' in text
        normalize_ticker(value)


def test_workspace_date_input_is_a_bounded_calendar(client):
    from datetime import date

    from ml.features.indicators import V1_HISTORY_START

    elements = Document(client.get("/").text).elements
    target = next(attrs for tag, attrs in elements if tag == "input" and attrs.get("name") == "target_date")
    assert target.get("type") == "date"
    assert target.get("min") == V1_HISTORY_START
    assert target.get("max") == date.today().isoformat()
