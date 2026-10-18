from datetime import datetime, timezone
from datetime import date

import pandas as pd
import pytest

from market_data import (
    fetch_alphavantage_daily,
    fetch_alphavantage_news,
    fetch_bluesky_posts,
    fetch_finnhub_news,
    fetch_finnhub_general_news,
    fetch_gdelt_news,
    fetch_marketaux_news,
    fetch_nse_announcements_rss,
    fetch_nse_company_announcements,
    fetch_reddit_posts,
    normalize_yahoo_news,
    point_in_time_news_feature,
)


def test_normalize_yahoo_news_current_payload():
    items = [
        {
            "content": {
                "title": "Company reports results",
                "pubDate": "2026-10-16T10:00:00Z",
                "provider": {"displayName": "Yahoo Finance"},
                "canonicalUrl": {"url": "https://finance.yahoo.com/story"},
            }
        },
        {"title": "Legacy story", "providerPublishTime": 1792144800, "link": "https://example.com"},
    ]

    normalized = normalize_yahoo_news(items)

    assert len(normalized) == 2
    assert normalized[0]["title"] == "Company reports results"
    assert normalized[0]["published"] == "2026-10-16T10:00:00+00:00"
    assert normalized[0]["source"] == "Yahoo Finance"
    assert normalized[0]["url"] == "https://finance.yahoo.com/story"
    assert normalized[1]["source"] == "Yahoo Finance"
    assert normalized[1]["published"] == datetime.fromtimestamp(
        1792144800, timezone.utc
    ).isoformat()


def test_historical_news_is_assigned_without_lookahead():
    sessions = pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20"])
    rows = [
        {"published_at": "2026-10-16T09:00:00Z", "sentiment": 0.5, "relevance": 1},
        # 16:00 IST is after Friday's close, so it belongs to Monday.
        {"published_at": "2026-10-16T10:30:00Z", "sentiment": -1, "relevance": 0.5},
        {"published_at": "2026-10-19T10:00:00Z", "sentiment": 0.25, "relevance": 1},
    ]

    feature = point_in_time_news_feature(sessions, rows)

    assert feature.iloc[0] == pytest.approx(0.5)
    assert feature.iloc[1] == pytest.approx((-1 * 0.5 + 0.25) / 1.5)
    assert feature.iloc[2] == pytest.approx(0)


def test_historical_news_ignores_missing_timestamps_and_bad_values():
    sessions = pd.to_datetime(["2026-10-16"])
    rows = [
        {"published_at": "not a date", "sentiment": 1, "relevance": 1},
        {"published_at": "2026-10-16T08:00:00Z", "sentiment": "bad", "relevance": 1},
        {"published_at": "2026-10-16T08:00:00Z", "sentiment": 1, "relevance": 0},
    ]

    assert point_in_time_news_feature(sessions, rows).iloc[0] == pytest.approx(0)


def test_point_in_time_news_feature_supports_trailing_calendar_window():
    sessions = pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20"])
    rows = [
        {"published_at": "2026-10-16T08:00:00Z", "sentiment": 1, "relevance": 1},
        {"published_at": "2026-10-19T08:00:00Z", "sentiment": -1, "relevance": 1},
    ]

    feature = point_in_time_news_feature(sessions, rows, lookback_days=2)

    assert feature.tolist() == pytest.approx([1, -1, -1])


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.content = payload if isinstance(payload, (str, bytes)) else b""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_gdelt_adapter_normalizes_recent_articles(monkeypatch):
    monkeypatch.setattr(
        "market_data.requests.get",
        lambda *args, **kwargs: FakeResponse(
            {
                "articles": [
                    {
                        "title": "Market report",
                        "url": "https://news.example/report",
                        "domain": "news.example",
                        "seendate": "20261017T100000Z",
                    }
                ]
            }
        ),
    )

    articles = fetch_gdelt_news("TCS", limit=10)

    assert articles[0]["title"] == "Market report"
    assert articles[0]["source"] == "news.example"
    assert articles[0]["published"] == "2026-10-17T10:00:00+00:00"
    assert articles[0]["relevance"] == pytest.approx(0.85)


def test_bluesky_adapter_builds_public_post_link(monkeypatch):
    monkeypatch.setattr(
        "market_data.requests.get",
        lambda *args, **kwargs: FakeResponse(
            {
                "posts": [
                    {
                        "uri": "at://did:plc:123/app.bsky.feed.post/post123",
                        "record": {
                            "text": "Company outlook improves",
                            "createdAt": "2026-10-17T10:00:00Z",
                        },
                        "author": {"handle": "investor.example"},
                    }
                ]
            }
        ),
    )

    posts = fetch_bluesky_posts("company stock")

    assert posts[0]["source"] == "Bluesky @investor.example"
    assert posts[0]["url"] == "https://bsky.app/profile/investor.example/post/post123"
    assert posts[0]["published"] == "2026-10-17T10:00:00Z"


def test_reddit_adapter_requires_token_and_normalizes_posts(monkeypatch):
    monkeypatch.setattr(
        "market_data.requests.get",
        lambda *args, **kwargs: FakeResponse(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "TCS discussion",
                                "selftext": "Quarterly results look strong.",
                                "created_utc": 1792231200,
                                "subreddit": "IndianStockMarket",
                                "permalink": "/r/IndianStockMarket/comments/example/",
                            }
                        }
                    ]
                }
            }
        ),
    )

    assert fetch_reddit_posts("TCS", "", "test-agent") == []
    posts = fetch_reddit_posts("TCS", "token", "test-agent")

    assert posts[0]["source"] == "Reddit r/IndianStockMarket"
    assert posts[0]["url"].endswith("/r/IndianStockMarket/comments/example/")
    assert "Quarterly results" in posts[0]["sentiment_text"]


def test_alphavantage_adapter_uses_ticker_sentiment(monkeypatch):
    monkeypatch.setattr(
        "market_data._request_json",
        lambda *args, **kwargs: (
            {
                "feed": [
                    {
                        "title": "TCS earnings",
                        "summary": "Revenue improved.",
                        "url": "https://example.com/tcs",
                        "source": "Example News",
                        "time_published": "20261018T100000",
                        "ticker_sentiment": [
                            {
                                "ticker": "TCS",
                                "ticker_sentiment_score": "0.42",
                                "relevance_score": "0.9",
                            }
                        ],
                    }
                ]
            },
            "OK",
        ),
    )

    articles, status = fetch_alphavantage_news("test-key", "TCS.NS")

    assert status == "OK"
    assert articles[0]["sentiment"] == pytest.approx(0.42)
    assert articles[0]["relevance"] == pytest.approx(0.9)
    assert "Revenue improved" in articles[0]["sentiment_text"]


def test_marketaux_adapter_uses_matching_entity_sentiment(monkeypatch):
    monkeypatch.setattr(
        "market_data._request_json",
        lambda *args, **kwargs: (
            {
                "data": [
                    {
                        "title": "TCS market update",
                        "description": "Company outlook changes.",
                        "url": "https://example.com/marketaux",
                        "source": "Example News",
                        "published_at": "2026-10-18T10:00:00Z",
                        "entities": [{"symbol": "TCS", "sentiment_score": -0.3}],
                    }
                ]
            },
            "OK",
        ),
    )

    articles, status = fetch_marketaux_news("test-token", "TCS.NS")

    assert status == "OK"
    assert articles[0]["sentiment"] == pytest.approx(-0.3)
    assert articles[0]["relevance"] == pytest.approx(0.85)


def test_finnhub_adapter_converts_epoch_and_api_errors(monkeypatch):
    monkeypatch.setattr(
        "market_data._request_json",
        lambda *args, **kwargs: (
            [{"headline": "TCS filing", "datetime": 1792317600, "source": "Finnhub"}],
            "OK",
        ),
    )

    articles, status = fetch_finnhub_news("test-key", "TCS.NS")

    assert status == "OK"
    assert articles[0]["source"] == "Finnhub"
    assert articles[0]["published"] == datetime.fromtimestamp(
        1792317600, timezone.utc
    ).isoformat()

    monkeypatch.setattr(
        "market_data._request_json",
        lambda *args, **kwargs: ({"Note": "quota message"}, "OK"),
    )
    articles, status = fetch_alphavantage_news("test-key", "TCS.NS")

    assert articles == []
    assert "API error" in status

    monkeypatch.setattr(
        "market_data._request_json",
        lambda *args, **kwargs: ({"Error Message": "Invalid key test-secret"}, "OK"),
    )
    articles, status = fetch_alphavantage_news("test-secret", "TCS.NS")

    assert articles == []
    assert "Invalid key" in status
    assert "test-secret" not in status
    assert "[redacted]" in status


def test_finnhub_general_news_adapter_uses_free_market_feed(monkeypatch):
    requested = {}

    def fake_request(url, params):
        requested["url"] = url
        requested["params"] = params
        return (
            [
                {
                    "headline": "Global markets rise",
                    "summary": "Broad market context.",
                    "datetime": 1792317600,
                    "source": "Finnhub",
                    "url": "https://example.com/markets",
                }
            ],
            "OK",
        )

    monkeypatch.setattr("market_data._request_json", fake_request)

    articles, status = fetch_finnhub_general_news("test-key", limit=5)

    assert status == "OK"
    assert requested["url"].endswith("/news")
    assert requested["params"]["category"] == "general"
    assert articles[0]["title"] == "Global markets rise"
    assert articles[0]["relevance"] == pytest.approx(0.25)


def test_alphavantage_daily_uses_free_compact_endpoint_and_sorts_rows(monkeypatch):
    requested = {}

    def fake_request(url, params):
        requested["url"] = url
        requested["params"] = params
        return (
            {
                "Meta Data": {},
                "Time Series (Daily)": {
                    "2026-10-19": {
                        "1. open": "102.0",
                        "2. high": "105.0",
                        "3. low": "101.0",
                        "4. close": "104.0",
                        "5. volume": "1200",
                    },
                    "2026-10-16": {
                        "1. open": "99.0",
                        "2. high": "103.0",
                        "3. low": "98.0",
                        "4. close": "102.0",
                        "5. volume": "900",
                    },
                },
            },
            "OK",
        )

    monkeypatch.setattr("market_data._request_json", fake_request)

    rows, status = fetch_alphavantage_daily("test-key", "RELIANCE.BSE")

    assert requested["url"].endswith("/query")
    assert requested["params"]["function"] == "TIME_SERIES_DAILY"
    assert requested["params"]["symbol"] == "RELIANCE.BSE"
    assert requested["params"]["outputsize"] == "compact"
    assert [row["Date"].strftime("%Y-%m-%d") for row in rows] == [
        "2026-10-16",
        "2026-10-19",
    ]
    assert rows[0]["Close"] == pytest.approx(102.0)
    assert rows[0]["Volume"] == 900
    assert "2 daily OHLCV rows" in status


def test_alphavantage_daily_reports_quota_error_without_exposing_key(monkeypatch):
    monkeypatch.setattr(
        "market_data._request_json",
        lambda *args, **kwargs: (
            {"Note": "API call frequency exceeded for test-secret"},
            "OK",
        ),
    )

    rows, status = fetch_alphavantage_daily("test-secret", "IBM")

    assert rows == []
    assert "Provider API error" in status
    assert "frequency exceeded" in status
    assert "test-secret" not in status


def test_nse_rss_filters_to_selected_symbol_and_converts_ist(monkeypatch):
    rss = b"""<?xml version='1.0'?>
    <rss version='2.0'><channel>
      <item><title>Tata Consultancy Services Limited</title>
        <link>https://nsearchives.nseindia.com/filing/tcs.pdf</link>
        <description>TCS : TCS Limited has informed the Exchange |SUBJECT: Board Meeting</description>
        <pubDate>16-Oct-2026 15:45:00</pubDate></item>
      <item><title>Other Company Limited</title><description>OTHER : unrelated update</description>
        <pubDate>16-Oct-2026 12:00:00</pubDate></item>
    </channel></rss>"""
    monkeypatch.setattr("market_data.requests.get", lambda *args, **kwargs: FakeResponse(rss))

    articles, status = fetch_nse_announcements_rss("TCS.NS", "TCS")

    assert len(articles) == 1
    assert articles[0]["source"] == "NSE India RSS Announcements"
    assert articles[0]["url"].endswith("tcs.pdf")
    assert articles[0]["published"] == "2026-10-16T10:15:00+00:00"
    assert "Board Meeting" in articles[0]["title"]
    assert "1 matching" in status


def test_nse_historical_adapter_uses_selected_symbol_and_date_range(monkeypatch):
    requested = []

    class FakeSession:
        def get(self, url, **kwargs):
            requested.append((url, kwargs))
            if url.endswith("/"):
                return FakeResponse({})
            return FakeResponse(
                [
                    {
                        "an_dt": "16-Oct-2026 16:29:00",
                        "symbol": "TCS",
                        "sm_name": "Tata Consultancy Services Limited",
                        "desc": "Analysts/Institutional Investor Meet/Con. Call Updates",
                        "attchmntFile": "https://nsearchives.nseindia.com/corporate/tcs.pdf",
                    }
                ]
            )

    monkeypatch.setattr("market_data.requests.Session", FakeSession)

    articles, status = fetch_nse_company_announcements(
        "TCS.NS", date(2026, 10, 1), date(2026, 10, 18)
    )

    assert len(requested) == 2
    assert requested[1][0].endswith("/api/corporate-announcements")
    assert requested[1][1]["params"] == {
        "index": "equities",
        "symbol": "TCS",
        "from_date": "01-10-2026",
        "to_date": "18-10-2026",
    }
    assert articles[0]["source"] == "NSE India Corporate Announcements"
    assert articles[0]["published"] == "2026-10-16T10:59:00+00:00"
    assert articles[0]["url"].endswith("tcs.pdf")
    assert "1 NSE announcements" in status
