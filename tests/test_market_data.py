from datetime import datetime, timezone

import pandas as pd
import pytest

from market_data import (
    fetch_bluesky_posts,
    fetch_gdelt_news,
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


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

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
