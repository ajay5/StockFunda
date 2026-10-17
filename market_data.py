"""Yahoo Finance news normalization and point-in-time feature helpers."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


def normalize_yahoo_news(items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize current Yahoo Finance news payloads, including legacy entries."""
    normalized: list[dict[str, str]] = []
    for item in items:
        content = item.get("content") or item
        title = str(content.get("title") or item.get("title") or "").strip()
        if not title:
            continue

        published = (
            content.get("pubDate")
            or content.get("displayTime")
            or item.get("published")
        )
        epoch = item.get("providerPublishTime")
        if not published and isinstance(epoch, (int, float)):
            published = datetime.fromtimestamp(epoch, UTC).isoformat()
        if published:
            parsed = pd.to_datetime(published, utc=True, errors="coerce")
            published = "" if pd.isna(parsed) else parsed.isoformat()
        else:
            published = ""

        provider = content.get("provider") or item.get("publisher") or {}
        source = provider.get("displayName") if isinstance(provider, dict) else provider
        canonical = content.get("canonicalUrl") or item.get("link") or {}
        if isinstance(canonical, dict):
            url = canonical.get("url") or ""
        else:
            url = str(canonical)

        normalized.append(
            {
                "title": title,
                "url": str(url),
                "source": str(source or "Yahoo Finance"),
                "published": published,
            }
        )
    return normalized


def fetch_gdelt_news(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch recent global news from the free GDELT DOC API."""
    try:
        response = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": min(max(limit, 1), 50),
                "sort": "HybridRel",
            },
            timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    articles: list[dict[str, Any]] = []
    for item in payload.get("articles", []):
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        published = pd.to_datetime(item.get("seendate"), utc=True, errors="coerce")
        articles.append(
            {
                "title": title,
                "url": str(item.get("url") or ""),
                "source": str(item.get("domain") or "GDELT"),
                "published": "" if pd.isna(published) else published.isoformat(),
                "relevance": 0.85,
            }
        )
    return articles


def fetch_bluesky_posts(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search recent public Bluesky posts through its public AppView API."""
    try:
        response = requests.get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={"q": query, "limit": min(max(limit, 1), 100), "sort": "latest"},
            timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    posts: list[dict[str, Any]] = []
    for item in payload.get("posts", []):
        record = item.get("record") or {}
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        author = item.get("author") or {}
        handle = str(author.get("handle") or "")
        uri = str(item.get("uri") or "")
        post_id = uri.rsplit("/", 1)[-1]
        profile = handle or str(author.get("did") or "")
        url = f"https://bsky.app/profile/{profile}/post/{post_id}" if profile and post_id else ""
        posts.append(
            {
                "title": text[:500],
                "url": url,
                "source": f"Bluesky @{handle}" if handle else "Bluesky",
                "published": str(record.get("createdAt") or item.get("indexedAt") or ""),
                "relevance": 0.6,
            }
        )
    return posts


def fetch_reddit_posts(
    query: str,
    access_token: str,
    user_agent: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search Reddit's official API using an OAuth bearer token."""
    if not access_token:
        return []
    try:
        response = requests.get(
            "https://oauth.reddit.com/search.json",
            params={"q": query, "sort": "new", "t": "week", "limit": min(max(limit, 1), 100)},
            headers={"Authorization": f"Bearer {access_token}", "User-Agent": user_agent},
            timeout=(3, 8),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    posts: list[dict[str, Any]] = []
    children = payload.get("data", {}).get("children", [])
    for child in children:
        item = child.get("data") or {}
        title = str(item.get("title") or "").strip()
        body = str(item.get("selftext") or "").strip()
        if not title:
            continue
        created = item.get("created_utc")
        try:
            published = datetime.fromtimestamp(float(created), UTC).isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            published = ""
        subreddit = str(item.get("subreddit") or "reddit")
        permalink = str(item.get("permalink") or "")
        posts.append(
            {
                "title": title,
                "sentiment_text": f"{title}. {body[:1000]}" if body else title,
                "url": f"https://www.reddit.com{permalink}" if permalink else "",
                "source": f"Reddit r/{subreddit}",
                "published": published,
                "relevance": 0.6,
            }
        )
    return posts


def point_in_time_news_feature(
    price_index: pd.Index,
    news_rows: Iterable[dict[str, Any]],
) -> pd.Series:
    """Map published sentiment to the first Indian market close after publication.

    News published after a session's 15:30 IST close is assigned to the next
    session, so historical rows never use information that was not yet public.
    """
    dates = pd.DatetimeIndex(price_index).date
    close_times = pd.DatetimeIndex(
        [datetime.combine(day, time(15, 30), tzinfo=IST).astimezone(UTC) for day in dates]
    )
    weighted_sentiment = np.zeros(len(dates), dtype=float)
    total_relevance = np.zeros(len(dates), dtype=float)

    if not len(close_times):
        return pd.Series(weighted_sentiment, index=price_index, name="NEWS")

    for row in news_rows:
        published = pd.to_datetime(row.get("published_at"), utc=True, errors="coerce")
        if pd.isna(published):
            continue
        session = int(close_times.searchsorted(published, side="left"))
        if session >= len(dates):
            continue
        try:
            sentiment = float(row.get("sentiment", 0.0))
            relevance = float(row.get("relevance", 1.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sentiment) or not np.isfinite(relevance) or relevance <= 0:
            continue
        weighted_sentiment[session] += float(np.clip(sentiment, -1, 1)) * float(
            np.clip(relevance, 0, 1)
        )
        total_relevance[session] += float(np.clip(relevance, 0, 1))

    scores = np.divide(
        weighted_sentiment,
        total_relevance,
        out=np.zeros_like(weighted_sentiment),
        where=total_relevance > 0,
    )
    return pd.Series(scores, index=price_index, name="NEWS")
