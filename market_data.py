"""Yahoo Finance news normalization and point-in-time feature helpers."""

from __future__ import annotations

import html
import io
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterable

import feedparser
import numpy as np
import pandas as pd
import requests


IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


def fetch_nse_equity_catalog() -> tuple[list[dict[str, str]], str]:
    """Fetch NSE EQ listings and attach live industry sectors where classified."""
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    industry_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BharatMarketResearch/1.0)",
        "Accept": "text/csv,application/csv,*/*",
    }
    try:
        response = requests.get(url, headers=headers, timeout=(4, 15))
        response.raise_for_status()
        industry_response = requests.get(industry_url, headers=headers, timeout=(4, 15))
        industry_response.raise_for_status()
    except requests.Timeout:
        return [], "NSE equity or sector catalog request timed out"
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return [], f"NSE equity or sector catalog request failed{f' (HTTP {status})' if status else ''}"

    try:
        frame = pd.read_csv(io.BytesIO(response.content), dtype=str)
        industry_frame = pd.read_csv(io.BytesIO(industry_response.content), dtype=str)
    except (ValueError, pd.errors.ParserError):
        return [], "NSE equity catalog response was not valid CSV"
    columns = {str(column).strip().upper(): column for column in frame.columns}
    symbol_column = columns.get("SYMBOL")
    name_column = columns.get("NAME OF COMPANY")
    series_column = columns.get("SERIES")
    if not symbol_column or not name_column or not series_column:
        return [], "NSE equity catalog is missing required columns"

    industry_columns = {str(column).strip().upper(): column for column in industry_frame.columns}
    industry_symbol_column = industry_columns.get("SYMBOL")
    industry_name_column = industry_columns.get("INDUSTRY")
    if not industry_symbol_column or not industry_name_column:
        return [], "NSE sector catalog is missing required columns"
    industries = {
        str(row[industry_symbol_column]).strip().upper(): str(row[industry_name_column]).strip()
        for _, row in industry_frame.iterrows()
        if pd.notna(row[industry_symbol_column]) and pd.notna(row[industry_name_column])
    }

    equities: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_column) or "").strip().upper()
        name = str(row.get(name_column) or "").strip()
        series = str(row.get(series_column) or "").strip().upper()
        if symbol and name and series == "EQ":
            industry = industries.get(symbol, "Outside NIFTY 500 / industry unavailable")
            equities.append(
                {
                    "symbol": f"{symbol}.NS",
                    "nse_symbol": symbol,
                    "name": name,
                    "series": series,
                    "industry": industry,
                }
            )
    equities.sort(key=lambda item: (item["name"].casefold(), item["nse_symbol"]))
    classified = sum(item["industry"] != "Outside NIFTY 500 / industry unavailable" for item in equities)
    return equities, f"OK; loaded {len(equities)} NSE EQ securities, {classified} with NIFTY 500 industry classifications"


def fetch_nse_commodity_etf_catalog() -> tuple[list[dict[str, str]], str]:
    """Fetch currently listed NSE ETFs whose underlying asset is a commodity."""
    url = "https://www.nseindia.com/api/etf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/market-data/exchange-traded-funds-etf",
    }
    try:
        response = requests.get(url, headers=headers, timeout=(4, 15))
    except requests.Timeout:
        return [], "NSE commodity ETF catalog request timed out"
    except requests.RequestException:
        return [], "NSE commodity ETF catalog network request failed"
    if response.status_code >= 400:
        return [], f"NSE commodity ETF catalog HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        return [], "NSE commodity ETF catalog response was not valid JSON"
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return [], "NSE commodity ETF catalog has an unexpected response format"

    # Map underlying asset descriptions (not individual fund symbols) to benchmarks.
    commodity_benchmarks = {
        "gold": ("Gold", "GC=F"),
        "silver": ("Silver", "SI=F"),
        "crude oil": ("Crude Oil", "CL=F"),
        "natural gas": ("Natural Gas", "NG=F"),
        "copper": ("Copper", "HG=F"),
    }
    funds: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in payload["data"]:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        asset = " ".join(str(entry.get("assets") or "").casefold().split())
        match = next(
            ((category, ticker) for commodity, (category, ticker) in commodity_benchmarks.items() if commodity in asset),
            None,
        )
        if not symbol or not match or symbol in seen:
            continue
        category, benchmark = match
        seen.add(symbol)
        funds.append(
            {
                "symbol": f"{symbol}.NS",
                "nse_symbol": symbol,
                "name": f"{symbol} — {entry.get('assets') or 'Commodity'}",
                "asset": str(entry.get("assets") or "Commodity"),
                "category": category,
                "benchmark": benchmark,
            }
        )
    funds.sort(key=lambda item: (item["asset"].casefold(), item["nse_symbol"]))
    return funds, f"OK; loaded {len(funds)} NSE commodity ETFs from the live ETF catalog"


def _request_json(url: str, params: dict[str, Any]) -> tuple[Any, str]:
    """Make a bounded API request and return a safe, non-secret status detail."""
    try:
        response = requests.get(url, params=params, timeout=(3, 10))
    except requests.Timeout:
        return None, "Request timed out"
    except requests.RequestException:
        return None, "Network request failed"

    if response.status_code >= 400:
        return None, f"HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        return None, "Response was not valid JSON"
    if not isinstance(payload, (dict, list)):
        return None, "Unexpected response format"
    return payload, "OK"


def _published_iso(value: Any, *, epoch: bool = False) -> str:
    if value in (None, ""):
        return ""
    if epoch:
        try:
            return datetime.fromtimestamp(float(value), UTC).isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            return ""
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.isoformat()


def _api_error_detail(value: Any, secret: str) -> str:
    """Return a short provider error while ensuring a configured key is hidden."""
    message = " ".join(str(value).split())[:240]
    if secret:
        message = message.replace(secret, "[redacted]")
    return f"Provider API error: {message or 'check key, quota, and provider response'}"


def fetch_alphavantage_news(
    api_key: str, symbol: str, limit: int = 20
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Alpha Vantage NEWS_SENTIMENT articles and ticker sentiment."""
    payload, detail = _request_json(
        "https://www.alphavantage.co/query",
        {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol.rsplit(".", 1)[0],
            "limit": min(max(limit, 1), 50),
            "apikey": api_key,
        },
    )
    if payload is None:
        return [], detail
    if not isinstance(payload, dict):
        return [], "Unexpected response format"
    for error_key in ("Error Message", "Note", "Information"):
        if payload.get(error_key):
            return [], _api_error_detail(payload[error_key], api_key)

    symbol_key = symbol.rsplit(".", 1)[0].upper()
    articles: list[dict[str, Any]] = []
    for item in payload.get("feed", [])[:limit]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        ticker_data = next(
            (
                entry
                for entry in item.get("ticker_sentiment", [])
                if str(entry.get("ticker") or "").upper() == symbol_key
            ),
            None,
        )
        score_data = ticker_data or {}
        try:
            sentiment = float(
                score_data.get("ticker_sentiment_score")
                or item.get("overall_sentiment_score", 0)
            )
            relevance = float(score_data.get("relevance_score", 0.8))
        except (TypeError, ValueError):
            sentiment, relevance = 0.0, 0.8
        summary = str(item.get("summary") or "")
        articles.append(
            {
                "title": title,
                "sentiment_text": f"{title}. {summary}" if summary else title,
                "url": str(item.get("url") or ""),
                "source": str(item.get("source") or "Alpha Vantage"),
                "published": _published_iso(item.get("time_published")),
                "sentiment": float(np.clip(sentiment, -1, 1)),
                "relevance": float(np.clip(relevance, 0, 1)),
            }
        )
    return articles, "OK" if articles else "Request succeeded; no matching articles returned"


def fetch_alphavantage_daily(
    api_key: str, symbol: str
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Alpha Vantage's free compact daily OHLCV series (up to 100 rows)."""
    if not api_key:
        return [], "API key is not configured"

    payload, detail = _request_json(
        "https://www.alphavantage.co/query",
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.strip(),
            "outputsize": "compact",
            "apikey": api_key,
        },
    )
    if payload is None:
        return [], detail
    if not isinstance(payload, dict):
        return [], "Unexpected response format"
    for error_key in ("Error Message", "Note", "Information"):
        if payload.get(error_key):
            return [], _api_error_detail(payload[error_key], api_key)

    time_series = payload.get("Time Series (Daily)")
    if not isinstance(time_series, dict):
        return [], "Daily series missing; verify the Alpha Vantage symbol"

    rows: list[dict[str, Any]] = []
    for date, values in time_series.items():
        if not isinstance(values, dict):
            continue
        try:
            rows.append(
                {
                    "Date": pd.Timestamp(date),
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Volume": int(float(values["5. volume"])),
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    rows.sort(key=lambda row: row["Date"])
    if not rows:
        return [], "Daily series contained no valid OHLCV rows"
    return rows, f"OK; {len(rows)} daily OHLCV rows (compact free endpoint)"


def _nse_published_iso(value: Any) -> str:
    """Parse NSE announcement timestamps, which are published in India time."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(IST)
    return stamp.tz_convert(UTC).isoformat()


def _clean_nse_text(value: Any) -> str:
    text = re.sub(r"<[^>]*>", " ", str(value or ""))
    return " ".join(html.unescape(text).split())


def fetch_nse_announcements_rss(
    symbol: str, company: str = "", limit: int = 30
) -> tuple[list[dict[str, Any]], str]:
    """Fetch the official NSE latest-announcements RSS and filter to one stock."""
    url = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BharatMarketResearch/1.0)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        response = requests.get(url, headers=headers, timeout=(4, 10))
    except requests.Timeout:
        return [], "Request timed out"
    except requests.RequestException:
        return [], "NSE RSS network request failed"
    if response.status_code >= 400:
        return [], f"NSE RSS HTTP {response.status_code}"

    parsed_feed = feedparser.parse(response.content)
    if parsed_feed.bozo and not parsed_feed.entries:
        return [], "NSE RSS response was not valid XML"

    bare_symbol = symbol.rsplit(".", 1)[0].strip()
    symbol_pattern = re.compile(
        rf"(?<![A-Z0-9]){re.escape(bare_symbol)}(?![A-Z0-9])", re.IGNORECASE
    )
    company_key = " ".join(company.casefold().split())
    articles: list[dict[str, Any]] = []
    for entry in parsed_feed.entries:
        description = _clean_nse_text(entry.get("summary") or entry.get("description"))
        title = _clean_nse_text(entry.get("title"))
        searchable = f"{title} {description}"
        if not symbol_pattern.search(searchable) and (
            not company_key or company_key not in searchable.casefold()
        ):
            continue

        subject = description.rsplit("|SUBJECT:", 1)[-1].strip() if "|SUBJECT:" in description else description
        company_title = title or company or bare_symbol
        headline = f"{company_title}: {subject}" if subject and subject != company_title else company_title
        articles.append(
            {
                "title": headline[:500],
                "sentiment_text": f"{headline}. {description}" if description else headline,
                "url": str(entry.get("link") or "https://www.nseindia.com/companies-listing/corporate-filings-announcements"),
                "source": "NSE India RSS Announcements",
                "published": _nse_published_iso(entry.get("published") or entry.get("updated")),
                "relevance": 1.0,
            }
        )
        if len(articles) >= min(max(limit, 1), 100):
            break

    detail = (
        f"Request succeeded; {len(articles)} matching NSE RSS announcements"
        if articles
        else "Request succeeded; no matching stock announcements in the current RSS window"
    )
    return articles, detail


def fetch_nse_company_announcements(
    symbol: str,
    from_date: Any = None,
    to_date: Any = None,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch official NSE company announcements for a historical date range."""
    today = datetime.now(IST).date()
    end_day = pd.Timestamp(to_date).date() if to_date is not None else today
    start_day = pd.Timestamp(from_date).date() if from_date is not None else end_day - timedelta(days=30)
    if start_day > end_day:
        return [], "Start date must be on or before end date"

    session = requests.Session()
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    try:
        session.get(
            "https://www.nseindia.com/",
            headers={**base_headers, "Accept": "text/html,application/xhtml+xml"},
            timeout=(4, 10),
        )
        response = session.get(
            "https://www.nseindia.com/api/corporate-announcements",
            params={
                "index": "equities",
                "symbol": symbol.rsplit(".", 1)[0],
                "from_date": start_day.strftime("%d-%m-%Y"),
                "to_date": end_day.strftime("%d-%m-%Y"),
            },
            headers={
                **base_headers,
                "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
            },
            timeout=(4, 12),
        )
    except requests.Timeout:
        return [], "NSE announcement request timed out"
    except requests.RequestException:
        return [], "NSE announcement network request failed"
    if response.status_code >= 400:
        return [], f"NSE announcement HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        return [], "NSE announcement response was not valid JSON"
    if isinstance(payload, dict):
        return [], _api_error_detail(payload.get("error") or payload.get("message"), "")
    if not isinstance(payload, list):
        return [], "Unexpected NSE announcement response format"

    articles: list[dict[str, Any]] = []
    for item in payload[: min(max(limit, 1), 500)]:
        item_symbol = str(item.get("symbol") or "").strip().upper()
        if item_symbol and item_symbol != symbol.rsplit(".", 1)[0].upper():
            continue
        company_name = _clean_nse_text(item.get("sm_name") or item_symbol or symbol)
        subject = _clean_nse_text(item.get("desc") or "Corporate announcement")
        headline = f"{company_name}: {subject}"
        articles.append(
            {
                "title": headline[:500],
                "sentiment_text": headline,
                "url": str(item.get("attchmntFile") or "https://www.nseindia.com/companies-listing/corporate-filings-announcements"),
                "source": "NSE India Corporate Announcements",
                "published": _nse_published_iso(item.get("an_dt")),
                "relevance": 1.0,
            }
        )
    detail = (
        f"OK; {len(articles)} NSE announcements from {start_day.isoformat()} to {end_day.isoformat()}"
        if articles
        else f"Request succeeded; no NSE announcements from {start_day.isoformat()} to {end_day.isoformat()}"
    )
    return articles, detail


def fetch_marketaux_news(
    api_token: str, symbol: str, limit: int = 20
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Marketaux articles and entity-level sentiment for the selected ticker."""
    payload, detail = _request_json(
        "https://api.marketaux.com/v1/news/all",
        {
            "symbols": symbol.rsplit(".", 1)[0],
            "filter_entities": "true",
            "language": "en",
            "limit": min(max(limit, 1), 50),
            "api_token": api_token,
        },
    )
    if payload is None:
        return [], detail
    if not isinstance(payload, dict):
        return [], "Unexpected response format"
    if payload.get("error"):
        error = payload["error"]
        detail = error.get("message", error) if isinstance(error, dict) else error
        return [], _api_error_detail(detail, api_token)

    symbol_key = symbol.rsplit(".", 1)[0].upper()
    articles: list[dict[str, Any]] = []
    for item in payload.get("data", [])[:limit]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        entity = next(
            (
                entry
                for entry in item.get("entities", [])
                if str(entry.get("symbol") or "").upper() == symbol_key
            ),
            None,
        )
        score = (entity or {}).get("sentiment_score")
        try:
            sentiment = float(score) if score is not None else None
        except (TypeError, ValueError):
            sentiment = None
        summary = str(item.get("description") or "")
        articles.append(
            {
                "title": title,
                "sentiment_text": f"{title}. {summary}" if summary else title,
                "url": str(item.get("url") or ""),
                "source": str(item.get("source") or "Marketaux"),
                "published": _published_iso(item.get("published_at")),
                "sentiment": float(np.clip(sentiment, -1, 1)) if sentiment is not None else None,
                "relevance": 0.85 if entity else 0.7,
            }
        )
    return articles, "OK" if articles else "Request succeeded; no matching articles returned"


def fetch_finnhub_news(
    api_key: str, symbol: str, limit: int = 20
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Finnhub company news for the previous seven days."""
    today = datetime.now(UTC).date()
    payload, detail = _request_json(
        "https://finnhub.io/api/v1/company-news",
        {
            "symbol": symbol.rsplit(".", 1)[0],
            "from": (today - timedelta(days=7)).isoformat(),
            "to": today.isoformat(),
            "token": api_key,
        },
    )
    if payload is None:
        return [], detail
    if not isinstance(payload, list):
        error = payload.get("error") if isinstance(payload, dict) else "unexpected response format"
        return [], _api_error_detail(error, api_key)

    articles: list[dict[str, Any]] = []
    for item in payload[:limit]:
        title = str(item.get("headline") or "").strip()
        if not title:
            continue
        summary = str(item.get("summary") or "")
        articles.append(
            {
                "title": title,
                "sentiment_text": f"{title}. {summary}" if summary else title,
                "url": str(item.get("url") or ""),
                "source": str(item.get("source") or "Finnhub"),
                "published": _published_iso(item.get("datetime"), epoch=True),
                "relevance": 0.85,
            }
        )
    return articles, "OK" if articles else "Request succeeded; no matching articles returned"


def fetch_finnhub_general_news(
    api_key: str, limit: int = 20
) -> tuple[list[dict[str, Any]], str]:
    """Fetch Finnhub's free general-market news feed (not company-specific)."""
    payload, detail = _request_json(
        "https://finnhub.io/api/v1/news",
        {"category": "general", "token": api_key},
    )
    if payload is None:
        return [], detail
    if not isinstance(payload, list):
        error = payload.get("error") if isinstance(payload, dict) else "unexpected response format"
        return [], _api_error_detail(error, api_key)

    articles: list[dict[str, Any]] = []
    for item in payload[:limit]:
        title = str(item.get("headline") or "").strip()
        if not title:
            continue
        summary = str(item.get("summary") or "")
        articles.append(
            {
                "title": title,
                "sentiment_text": f"{title}. {summary}" if summary else title,
                "url": str(item.get("url") or ""),
                "source": str(item.get("source") or "Finnhub General News"),
                "published": _published_iso(item.get("datetime"), epoch=True),
                # Broad market headlines provide context, not stock-specific evidence.
                "relevance": 0.25,
            }
        )
    return articles, "OK" if articles else "Request succeeded; no general-market articles returned"


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
    lookback_days: int | None = None,
) -> pd.Series:
    """Map sentiment to market closes, optionally using a trailing calendar window.

    News published after a session's 15:30 IST close is assigned to the next
    session, so historical rows never use information that was not yet public.
    When ``lookback_days`` is set, each close receives a relevance-weighted mean
    of items published in the trailing window, including that close.
    """
    if lookback_days is not None and lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")
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
        if lookback_days is not None and published < close_times[0] - pd.Timedelta(days=lookback_days):
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

    if lookback_days is None:
        scores = np.divide(
            weighted_sentiment,
            total_relevance,
            out=np.zeros_like(weighted_sentiment),
            where=total_relevance > 0,
        )
    else:
        close_index = pd.DatetimeIndex(close_times)
        weighted = pd.Series(weighted_sentiment, index=close_index)
        relevance = pd.Series(total_relevance, index=close_index)
        window = f"{lookback_days}D"
        weighted = weighted.rolling(window, closed="both", min_periods=1).sum()
        relevance = relevance.rolling(window, closed="both", min_periods=1).sum()
        scores = weighted.div(relevance.replace(0, np.nan)).fillna(0).to_numpy()
    return pd.Series(scores, index=price_index, name="NEWS")
