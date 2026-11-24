"""Multi-source news aggregation and deduplication across Finnhub and Alpaca News."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any

import finnhub
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

from src.analyst.models import NY_TIMEZONE, NewsItem, normalize_ny_datetime

logger = logging.getLogger(__name__)

_MACRO_POLITICAL_SOCIAL_KEYWORDS: tuple[str, ...] = (
    "tariff",
    "tariffs",
    "white house",
    "congress",
    "senate",
    "president",
    "election",
    "geopolit",
    "sanction",
    "export control",
    "china",
    "taiwan",
    "middle east",
    "oil",
    "opec",
    "fed",
    "fomc",
    "powell",
    "interest rate",
    "inflation",
    "cpi",
    "treasury",
    "antitrust",
    "doj",
    "ftc",
    "fda",
    "sec",
    "regulation",
    "regulatory",
    "subsidy",
    "defense",
    "pentagon",
    "boycott",
    "strike",
    "union",
    "consumer sentiment",
    "consumer spending",
    "social",
    "viral",
    "layoff",
    "tax",
)


def _normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must be a non-empty string")
    return cleaned


def _normalize_headline(headline: str) -> str:
    return re.sub(r"\s+", " ", headline.strip().lower())


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    return url.strip().rstrip("/").lower()


def _read_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_alpaca_articles(response: Any) -> list[Any]:
    """Extract article payloads from Alpaca NewsSet, raw dicts, or test doubles."""
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        if "news" in response and isinstance(response["news"], list):
            return response["news"]
        articles: list[Any] = []
        for value in response.values():
            if isinstance(value, list):
                articles.extend(value)
        return articles

    data = getattr(response, "data", None)
    if isinstance(data, dict):
        if "news" in data and isinstance(data["news"], list):
            return data["news"]
        articles = []
        for value in data.values():
            if isinstance(value, list):
                articles.extend(value)
        return articles

    news_attr = getattr(response, "news", None)
    if isinstance(news_attr, list):
        return news_attr
    return []


class NewsAggregator:
    """Collects, normalizes, and deduplicates ticker news from Finnhub and Alpaca."""

    def __init__(
        self,
        finnhub_client: Any | None = None,
        news_client: Any | None = None,
        finnhub_api_key: str | None = None,
        alpaca_api_key: str | None = None,
        alpaca_secret_key: str | None = None,
    ) -> None:
        """Initialize the aggregator with optional injected clients or environment keys."""
        if finnhub_client is not None:
            self.finnhub_client = finnhub_client
        else:
            resolved_finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
            self.finnhub_client = (
                finnhub.Client(api_key=resolved_finnhub_key)
                if resolved_finnhub_key
                else None
            )

        if news_client is not None:
            self.news_client = news_client
        else:
            resolved_alpaca_key = alpaca_api_key or os.getenv("ALPACA_API_KEY")
            resolved_alpaca_secret = alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")
            self.news_client = (
                NewsClient(api_key=resolved_alpaca_key, secret_key=resolved_alpaca_secret)
                if resolved_alpaca_key and resolved_alpaca_secret
                else None
            )
        self._macro_cache_ts: datetime | None = None
        self._macro_cache_items: list[NewsItem] = []

    def fetch_finnhub_news(
        self,
        symbol: str,
        start_ny: datetime,
        end_ny: datetime,
    ) -> list[NewsItem]:
        """Fetch and normalize company news from Finnhub."""
        if self.finnhub_client is None:
            return []

        normalized_symbol = _normalize_symbol(symbol)
        raw_items = self.finnhub_client.company_news(
            normalized_symbol,
            _from=start_ny.date().isoformat(),
            to=end_ny.date().isoformat(),
        )
        if not raw_items:
            return []

        items: list[NewsItem] = []
        for entry in raw_items:
            headline = str(_read_attr_or_key(entry, "headline", "") or "").strip()
            if not headline:
                continue
            source = str(_read_attr_or_key(entry, "source", "") or "finnhub").strip() or "finnhub"
            summary = str(_read_attr_or_key(entry, "summary", "") or "").strip()
            url = str(_read_attr_or_key(entry, "url", "") or "").strip()
            raw_dt = _read_attr_or_key(entry, "datetime", None)
            items.append(
                NewsItem(
                    headline=headline,
                    summary=summary,
                    source=source,
                    url=url,
                    published_at=normalize_ny_datetime(raw_dt),
                    provider="finnhub",
                )
            )
        return items

    def fetch_alpaca_news(
        self,
        symbol: str,
        start_ny: datetime,
        end_ny: datetime,
    ) -> list[NewsItem]:
        """Fetch and normalize symbol news from Alpaca News API."""
        if self.news_client is None:
            return []

        normalized_symbol = _normalize_symbol(symbol)
        request = NewsRequest(
            symbols=normalized_symbol,
            start=start_ny,
            end=end_ny,
        )
        response = self.news_client.get_news(request)
        articles = _extract_alpaca_articles(response)

        items: list[NewsItem] = []
        for article in articles:
            headline = str(_read_attr_or_key(article, "headline", "") or "").strip()
            if not headline:
                continue
            source = str(_read_attr_or_key(article, "source", "") or "alpaca").strip() or "alpaca"
            summary = str(_read_attr_or_key(article, "summary", "") or "").strip()
            url = str(_read_attr_or_key(article, "url", "") or "").strip()
            raw_dt = _read_attr_or_key(article, "created_at", None) or _read_attr_or_key(
                article, "updated_at", None
            )
            items.append(
                NewsItem(
                    headline=headline,
                    summary=summary,
                    source=source,
                    url=url,
                    published_at=normalize_ny_datetime(raw_dt),
                    provider="alpaca",
                )
            )
        return items

    @staticmethod
    def deduplicate(items: Iterable[NewsItem]) -> list[NewsItem]:
        """Deduplicate news items by normalized URL and headline, keeping earliest source."""
        seen_urls: set[str] = set()
        seen_headlines: set[str] = set()
        unique_items: list[NewsItem] = []

        # Process oldest-first so the original breaking article is preserved, then return newest-first.
        chronological = sorted(items, key=lambda item: item.published_at)
        for item in chronological:
            norm_url = _normalize_url(item.url)
            norm_headline = _normalize_headline(item.headline)

            if norm_url and norm_url in seen_urls:
                continue
            if norm_headline and norm_headline in seen_headlines:
                continue

            if norm_url:
                seen_urls.add(norm_url)
            if norm_headline:
                seen_headlines.add(norm_headline)
            unique_items.append(item)

        unique_items.sort(key=lambda item: item.published_at, reverse=True)
        return unique_items

    @staticmethod
    def distinct_sources(items: Sequence[NewsItem]) -> list[str]:
        """Return a sorted, case-insensitively deduplicated list of news sources."""
        canonical: dict[str, str] = {}
        for item in items:
            cleaned = item.source.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in canonical:
                canonical[key] = cleaned
        return sorted(canonical.values(), key=str.lower)

    def fetch_news(
        self,
        symbol: str,
        lookback_hours: int = 48,
        as_of: datetime | None = None,
    ) -> list[NewsItem]:
        """Aggregate, normalize, and deduplicate news from Finnhub and Alpaca."""
        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")

        normalized_symbol = _normalize_symbol(symbol)
        end_ny = normalize_ny_datetime(as_of) if as_of is not None else datetime.now(NY_TIMEZONE)
        start_ny = end_ny - timedelta(hours=lookback_hours)

        collected: list[NewsItem] = []
        for provider_name, fetcher in (
            ("finnhub", self.fetch_finnhub_news),
            ("alpaca", self.fetch_alpaca_news),
        ):
            try:
                collected.extend(fetcher(normalized_symbol, start_ny, end_ny))
            except Exception as exc:
                logger.warning(
                    "Failed fetching %s news for %s: %s",
                    provider_name,
                    normalized_symbol,
                    exc,
                )

        return self.deduplicate(collected)

    def fetch_macro_context_news(
        self,
        max_items: int = 5,
        as_of: datetime | None = None,
        cache_ttl_minutes: float = 30.0,
    ) -> list[NewsItem]:
        """Fetch and filter macro, political, regulatory, geopolitical, and social-trend news."""
        if max_items <= 0 or self.finnhub_client is None:
            return []
        general_fn = getattr(self.finnhub_client, "general_news", None)
        if not callable(general_fn):
            return []

        now_ny = normalize_ny_datetime(as_of) if as_of is not None else datetime.now(NY_TIMEZONE)
        if (
            self._macro_cache_ts is not None
            and (now_ny - self._macro_cache_ts).total_seconds() < cache_ttl_minutes * 60.0
        ):
            return list(self._macro_cache_items[:max_items])

        try:
            raw_articles = general_fn("general", min_id=0)
        except Exception as exc:
            logger.debug("Macro general_news fetch skipped: %s", exc)
            return []

        if not isinstance(raw_articles, list):
            return []

        matched: list[NewsItem] = []
        for entry in raw_articles:
            if not isinstance(entry, dict):
                continue
            headline = str(entry.get("headline") or "").strip()
            if not headline:
                continue
            summary = str(entry.get("summary") or "").strip()
            combined_lower = f"{headline} {summary}".lower()
            if not any(kw in combined_lower for kw in _MACRO_POLITICAL_SOCIAL_KEYWORDS):
                continue
            source = str(entry.get("source") or "finnhub").strip() or "finnhub"
            url = str(entry.get("url") or "").strip()
            raw_dt = entry.get("datetime")
            matched.append(
                NewsItem(
                    headline=headline,
                    summary=summary,
                    source=source,
                    url=url,
                    published_at=normalize_ny_datetime(raw_dt),
                    provider="finnhub",
                )
            )

        deduped = self.deduplicate(matched)
        self._macro_cache_ts = now_ny
        self._macro_cache_items = deduped
        return list(deduped[:max_items])
