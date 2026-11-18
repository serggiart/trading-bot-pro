"""Financial Modeling Prep (FMP) client with SQLite daily caching and quota enforcement."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import requests

from src.storage import DEFAULT_FMP_DAILY_LIMIT, TradingAuditStore
from src.universe.models import FundamentalData


class FmpRateLimitError(RuntimeError):
    """Raised when the daily FMP API request quota is exhausted."""


class FmpApiError(RuntimeError):
    """Raised when FMP returns an unexpected or incomplete response."""


def _sanitize_error_message(message: str) -> str:
    """Redact API keys from URLs or HTTP error strings before logging/raising."""
    return re.sub(r"([?&]apikey=)[^&\s'\"]+", r"\1***", message)


def _normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must be a non-empty string")
    return cleaned


def _extract_first_record(payload: Any) -> dict[str, Any] | None:
    """Return the first dict record from a list or dict FMP JSON payload."""
    if isinstance(payload, list):
        if not payload:
            return None
        first = payload[0]
        return first if isinstance(first, dict) else None
    if isinstance(payload, dict):
        return payload
    return None


def _parse_iso_date(raw: Any) -> str | None:
    """Normalize a date string into ISO YYYY-MM-DD format if valid."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if len(cleaned) < 10:
        return None
    try:
        return date.fromisoformat(cleaned[:10]).isoformat()
    except ValueError:
        return None


def _parse_free_cash_flow(payload: Any, symbol: str) -> float:
    record = _extract_first_record(payload)
    if record is None:
        raise FmpApiError(f"Empty cash flow statement returned by FMP for {symbol}")
    for key in ("freeCashFlow", "free_cash_flow"):
        val = record.get(key)
        if val is not None:
            return float(val)
    raise FmpApiError(f"Missing freeCashFlow in FMP cash flow payload for {symbol}")


def _parse_debt_to_equity(payload: Any, symbol: str) -> float:
    record = _extract_first_record(payload)
    if record is None:
        raise FmpApiError(f"Empty ratios payload returned by FMP for {symbol}")
    for key in ("debtToEquityRatio", "debtEquityRatio", "debtToEquity", "debt_to_equity"):
        val = record.get(key)
        if val is not None:
            return float(val)

    total_debt = record.get("totalDebt")
    total_equity = record.get("totalStockholdersEquity")
    if total_debt is not None and total_equity is not None and float(total_equity) != 0.0:
        return float(total_debt) / float(total_equity)

    raise FmpApiError(f"Missing debtEquityRatio in FMP ratios payload for {symbol}")


def _parse_earnings_dates(payload: Any) -> list[str]:
    if isinstance(payload, dict) and isinstance(payload.get("earnings_dates"), list):
        candidates = payload["earnings_dates"]
    elif isinstance(payload, list):
        candidates = [
            item.get("date")
            for item in payload
            if isinstance(item, dict) and item.get("date") is not None
        ]
    else:
        candidates = []

    dates: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        parsed = _parse_iso_date(raw)
        if parsed is not None and parsed not in seen:
            seen.add(parsed)
            dates.append(parsed)
    return dates


class FmpClient:
    """FMP fundamentals client backed by TradingAuditStore daily cache & 250/day cap."""

    def __init__(
        self,
        store: TradingAuditStore,
        api_key: str | None = None,
        http_get: Callable[..., Any] | None = None,
        daily_limit: int = DEFAULT_FMP_DAILY_LIMIT,
        cache_ttl_hours: float = 24.0,
        timeout: float = 10.0,
        base_url: str = "https://financialmodelingprep.com/stable",
    ) -> None:
        self.store = store
        self._api_key = api_key
        self._http_get = http_get or requests.get
        self._has_custom_http_get = http_get is not None
        self.daily_limit = daily_limit
        self.cache_ttl_hours = cache_ttl_hours
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")

    def _resolve_api_key(self) -> str:
        key = self._api_key if self._api_key is not None else os.getenv("FMP_API_KEY", "")
        if not key and not self._has_custom_http_get:
            raise ValueError("FMP_API_KEY must be provided or set in the environment")
        return key

    def _fetch_endpoint(
        self,
        endpoint_path: str,
        endpoint_name: str,
        symbol: str,
        params: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Any:
        if not self.store.can_make_fmp_request(daily_limit=self.daily_limit, now=now):
            used = self.store.get_fmp_requests_today(now=now)
            raise FmpRateLimitError(
                f"FMP daily request quota exhausted ({used}/{self.daily_limit})"
            )

        url = f"{self.base_url}/{endpoint_path.lstrip('/')}"
        query_params = {"apikey": self._resolve_api_key(), **(params or {})}

        response = self._http_get(url, params=query_params, timeout=self.timeout)
        self.store.record_fmp_request(endpoint=endpoint_name, symbol=symbol, now=now)
        try:
            response.raise_for_status()
        except Exception as exc:
            raise FmpApiError(_sanitize_error_message(str(exc))) from None
        return response.json()

    def get_fundamentals(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> FundamentalData:
        """Retrieve FCF, debt-to-equity, and earnings dates with SQLite daily caching."""
        norm_symbol = _normalize_symbol(symbol)

        cached = self.store.get_fmp_cache(norm_symbol, "fundamentals", now=now)
        if cached is not None:
            payload = cached.payload if isinstance(cached.payload, dict) else {}
            return FundamentalData(
                symbol=norm_symbol,
                free_cash_flow=float(payload["free_cash_flow"]),
                debt_to_equity=float(payload["debt_to_equity"]),
                earnings_dates=list(payload.get("earnings_dates", [])),
                from_cache=True,
            )

        is_legacy_v3 = "/api/v3" in self.base_url
        if is_legacy_v3:
            cashflow_payload = self._fetch_endpoint(
                endpoint_path=f"cash-flow-statement/{norm_symbol}",
                endpoint_name="cash-flow-statement",
                symbol=norm_symbol,
                params={"limit": 1},
                now=now,
            )
            ratios_payload = self._fetch_endpoint(
                endpoint_path=f"ratios/{norm_symbol}",
                endpoint_name="ratios",
                symbol=norm_symbol,
                params={"limit": 1},
                now=now,
            )
            earnings_payload = self._fetch_endpoint(
                endpoint_path=f"historical/earning_calendar/{norm_symbol}",
                endpoint_name="earning_calendar",
                symbol=norm_symbol,
                params={"limit": 4},
                now=now,
            )
        else:
            cashflow_payload = self._fetch_endpoint(
                endpoint_path="cash-flow-statement",
                endpoint_name="cash-flow-statement",
                symbol=norm_symbol,
                params={"symbol": norm_symbol, "limit": 1},
                now=now,
            )
            ratios_payload = self._fetch_endpoint(
                endpoint_path="ratios",
                endpoint_name="ratios",
                symbol=norm_symbol,
                params={"symbol": norm_symbol, "limit": 1},
                now=now,
            )
            earnings_payload = self._fetch_endpoint(
                endpoint_path="earnings",
                endpoint_name="earnings",
                symbol=norm_symbol,
                params={"symbol": norm_symbol},
                now=now,
            )

        free_cash_flow = _parse_free_cash_flow(cashflow_payload, norm_symbol)
        debt_to_equity = _parse_debt_to_equity(ratios_payload, norm_symbol)
        earnings_dates = _parse_earnings_dates(earnings_payload)

        cache_payload: dict[str, Any] = {
            "free_cash_flow": free_cash_flow,
            "debt_to_equity": debt_to_equity,
            "earnings_dates": earnings_dates,
        }
        primary_earnings = earnings_dates[0] if earnings_dates else None
        self.store.upsert_fmp_cache(
            symbol=norm_symbol,
            data_type="fundamentals",
            payload=cache_payload,
            earnings_date=primary_earnings,
            ttl_hours=self.cache_ttl_hours,
            now=now,
        )

        return FundamentalData(
            symbol=norm_symbol,
            free_cash_flow=free_cash_flow,
            debt_to_equity=debt_to_equity,
            earnings_dates=earnings_dates,
            from_cache=False,
        )
