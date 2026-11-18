"""Market data ingestion, technical indicators, and cross-source price validation."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

from src.universe.models import (
    NY_TIMEZONE,
    MarketQuote,
    PriceValidationResult,
    TechnicalIndicators,
)


def _normalize_symbol(symbol: str) -> str:
    """Normalize and validate a ticker symbol."""
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must be a non-empty string")
    return cleaned


def _normalize_timestamp(ts: datetime | None) -> datetime:
    """Normalize a datetime to America/New_York, defaulting to current NY time."""
    if ts is None:
        return datetime.now(NY_TIMEZONE)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=NY_TIMEZONE)
    return ts.astimezone(NY_TIMEZONE)


def _to_clean_float_array(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    """Convert a sequence or Series into a 1D float64 array dropping NaN/inf entries."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Input series must be one-dimensional")
    return arr[np.isfinite(arr)]


def calculate_sma(
    closes: Sequence[float] | pd.Series | np.ndarray,
    period: int = 200,
) -> float:
    """Compute the simple moving average over the trailing `period` close prices."""
    if period <= 0:
        raise ValueError("period must be positive")
    arr = _to_clean_float_array(closes)
    if len(arr) < period:
        raise ValueError(f"Insufficient close prices ({len(arr)}) for SMA({period})")
    return float(np.mean(arr[-period:]))


def calculate_ema(
    closes: Sequence[float] | pd.Series | np.ndarray,
    period: int = 20,
) -> float:
    """Compute the exponential moving average (EMA) over `closes` seeded with SMA."""
    if period <= 0:
        raise ValueError("period must be positive")
    arr = _to_clean_float_array(closes)
    if len(arr) < period:
        raise ValueError(f"Insufficient close prices ({len(arr)}) for EMA({period})")

    ema = float(np.mean(arr[:period]))
    multiplier = 2.0 / (period + 1.0)
    for price in arr[period:]:
        ema = float(price) * multiplier + ema * (1.0 - multiplier)
    return float(ema)


def calculate_rsi(
    closes: Sequence[float] | pd.Series | np.ndarray,
    period: int = 14,
) -> float:
    """Compute Wilder's Relative Strength Index (RSI) safely on zero-loss series."""
    if period <= 0:
        raise ValueError("period must be positive")
    arr = _to_clean_float_array(closes)
    if len(arr) < period + 1:
        raise ValueError(f"Insufficient close prices ({len(arr)}) for RSI({period})")

    deltas = np.diff(arr)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + float(gain)) / period
        avg_loss = (avg_loss * (period - 1) + float(loss)) / period

    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0

    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def calculate_atr(
    highs: Sequence[float] | pd.Series | np.ndarray,
    lows: Sequence[float] | pd.Series | np.ndarray,
    closes: Sequence[float] | pd.Series | np.ndarray,
    period: int = 14,
) -> float:
    """Compute Wilder's Average True Range (ATR) over `period` bars."""
    if period <= 0:
        raise ValueError("period must be positive")

    h = np.asarray(highs, dtype=np.float64)
    l = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)

    if h.ndim != 1 or l.ndim != 1 or c.ndim != 1:
        raise ValueError("highs, lows, and closes must be one-dimensional")
    if not (len(h) == len(l) == len(c)):
        raise ValueError("highs, lows, and closes must have equal lengths")

    valid_mask = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    h = h[valid_mask]
    l = l[valid_mask]
    c = c[valid_mask]

    n = len(c)
    if n < period:
        raise ValueError(f"Insufficient OHLC bars ({n}) for ATR({period})")

    if n == period:
        true_ranges = np.empty(n, dtype=np.float64)
        true_ranges[0] = h[0] - l[0]
        if n > 1:
            prev_c = c[:-1]
            true_ranges[1:] = np.maximum.reduce(
                [h[1:] - l[1:], np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)]
            )
        return float(np.mean(true_ranges))

    prev_c = c[:-1]
    true_ranges = np.maximum.reduce(
        [h[1:] - l[1:], np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)]
    )
    atr = float(np.mean(true_ranges[:period]))
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + float(tr)) / period
    return float(atr)


def calculate_avg_volume(
    volumes: Sequence[float] | pd.Series | np.ndarray,
    period: int = 20,
    exclude_latest: bool = True,
) -> float:
    """Compute trailing average trading volume over `period` completed sessions."""
    if period <= 0:
        raise ValueError("period must be positive")
    arr = _to_clean_float_array(volumes)
    if len(arr) == 0:
        return 0.0
    window = arr[:-1] if (exclude_latest and len(arr) > 1) else arr
    if len(window) == 0:
        return 0.0
    sample = window[-period:] if len(window) >= period else window
    return float(np.mean(sample))


def validate_cross_price(
    symbol: str,
    alpaca_price: float,
    yfinance_price: float,
    return_std: float | None = None,
    sigma_threshold: float = 3.0,
    fallback_pct_threshold: float = 0.03,
) -> PriceValidationResult:
    """Validate cross-source price consistency against a 3-sigma outlier threshold.

    Uses `return_std` (daily return standard deviation) when positive and finite.
    Safely falls back to `fallback_pct_threshold` when `return_std` is None or zero,
    and flags non-positive/non-finite prices without raising ZeroDivisionError.
    """
    norm_symbol = _normalize_symbol(symbol)

    if (
        not math.isfinite(alpaca_price)
        or not math.isfinite(yfinance_price)
        or alpaca_price <= 0.0
        or yfinance_price <= 0.0
    ):
        return PriceValidationResult(
            symbol=norm_symbol,
            alpaca_price=float(alpaca_price) if math.isfinite(alpaca_price) else 0.0,
            yfinance_price=float(yfinance_price) if math.isfinite(yfinance_price) else 0.0,
            z_score=0.0,
            pct_diff=0.0,
            is_valid=False,
            alert_reason="Invalid non-positive or non-finite price for cross-source check",
        )

    pct_diff = abs(alpaca_price - yfinance_price) / yfinance_price

    if return_std is not None and math.isfinite(return_std) and return_std > 0.0:
        z_score = pct_diff / return_std
        is_outlier = z_score > sigma_threshold
    elif fallback_pct_threshold > 0.0 and sigma_threshold > 0.0:
        implied_std = fallback_pct_threshold / sigma_threshold
        z_score = pct_diff / implied_std
        is_outlier = pct_diff > fallback_pct_threshold
    else:
        z_score = 0.0
        is_outlier = pct_diff > 0.0

    alert_reason = (
        f"Cross-price discrepancy {pct_diff:.2%} exceeds {sigma_threshold:.2f}σ "
        f"(z={z_score:.2f})"
        if is_outlier
        else None
    )

    return PriceValidationResult(
        symbol=norm_symbol,
        alpaca_price=float(alpaca_price),
        yfinance_price=float(yfinance_price),
        z_score=float(z_score),
        pct_diff=float(pct_diff),
        is_valid=not is_outlier,
        alert_reason=alert_reason,
    )


def _extract_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Extract a column from an OHLCV DataFrame supporting standard or lowercase names."""
    if name in df.columns:
        series = df[name]
    elif name.lower() in df.columns:
        series = df[name.lower()]
    else:
        raise KeyError(f"Missing column '{name}' in OHLCV DataFrame")

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.dropna().astype(float)


class MarketDataClient:
    """Dual-source market data client (Alpaca IEX primary + yfinance secondary)."""

    def __init__(
        self,
        alpaca_client: Any | None = None,
        ticker_factory: Callable[[str], Any] | None = None,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._alpaca_client = alpaca_client
        self._ticker_factory = ticker_factory or yf.Ticker
        self._api_key = api_key
        self._secret_key = secret_key

    def _get_alpaca_client(self) -> Any:
        """Return injected Alpaca client or lazily build one from env/credentials."""
        if self._alpaca_client is None:
            api_key = self._api_key or os.getenv("ALPACA_API_KEY")
            secret_key = self._secret_key or os.getenv("ALPACA_SECRET_KEY")
            self._alpaca_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )
        return self._alpaca_client

    @staticmethod
    def _parse_snapshot(norm_symbol: str, snapshot: Any) -> MarketQuote:
        """Convert an Alpaca snapshot object into a normalized MarketQuote."""
        latest_trade = getattr(snapshot, "latest_trade", None)
        latest_quote = getattr(snapshot, "latest_quote", None)
        daily_bar = getattr(snapshot, "daily_bar", None)
        prev_bar = getattr(snapshot, "previous_daily_bar", None)

        bid_price = (
            float(latest_quote.bid_price)
            if latest_quote is not None and latest_quote.bid_price is not None
            else 0.0
        )
        ask_price = (
            float(latest_quote.ask_price)
            if latest_quote is not None and latest_quote.ask_price is not None
            else 0.0
        )

        if latest_trade is not None and latest_trade.price is not None and float(latest_trade.price) > 0.0:
            price = float(latest_trade.price)
        elif bid_price > 0.0 and ask_price > 0.0:
            price = (bid_price + ask_price) / 2.0
        elif daily_bar is not None and daily_bar.close is not None:
            price = float(daily_bar.close)
        else:
            price = 0.0

        if bid_price > 0.0 and ask_price > 0.0:
            midpoint = (bid_price + ask_price) / 2.0
            spread_pct = max(0.0, ask_price - bid_price) / midpoint if midpoint > 0.0 else 0.0
        else:
            spread_pct = 0.0

        if daily_bar is not None and daily_bar.volume is not None:
            volume = float(daily_bar.volume)
        elif prev_bar is not None and prev_bar.volume is not None:
            volume = float(prev_bar.volume)
        else:
            volume = 0.0

        raw_ts = (
            getattr(latest_trade, "timestamp", None)
            or getattr(latest_quote, "timestamp", None)
            or getattr(daily_bar, "timestamp", None)
        )
        timestamp = _normalize_timestamp(raw_ts)

        return MarketQuote(
            symbol=norm_symbol,
            price=price,
            bid_price=bid_price,
            ask_price=ask_price,
            spread_pct=spread_pct,
            volume=volume,
            source="alpaca_iex",
            timestamp=timestamp,
        )

    def get_quote(self, symbol: str) -> MarketQuote:
        """Fetch primary snapshot quote and liquidity metrics via Alpaca IEX feed."""
        norm_symbol = _normalize_symbol(symbol)
        client = self._get_alpaca_client()
        request = StockSnapshotRequest(symbol_or_symbols=norm_symbol, feed=DataFeed.IEX)
        snapshots = client.get_stock_snapshot(request)

        snapshot = snapshots[norm_symbol] if isinstance(snapshots, dict) else snapshots
        return self._parse_snapshot(norm_symbol, snapshot)

    def get_quotes_batch(self, symbols: Sequence[str]) -> dict[str, MarketQuote]:
        """Fetch snapshot quotes for multiple symbols in a single Alpaca IEX request."""
        norm_symbols = list(dict.fromkeys(_normalize_symbol(symbol) for symbol in symbols))
        if not norm_symbols:
            return {}

        client = self._get_alpaca_client()
        request = StockSnapshotRequest(symbol_or_symbols=norm_symbols, feed=DataFeed.IEX)
        snapshots = client.get_stock_snapshot(request)

        if not isinstance(snapshots, dict):
            if len(norm_symbols) == 1 and snapshots is not None:
                sym = norm_symbols[0]
                return {sym: self._parse_snapshot(sym, snapshots)}
            return {}

        quotes: dict[str, MarketQuote] = {}
        for sym in norm_symbols:
            snap = snapshots.get(sym)
            if snap is not None:
                quotes[sym] = self._parse_snapshot(sym, snap)
        return quotes

    def get_historical_ohlcv(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical OHLCV bars from yfinance."""
        norm_symbol = _normalize_symbol(symbol)
        ticker = self._ticker_factory(norm_symbol)
        history = ticker.history(period=period)
        if history is None or history.empty:
            raise ValueError(f"No yfinance historical data returned for {norm_symbol}")
        return history

    def get_vix(self) -> float:
        """Fetch the latest CBOE Volatility Index (^VIX) level via yfinance."""
        ticker = self._ticker_factory("^VIX")
        history = ticker.history(period="5d")
        if history is None or history.empty:
            raise ValueError("No yfinance data returned for ^VIX")
        closes = _extract_column(history, "Close")
        if closes.empty:
            raise ValueError("No valid Close prices returned for ^VIX")
        return float(closes.iloc[-1])

    def get_technical_indicators(
        self,
        symbol: str,
        vix: float | None = None,
        sma_period: int = 200,
        atr_period: int = 14,
        history: pd.DataFrame | None = None,
    ) -> TechnicalIndicators:
        """Compute SMA_200, ATR(14), and macro VIX state from 1y yfinance OHLCV."""
        norm_symbol = _normalize_symbol(symbol)
        df = history if history is not None else self.get_historical_ohlcv(norm_symbol, period="1y")

        closes = _extract_column(df, "Close")
        highs = _extract_column(df, "High")
        lows = _extract_column(df, "Low")

        current_price = float(closes.iloc[-1])
        sma_200 = calculate_sma(closes, period=sma_period)
        atr_14 = calculate_atr(highs, lows, closes, period=atr_period)
        clean_closes = _to_clean_float_array(closes)
        ema_20 = calculate_ema(clean_closes, period=20) if len(clean_closes) >= 20 else current_price
        rsi_14 = calculate_rsi(clean_closes, period=14) if len(clean_closes) >= 15 else 50.0
        resolved_vix = float(vix) if vix is not None else self.get_vix()

        latest_volume = 0.0
        avg_volume_20d = 0.0
        try:
            volumes = _extract_column(df, "Volume")
            if not volumes.empty:
                latest_volume = float(volumes.iloc[-1])
                avg_volume_20d = calculate_avg_volume(volumes, period=20, exclude_latest=True)
        except KeyError:
            pass

        return TechnicalIndicators(
            symbol=norm_symbol,
            current_price=current_price,
            sma_200=sma_200,
            atr_14=atr_14,
            above_sma_200=current_price > sma_200,
            vix=resolved_vix,
            latest_volume=latest_volume,
            avg_volume_20d=avg_volume_20d,
            ema_20=ema_20,
            rsi_14=rsi_14,
        )

    def validate_price(
        self,
        symbol: str,
        alpaca_price: float,
        sigma_threshold: float = 3.0,
        fallback_pct_threshold: float = 0.03,
        history: pd.DataFrame | None = None,
    ) -> PriceValidationResult:
        """Cross-validate Alpaca price against yfinance close and daily return volatility."""
        norm_symbol = _normalize_symbol(symbol)
        df = history if history is not None else self.get_historical_ohlcv(norm_symbol, period="1y")
        closes = _extract_column(df, "Close")
        yfinance_price = float(closes.iloc[-1])
        returns = closes.pct_change().dropna()
        return_std = float(returns.std(ddof=1)) if len(returns) >= 2 else None

        return validate_cross_price(
            symbol=norm_symbol,
            alpaca_price=alpaca_price,
            yfinance_price=yfinance_price,
            return_std=return_std,
            sigma_threshold=sigma_threshold,
            fallback_pct_threshold=fallback_pct_threshold,
        )
