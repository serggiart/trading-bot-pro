"""Typed data models and configuration for universe screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

NY_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Deterministic screening thresholds for liquidity, fundamentals, and trend."""

    min_avg_volume: float = 500_000.0
    max_spread_pct: float = 0.005
    max_debt_to_equity: float = 2.0
    min_free_cash_flow: float = 0.0
    outlier_sigma_threshold: float = 3.0
    fallback_pct_threshold: float = 0.03
    atr_period: int = 14
    sma_period: int = 200
    fmp_daily_limit: int = 250
    fmp_cache_ttl_hours: float = 24.0
    max_candidates_for_ai: int = 5


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """Normalized market quote and liquidity snapshot for a single ticker."""

    symbol: str
    price: float
    bid_price: float
    ask_price: float
    spread_pct: float
    volume: float
    source: str = "alpaca_iex"
    timestamp: datetime = field(default_factory=lambda: datetime.now(NY_TIMEZONE))


@dataclass(frozen=True, slots=True)
class PriceValidationResult:
    """Cross-source price consistency check between Alpaca IEX and Yahoo Finance."""

    symbol: str
    alpaca_price: float
    yfinance_price: float
    z_score: float
    pct_diff: float
    is_valid: bool
    alert_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TechnicalIndicators:
    """Trend, volatility, volume baseline, and macro regime indicators for a ticker."""

    symbol: str
    current_price: float
    sma_200: float
    atr_14: float
    above_sma_200: bool
    vix: float
    latest_volume: float = 0.0
    avg_volume_20d: float = 0.0
    ema_20: float = 0.0
    rsi_14: float = 50.0


@dataclass(frozen=True, slots=True)
class FundamentalData:
    """Company fundamental health metrics and upcoming/known earnings dates."""

    symbol: str
    free_cash_flow: float
    debt_to_equity: float
    earnings_dates: list[str] = field(default_factory=list)
    from_cache: bool = False


@dataclass(frozen=True, slots=True)
class UniverseScreeningResult:
    """Complete Stage-1 Universe screening output with full audit details."""

    symbol: str
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    quote: MarketQuote | None = None
    validation: PriceValidationResult | None = None
    indicators: TechnicalIndicators | None = None
    fundamentals: FundamentalData | None = None
