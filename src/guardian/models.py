"""Deterministic data models and timezone helpers for the Risk Guardian."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


def normalize_to_ny(dt: datetime) -> datetime:
    """Normalize a datetime to America/New_York timezone.

    Naive datetimes are interpreted as America/New_York wall time.
    Aware datetimes are converted to America/New_York.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=NY_TZ)
    return dt.astimezone(NY_TZ)


@dataclass(frozen=True, slots=True)
class GuardianConfig:
    """Deterministic risk parameters for RiskGuardian."""

    max_vix: float = 30.0
    max_correlation: float = 0.75
    blackout_hours: float = 24.0
    max_daily_drawdown_pct: float = 0.03
    risk_per_trade_pct: float = 0.01
    atr_multiplier: float = 2.0
    kelly_fraction: float = 0.25
    max_asset_exposure_pct: float = 0.20
    max_sector_exposure_pct: float = 0.35
    use_kelly_cap: bool = True


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    """Represents an existing open position in the portfolio."""

    symbol: str
    sector: str = ""
    market_value: float = 0.0
    shares: float = 0.0
    avg_price: float = 0.0

    @property
    def effective_market_value(self) -> float:
        """Return explicit market_value, falling back to shares * avg_price."""
        if self.market_value > 0.0:
            return self.market_value
        if self.shares > 0.0 and self.avg_price > 0.0:
            return self.shares * self.avg_price
        return 0.0


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    """Represents a proposed equity trade candidate for risk evaluation."""

    symbol: str
    sector: str
    entry_price: float
    atr: float
    win_rate: float | None = None
    payoff_ratio: float | None = None
    earnings_dates: Sequence[datetime] = ()
    side: str = "BUY"


@dataclass(slots=True)
class MarketContext:
    """Snapshot of deterministic market, portfolio, and calendar state."""

    now: datetime
    vix: float
    starting_equity: float
    daily_pnl: float = 0.0
    available_capital: float | None = None
    portfolio_positions: Sequence[PortfolioPosition] = ()
    correlation_matrix: Mapping[tuple[str, str], float] | Mapping[str, float] = field(
        default_factory=dict
    )
    fed_dates: Sequence[datetime] = ()
    earnings_dates: Sequence[datetime] = ()

    def __post_init__(self) -> None:
        self.now = normalize_to_ny(self.now)


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Result of a single deterministic risk rule check."""

    rule_name: str
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PositionSizeResult:
    """Output of deterministic Kelly/ATR capital allocation."""

    shares: int
    allocation_pct: float
    position_value: float
    stop_distance: float
    stop_price: float
    risk_amount: float
    capped_by: str | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GuardianDecision:
    """Final deterministic decision returned by RiskGuardian."""

    approved: bool
    symbol: str
    timestamp: datetime
    rule_results: list[RuleResult]
    veto_reasons: list[str]
    recommended_shares: int
    position_size: PositionSizeResult | None = None
