"""Typed data models and configuration for limit-only order execution and exit management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

NY_TIMEZONE = ZoneInfo("America/New_York")


def ensure_ny_datetime(dt: datetime | None = None) -> datetime:
    """Return a timezone-aware datetime normalized to America/New_York."""
    if dt is None:
        return datetime.now(NY_TIMEZONE)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=NY_TIMEZONE)
    return dt.astimezone(NY_TIMEZONE)


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Configuration governing limit order slippage, trailing stops, and exit guards."""

    slippage_tolerance_pct: float = 0.001
    exit_slippage_pct: float = 0.003
    emergency_slippage_pct: float = 0.015
    trailing_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 2.0
    enable_take_profit_exit: bool = False
    max_hold_hours: float = 72.0
    min_target_return_pct: float = 0.01
    flash_crash_window_minutes: float = 5.0
    flash_crash_drop_pct: float = 0.03
    paper: bool = True

    def __post_init__(self) -> None:
        if self.slippage_tolerance_pct < 0:
            raise ValueError("slippage_tolerance_pct must be non-negative")
        if not 0 <= self.exit_slippage_pct < 1.0:
            raise ValueError("exit_slippage_pct must be in [0, 1.0)")
        if not 0 <= self.emergency_slippage_pct < 1.0:
            raise ValueError("emergency_slippage_pct must be in [0, 1.0)")
        if self.trailing_atr_multiplier <= 0:
            raise ValueError("trailing_atr_multiplier must be positive")
        if self.take_profit_atr_multiplier < 0:
            raise ValueError("take_profit_atr_multiplier must be non-negative")
        if self.max_hold_hours <= 0:
            raise ValueError("max_hold_hours must be positive")
        if self.flash_crash_window_minutes <= 0:
            raise ValueError("flash_crash_window_minutes must be positive")
        if not 0 < self.flash_crash_drop_pct < 1.0:
            raise ValueError("flash_crash_drop_pct must be in (0, 1.0)")


@dataclass(slots=True)
class ManagedPosition:
    """Active equity position tracked by PositionManager."""

    symbol: str
    qty: float
    entry_price: float
    current_stop_price: float
    atr_14: float
    opened_at: datetime
    highest_price: float | None = None
    stop_order_id: str | None = None
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.current_stop_price < 0:
            raise ValueError("current_stop_price must be non-negative")
        self.opened_at = ensure_ny_datetime(self.opened_at)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Broker account equity, liquidity, and daily PnL snapshot."""

    equity: float
    cash: float
    buying_power: float
    last_equity: float
    daily_pnl: float


@dataclass(frozen=True, slots=True)
class ExecutionOrderResult:
    """Result returned by AlpacaExecutionClient after submitting or simulating an order."""

    submitted: bool
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: str
    qty: float
    limit_price: float
    stop_price: float | None
    order_type: str = "limit"
    status: str = "submitted"
    reason: str = ""


ExitAction = Literal[
    "HOLD",
    "TRAIL_UPDATE",
    "TIME_EXIT",
    "TAKE_PROFIT_EXIT",
    "TRAILING_STOP_EXIT",
    "EMERGENCY_EXIT",
]


@dataclass(frozen=True, slots=True)
class ExitEvaluationResult:
    """Outcome of evaluating an open position against all deterministic exit rules."""

    symbol: str
    action: ExitAction
    should_exit: bool
    new_stop_price: float
    exit_reason: str = ""
    order_result: ExecutionOrderResult | None = None
