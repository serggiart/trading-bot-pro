"""Typed data models and configuration for the deterministic backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration governing backtest capital, indicators, exits, and cache storage."""

    initial_capital: float = 10_000.0
    atr_stop_multiplier: float = 2.0
    atr_take_profit_multiplier: float = 2.0
    max_hold_bars: int = 10
    slippage_pct: float = 0.001
    sma_period: int = 200
    ema_period: int = 20
    rsi_period: int = 14
    rsi_entry_max: float = 60.0
    min_relative_volume: float = 1.2
    default_vix: float = 18.0
    max_stop_loss_pct: float = 0.0
    max_ema_extension_atr: float = 0.0
    breakeven_trigger_pct: float = 0.0
    peak_activation_pct: float = 0.0
    peak_rsi_threshold: float = 66.0
    allow_fractional_shares: bool = False
    cache_dir: Path = Path("data/candles")

    def __post_init__(self) -> None:
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if self.atr_stop_multiplier <= 0.0:
            raise ValueError("atr_stop_multiplier must be positive")
        if self.atr_take_profit_multiplier <= 0.0:
            raise ValueError("atr_take_profit_multiplier must be positive")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive")
        if not 0.0 <= self.slippage_pct < 1.0:
            raise ValueError("slippage_pct must be in [0.0, 1.0)")
        if self.sma_period <= 0 or self.ema_period <= 0 or self.rsi_period <= 0:
            raise ValueError("Indicator periods must be positive")
        if self.min_relative_volume < 0.0:
            raise ValueError("min_relative_volume must be non-negative")
        if self.default_vix < 0.0:
            raise ValueError("default_vix must be non-negative")
        if not 0.0 <= self.max_stop_loss_pct < 1.0:
            raise ValueError("max_stop_loss_pct must be in [0.0, 1.0)")
        if self.max_ema_extension_atr < 0.0:
            raise ValueError("max_ema_extension_atr must be non-negative")
        if not 0.0 <= self.breakeven_trigger_pct < 1.0:
            raise ValueError("breakeven_trigger_pct must be in [0.0, 1.0)")
        if not 0.0 <= self.peak_activation_pct < 1.0:
            raise ValueError("peak_activation_pct must be in [0.0, 1.0)")
        if not 0.0 < self.peak_rsi_threshold <= 100.0:
            raise ValueError("peak_rsi_threshold must be in (0.0, 100.0]")


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    """Completed simulated trade produced during a backtest run."""

    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    stop_price: float
    take_profit_price: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate performance and risk metrics for a completed backtest.

    Percentages (`win_rate`, `total_return_pct`, `max_drawdown_pct`) are expressed
    as decimal ratios (e.g. 0.15 represents 15%). `avg_loss` is the positive mean
    loss magnitude across losing trades.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    initial_capital: float
    final_equity: float
    total_return_pct: float
    compounded_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    expectancy_per_trade: float
    avg_win: float
    avg_loss: float


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Complete output of a backtest simulation including trades and equity curve."""

    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[SimulatedTrade]
    equity_curve: list[tuple[str, float]]
    vetoed_entries_count: int
