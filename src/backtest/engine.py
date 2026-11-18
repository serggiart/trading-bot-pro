"""Deterministic historical backtest engine gated by RiskGuardian."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.models import (
    BacktestConfig,
    BacktestMetrics,
    BacktestReport,
    SimulatedTrade,
)
from src.execution.position_manager import compute_trailing_stop
from src.guardian import (
    MarketContext,
    PortfolioPosition,
    RiskGuardian,
    TradeCandidate,
    normalize_to_ny,
)
from src.universe.market_data import (
    calculate_atr,
    calculate_avg_volume,
    calculate_sma,
)


def calculate_ema_series(closes: pd.Series, period: int = 20) -> pd.Series:
    """Compute the causal Exponential Moving Average series over `period` bars."""
    if period <= 0:
        raise ValueError("period must be positive")
    series = pd.Series(closes, dtype=np.float64)
    if series.empty:
        return series
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's Relative Strength Index (RSI) series safely.

    Handles flat or zero-loss windows without ZeroDivisionError:
    - avg_loss == 0 and avg_gain > 0 -> 100.0
    - avg_loss == 0 and avg_gain == 0 -> 50.0
    """
    if period <= 0:
        raise ValueError("period must be positive")

    series = pd.Series(closes, dtype=np.float64)
    n = len(series)
    result = pd.Series(np.full(n, np.nan, dtype=np.float64), index=series.index)
    if n <= period:
        return result

    values = series.to_numpy(dtype=np.float64)
    deltas = np.diff(values)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def _rsi_from_avgs(gain: float, loss: float) -> float:
        if loss == 0.0 and gain == 0.0:
            return 50.0
        if loss == 0.0:
            return 100.0
        if gain == 0.0:
            return 0.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    result.iloc[period] = _rsi_from_avgs(avg_gain, avg_loss)
    for idx in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + float(gains[idx - 1])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[idx - 1])) / period
        result.iloc[idx] = _rsi_from_avgs(avg_gain, avg_loss)

    return result


def compute_backtest_metrics(
    trades: Sequence[SimulatedTrade],
    initial_capital: float,
    equity_curve: Sequence[tuple[str, float]],
) -> BacktestMetrics:
    """Compute aggregate performance metrics safely for 0-trade and 0-loss edge cases."""
    total_trades = len(trades)
    winning = [t for t in trades if t.pnl > 0.0]
    losing = [t for t in trades if t.pnl < 0.0]

    winning_trades = len(winning)
    losing_trades = len(losing)
    win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in winning)
    gross_loss = abs(sum(t.pnl for t in losing))

    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0

    total_trade_pnl = sum(t.pnl for t in trades)
    expectancy_per_trade = (total_trade_pnl / total_trades) if total_trades > 0 else 0.0

    final_equity = (
        float(equity_curve[-1][1])
        if equity_curve
        else float(initial_capital) + total_trade_pnl
    )
    compounded_pnl = final_equity - float(initial_capital)
    total_return_pct = (
        (compounded_pnl / float(initial_capital)) if initial_capital > 0.0 else 0.0
    )

    max_drawdown_pct = 0.0
    if equity_curve:
        peak = float(initial_capital)
        for _, equity in equity_curve:
            eq_val = float(equity)
            if eq_val > peak:
                peak = eq_val
            if peak > 0.0:
                drawdown = (peak - eq_val) / peak
                if drawdown > max_drawdown_pct:
                    max_drawdown_pct = drawdown

    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        initial_capital=float(initial_capital),
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        compounded_pnl=compounded_pnl,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        expectancy_per_trade=expectancy_per_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def _format_bar_date(ts: Any) -> str:
    """Format a bar index timestamp cleanly as YYYY-MM-DD (or ISO if intraday)."""
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.isoformat()
    if isinstance(ts, datetime):
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
            return ts.strftime("%Y-%m-%d")
        return ts.isoformat()
    return str(ts)


def _to_ny_datetime(ts: Any) -> datetime:
    """Convert an index value to an America/New_York aware datetime."""
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        dt = pd.Timestamp(ts).to_pydatetime()
    return normalize_to_ny(dt)


def _extract_ohlcv_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Extract an OHLCV column supporting standard capitalized or lowercase names."""
    if name in df.columns:
        col = df[name]
    elif name.lower() in df.columns:
        col = df[name.lower()]
    else:
        raise KeyError(f"Missing required OHLCV column '{name}'")
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.astype(np.float64)


@dataclass(slots=True)
class _PreparedSymbol:
    """Pre-aligned OHLCV arrays and causal indicator series for one symbol."""

    symbol: str
    sector: str
    dates: list[pd.Timestamp]
    date_to_pos: dict[pd.Timestamp, int]
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    ema_series: pd.Series
    rsi_series: pd.Series


@dataclass(slots=True)
class _OpenPosition:
    """Tracks an active simulated position during a backtest run."""

    symbol: str
    sector: str
    entry_date: str
    entry_price: float
    shares: int
    initial_stop_price: float
    stop_price: float
    take_profit_price: float
    highest_price: float
    bars_held: int = 0


def _normalize_timestamp_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert a DataFrame/Series index into a tz-naive normalized DatetimeIndex."""
    dt_idx = pd.to_datetime(index)
    if dt_idx.tz is not None:
        dt_idx = dt_idx.tz_convert("America/New_York").tz_localize(None)
    return dt_idx


class BacktestEngine:
    """Chronological OHLCV simulator integrating deterministic exits and RiskGuardian."""

    def __init__(
        self,
        guardian: RiskGuardian | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.guardian = guardian or RiskGuardian()

    def _prepare_symbols(
        self,
        market_data: Mapping[str, pd.DataFrame],
        sector_map: Mapping[str, str] | None,
    ) -> dict[str, _PreparedSymbol]:
        """Validate, sort, and precompute causal EMA/RSI series for each symbol."""
        normalized_sectors = (
            {k.strip().upper(): v for k, v in sector_map.items()}
            if sector_map
            else {}
        )
        prepared: dict[str, _PreparedSymbol] = {}
        for raw_symbol, raw_df in market_data.items():
            symbol = raw_symbol.strip().upper()
            if not symbol:
                raise ValueError("market_data contains an empty symbol key")
            if raw_df is None or raw_df.empty:
                continue

            df = raw_df.copy()
            df.index = _normalize_timestamp_index(df.index)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            opens = _extract_ohlcv_column(df, "Open").to_numpy(dtype=np.float64)
            highs = _extract_ohlcv_column(df, "High").to_numpy(dtype=np.float64)
            lows = _extract_ohlcv_column(df, "Low").to_numpy(dtype=np.float64)
            closes_s = _extract_ohlcv_column(df, "Close")
            closes = closes_s.to_numpy(dtype=np.float64)
            volumes = _extract_ohlcv_column(df, "Volume").to_numpy(dtype=np.float64)

            ema_series = calculate_ema_series(closes_s, period=self.config.ema_period)
            rsi_series = calculate_rsi_series(closes_s, period=self.config.rsi_period)

            dates = list(df.index)
            date_to_pos = {ts: idx for idx, ts in enumerate(dates)}

            prepared[symbol] = _PreparedSymbol(
                symbol=symbol,
                sector=normalized_sectors.get(symbol, ""),
                dates=dates,
                date_to_pos=date_to_pos,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                ema_series=ema_series,
                rsi_series=rsi_series,
            )
        return prepared

    def _build_vix_lookup(self, vix_series: pd.Series | None) -> pd.Series | None:
        """Normalize an optional VIX series to a sorted, tz-naive DatetimeIndex."""
        if vix_series is None or vix_series.empty:
            return None
        s = vix_series.copy()
        s.index = _normalize_timestamp_index(s.index)
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.astype(np.float64)

    def _resolve_vix(self, vix_series: pd.Series | None, dt: pd.Timestamp) -> float:
        """Resolve VIX level at `dt` without future lookahead, falling back to default_vix."""
        if vix_series is None or vix_series.empty:
            return self.config.default_vix
        if dt in vix_series.index:
            val = float(vix_series.loc[dt])
            if math.isfinite(val):
                return val
        prior = vix_series.loc[:dt]
        if not prior.empty:
            val = float(prior.iloc[-1])
            if math.isfinite(val):
                return val
        return self.config.default_vix

    def run(
        self,
        market_data: Mapping[str, pd.DataFrame],
        vix_series: pd.Series | None = None,
        sector_map: Mapping[str, str] | None = None,
        correlation_matrix: Mapping[tuple[str, str], float] | None = None,
    ) -> BacktestReport:
        """Run chronological backtest simulation across aligned OHLCV data."""
        prepared = self._prepare_symbols(market_data, sector_map)
        if not prepared:
            metrics = compute_backtest_metrics([], self.config.initial_capital, [])
            return BacktestReport(
                config=self.config,
                metrics=metrics,
                trades=[],
                equity_curve=[],
                vetoed_entries_count=0,
            )

        vix_lookup = self._build_vix_lookup(vix_series)
        all_dates = sorted({ts for sym in prepared.values() for ts in sym.dates})
        warmup = max(
            self.config.sma_period,
            self.config.ema_period,
            self.config.rsi_period,
            21,
        )

        cash = float(self.config.initial_capital)
        prev_equity = float(self.config.initial_capital)
        open_positions: dict[str, _OpenPosition] = {}
        last_known_close: dict[str, float] = {}
        trades: list[SimulatedTrade] = []
        equity_curve: list[tuple[str, float]] = []
        vetoed_entries_count = 0

        for global_idx, bar_ts in enumerate(all_dates):
            # Track latest observed closes for mark-to-market accounting
            for sym, p in prepared.items():
                pos_idx = p.date_to_pos.get(bar_ts)
                if pos_idx is not None:
                    last_known_close[sym] = float(p.closes[pos_idx])

            if global_idx < warmup:
                continue

            date_str = _format_bar_date(bar_ts)
            bar_dt = _to_ny_datetime(bar_ts)
            bar_vix = self._resolve_vix(vix_lookup, bar_ts)

            # 1. Manage open positions first
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                p = prepared[sym]
                pos_idx = p.date_to_pos.get(bar_ts)
                if pos_idx is None:
                    continue

                pos.bars_held += 1
                bar_open = float(p.opens[pos_idx])
                bar_high = float(p.highs[pos_idx])
                bar_low = float(p.lows[pos_idx])
                bar_close = float(p.closes[pos_idx])

                exit_price: float | None = None
                exit_reason: str | None = None

                if bar_low <= pos.stop_price:
                    raw_fill = min(bar_open, pos.stop_price)
                    exit_price = max(0.0, raw_fill * (1.0 - self.config.slippage_pct))
                    exit_reason = (
                        "TRAILING_STOP"
                        if pos.stop_price > pos.initial_stop_price
                        else "STOP_LOSS"
                    )
                elif bar_high >= pos.take_profit_price:
                    exit_price = max(
                        0.0, pos.take_profit_price * (1.0 - self.config.slippage_pct)
                    )
                    exit_reason = "TAKE_PROFIT"
                elif pos.bars_held >= self.config.max_hold_bars:
                    exit_price = max(0.0, bar_close * (1.0 - self.config.slippage_pct))
                    exit_reason = "TIME_EXIT"
                else:
                    bar_atr = calculate_atr(
                        p.highs[: pos_idx + 1],
                        p.lows[: pos_idx + 1],
                        p.closes[: pos_idx + 1],
                        period=14,
                    )
                    new_stop = compute_trailing_stop(
                        entry_price=pos.entry_price,
                        current_price=bar_close,
                        current_stop=pos.stop_price,
                        atr_14=bar_atr,
                        atr_multiplier=self.config.atr_stop_multiplier,
                        highest_price=pos.highest_price,
                    )
                    pos.highest_price = max(pos.highest_price, bar_close)
                    pos.stop_price = new_stop

                if exit_price is not None and exit_reason is not None:
                    cash += pos.shares * exit_price
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    return_pct = (exit_price - pos.entry_price) / pos.entry_price
                    trades.append(
                        SimulatedTrade(
                            symbol=pos.symbol,
                            entry_date=pos.entry_date,
                            exit_date=date_str,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            shares=pos.shares,
                            stop_price=pos.stop_price,
                            take_profit_price=pos.take_profit_price,
                            pnl=pnl,
                            return_pct=return_pct,
                            exit_reason=exit_reason,
                        )
                    )
                    del open_positions[sym]

            # 2. Mark realized + open equity for compounding & circuit-breaker context
            open_positions_value = sum(
                pos.shares * last_known_close[pos.symbol]
                for pos in open_positions.values()
            )
            equity = cash + open_positions_value
            day_start_equity = prev_equity
            daily_pnl = equity - day_start_equity

            # 3. Evaluate deterministic entry setups and gate through RiskGuardian
            for sym, p in prepared.items():
                if sym in open_positions:
                    continue
                pos_idx = p.date_to_pos.get(bar_ts)
                if pos_idx is None or pos_idx < warmup:
                    continue

                bar_close = float(p.closes[pos_idx])
                bar_vol = float(p.volumes[pos_idx])
                if bar_close <= 0.0:
                    continue

                sma_val = calculate_sma(
                    p.closes[pos_idx + 1 - self.config.sma_period : pos_idx + 1],
                    period=self.config.sma_period,
                )
                ema_val = float(p.ema_series.iloc[pos_idx])
                rsi_val = float(p.rsi_series.iloc[pos_idx])
                avg_vol_20 = calculate_avg_volume(
                    p.volumes[max(0, pos_idx - 20) : pos_idx + 1],
                    period=20,
                    exclude_latest=True,
                )
                rel_vol = (bar_vol / avg_vol_20) if avg_vol_20 > 0.0 else 0.0

                if not (
                    bar_close > sma_val
                    and bar_close > ema_val
                    and math.isfinite(rsi_val)
                    and rsi_val <= self.config.rsi_entry_max
                    and rel_vol >= self.config.min_relative_volume
                ):
                    continue

                bar_atr = calculate_atr(
                    p.highs[: pos_idx + 1],
                    p.lows[: pos_idx + 1],
                    p.closes[: pos_idx + 1],
                    period=14,
                )
                if bar_atr <= 0.0:
                    continue

                entry_price = bar_close * (1.0 + self.config.slippage_pct)
                portfolio_positions = [
                    PortfolioPosition(
                        symbol=open_pos.symbol,
                        sector=open_pos.sector,
                        market_value=open_pos.shares * last_known_close[open_pos.symbol],
                        shares=float(open_pos.shares),
                        avg_price=open_pos.entry_price,
                    )
                    for open_pos in open_positions.values()
                ]
                candidate = TradeCandidate(
                    symbol=sym,
                    sector=p.sector,
                    entry_price=entry_price,
                    atr=bar_atr,
                )
                context = MarketContext(
                    now=bar_dt,
                    vix=bar_vix,
                    starting_equity=day_start_equity,
                    daily_pnl=daily_pnl,
                    available_capital=cash,
                    portfolio_positions=portfolio_positions,
                    correlation_matrix=correlation_matrix or {},
                )

                decision = self.guardian.evaluate(candidate, context)
                if not decision.approved or decision.recommended_shares <= 0:
                    vetoed_entries_count += 1
                    continue

                shares = decision.recommended_shares
                if shares * entry_price > cash:
                    shares = math.floor(cash / entry_price)
                if shares <= 0:
                    vetoed_entries_count += 1
                    continue

                initial_stop = max(
                    0.0, entry_price - bar_atr * self.config.atr_stop_multiplier
                )
                take_profit = (
                    entry_price + bar_atr * self.config.atr_take_profit_multiplier
                )
                cash -= shares * entry_price
                open_positions[sym] = _OpenPosition(
                    symbol=sym,
                    sector=p.sector,
                    entry_date=date_str,
                    entry_price=entry_price,
                    shares=shares,
                    initial_stop_price=initial_stop,
                    stop_price=initial_stop,
                    take_profit_price=take_profit,
                    highest_price=entry_price,
                    bars_held=0,
                )

            end_positions_value = sum(
                pos.shares * last_known_close[pos.symbol]
                for pos in open_positions.values()
            )
            end_equity = cash + end_positions_value
            equity_curve.append((date_str, end_equity))
            prev_equity = end_equity

        # 4. Close remaining open positions at END_OF_DATA
        if open_positions:
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                p = prepared[sym]
                final_ts = p.dates[-1]
                final_close = float(p.closes[-1])
                exit_price = max(0.0, final_close * (1.0 - self.config.slippage_pct))
                cash += pos.shares * exit_price
                pnl = (exit_price - pos.entry_price) * pos.shares
                return_pct = (exit_price - pos.entry_price) / pos.entry_price
                trades.append(
                    SimulatedTrade(
                        symbol=pos.symbol,
                        entry_date=pos.entry_date,
                        exit_date=_format_bar_date(final_ts),
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        shares=pos.shares,
                        stop_price=pos.stop_price,
                        take_profit_price=pos.take_profit_price,
                        pnl=pnl,
                        return_pct=return_pct,
                        exit_reason="END_OF_DATA",
                    )
                )
                del open_positions[sym]

            if equity_curve:
                last_date, _ = equity_curve[-1]
                equity_curve[-1] = (last_date, cash)

        metrics = compute_backtest_metrics(
            trades=trades,
            initial_capital=self.config.initial_capital,
            equity_curve=equity_curve,
        )
        return BacktestReport(
            config=self.config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            vetoed_entries_count=vetoed_entries_count,
        )
