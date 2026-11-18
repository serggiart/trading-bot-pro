"""Unit and integration tests for src/backtest/ (cache, indicators, metrics, engine)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestReport,
    ParquetCandleStore,
    SimulatedTrade,
    calculate_ema_series,
    calculate_rsi_series,
    compute_backtest_metrics,
)
from src.guardian import GuardianConfig, RiskGuardian


def _make_ohlcv_df(
    dates: pd.DatetimeIndex,
    opens: list[float] | np.ndarray,
    highs: list[float] | np.ndarray,
    lows: list[float] | np.ndarray,
    closes: list[float] | np.ndarray,
    volumes: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame indexed by trading dates."""
    return pd.DataFrame(
        {
            "Open": np.asarray(opens, dtype=float),
            "High": np.asarray(highs, dtype=float),
            "Low": np.asarray(lows, dtype=float),
            "Close": np.asarray(closes, dtype=float),
            "Volume": np.asarray(volumes, dtype=float),
        },
        index=dates,
    )


def _make_base_uptrend_df(
    n_bars: int = 240,
    base_price: float = 100.0,
    daily_step: float = 0.05,
    start_date: str = "2025-01-02",
) -> pd.DataFrame:
    """Create deterministic baseline OHLCV bars with gentle oscillations so RSI <= 65."""
    dates = pd.bdate_range(start=start_date, periods=n_bars)
    closes = np.empty(n_bars, dtype=float)
    for i in range(n_bars):
        # Gentle upward drift with +/-0.45 oscillation keeps RSI around ~54-60
        osc = 0.45 if i % 2 == 0 else -0.45
        closes[i] = base_price + i * daily_step + osc
    opens = closes - 0.1
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n_bars, 1_000_000.0)
    return _make_ohlcv_df(dates, opens, highs, lows, closes, volumes)


# ---------------------------------------------------------------------------
# 1. ParquetCandleStore tests
# ---------------------------------------------------------------------------


def test_parquet_store_save_load_and_cache_hit(tmp_path: Path) -> None:
    """Saving and loading round-trips data, and cache hit makes zero yfinance calls."""
    dates = pd.bdate_range("2025-01-02", periods=5)
    df = _make_ohlcv_df(
        dates,
        [100.0, 101.0, 102.0, 103.0, 104.0],
        [101.0, 102.0, 103.0, 104.0, 105.0],
        [99.0, 100.0, 101.0, 102.0, 103.0],
        [100.5, 101.5, 102.5, 103.5, 104.5],
        [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df
    ticker_factory = MagicMock(return_value=mock_ticker)

    store = ParquetCandleStore(cache_dir=tmp_path, ticker_factory=ticker_factory)
    assert store.load_candles("AAPL") is None

    # First call: cache miss -> fetches once and writes AAPL.parquet
    fetched = store.get_or_fetch_candles("aapl", period="1y")
    assert ticker_factory.call_count == 1
    ticker_factory.assert_called_with("AAPL")
    mock_ticker.history.assert_called_once_with(period="1y")
    assert (tmp_path / "AAPL.parquet").exists()
    pd.testing.assert_frame_equal(fetched, df, check_freq=False)

    # Second call: cache hit -> 0 calls to ticker_factory
    ticker_factory.reset_mock()
    mock_ticker.history.reset_mock()
    cached = store.get_or_fetch_candles("AAPL", period="1y", refresh=False)
    assert ticker_factory.call_count == 0
    assert mock_ticker.history.call_count == 0
    pd.testing.assert_frame_equal(cached, df, check_freq=False)

    # Refresh=True forces network fetch and updates cache
    refreshed = store.get_or_fetch_candles("AAPL", period="1y", refresh=True)
    assert ticker_factory.call_count == 1
    pd.testing.assert_frame_equal(refreshed, df, check_freq=False)


def test_parquet_store_validation_errors(tmp_path: Path) -> None:
    """Reject empty symbols, empty DataFrames, and empty remote responses."""
    store = ParquetCandleStore(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        store.load_candles("   ")

    with pytest.raises(ValueError, match="empty"):
        store.save_candles("MSFT", pd.DataFrame())

    empty_ticker = MagicMock()
    empty_ticker.history.return_value = pd.DataFrame()
    store_empty = ParquetCandleStore(
        cache_dir=tmp_path,
        ticker_factory=lambda _sym: empty_ticker,
    )
    with pytest.raises(ValueError, match="No historical candle data"):
        store_empty.get_or_fetch_candles("NVDA")


# ---------------------------------------------------------------------------
# 2. Indicator & metrics helper tests
# ---------------------------------------------------------------------------


def test_calculate_ema_series_matches_recursive_definition() -> None:
    """EMA series matches exact alpha = 2 / (period + 1) recursion."""
    closes = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    ema = calculate_ema_series(closes, period=3)
    alpha = 2.0 / (3 + 1)
    expected = [10.0]
    for price in closes.iloc[1:]:
        expected.append(alpha * float(price) + (1.0 - alpha) * expected[-1])
    np.testing.assert_allclose(ema.to_numpy(), expected, rtol=1e-12)

    with pytest.raises(ValueError, match="positive"):
        calculate_ema_series(closes, period=0)


def test_calculate_rsi_series_edge_cases() -> None:
    """RSI safely handles zero-loss, flat, zero-gain, and short series."""
    # Strictly increasing -> avg_loss == 0, avg_gain > 0 -> 100.0
    rising = pd.Series(np.linspace(100.0, 120.0, 25))
    rsi_rising = calculate_rsi_series(rising, period=14)
    assert rsi_rising.iloc[:14].isna().all()
    assert np.allclose(rsi_rising.iloc[14:], 100.0)

    # Completely flat -> avg_loss == 0, avg_gain == 0 -> 50.0
    flat = pd.Series(np.full(25, 100.0))
    rsi_flat = calculate_rsi_series(flat, period=14)
    assert np.allclose(rsi_flat.iloc[14:], 50.0)

    # Strictly decreasing -> avg_gain == 0, avg_loss > 0 -> 0.0
    falling = pd.Series(np.linspace(120.0, 100.0, 25))
    rsi_falling = calculate_rsi_series(falling, period=14)
    assert np.allclose(rsi_falling.iloc[14:], 0.0)

    # Short series -> all NaN, no exception
    short = pd.Series([100.0, 101.0, 102.0])
    rsi_short = calculate_rsi_series(short, period=14)
    assert len(rsi_short) == 3
    assert rsi_short.isna().all()


def test_compute_backtest_metrics_zero_trades_zero_losses_and_drawdown() -> None:
    """Metrics handle 0 trades, 0 losses, and compute exact max drawdown."""
    # 0 trades
    m_empty = compute_backtest_metrics([], 10_000.0, [("2025-01-02", 10_000.0)])
    assert m_empty.total_trades == 0
    assert m_empty.win_rate == 0.0
    assert m_empty.profit_factor == 0.0
    assert m_empty.max_drawdown_pct == 0.0
    assert m_empty.final_equity == 10_000.0
    assert m_empty.avg_win == 0.0
    assert m_empty.avg_loss == 0.0
    assert m_empty.expectancy_per_trade == 0.0

    # 0 losing trades
    winners = [
        SimulatedTrade(
            symbol="AAPL",
            entry_date="2025-01-02",
            exit_date="2025-01-03",
            entry_price=100.0,
            exit_price=105.0,
            shares=10,
            stop_price=97.0,
            take_profit_price=105.0,
            pnl=50.0,
            return_pct=0.05,
            exit_reason="TAKE_PROFIT",
        )
    ]
    m_all_win = compute_backtest_metrics(
        winners, 10_000.0, [("2025-01-02", 10_000.0), ("2025-01-03", 10_050.0)]
    )
    assert m_all_win.winning_trades == 1
    assert m_all_win.losing_trades == 0
    assert m_all_win.win_rate == 1.0
    assert math.isinf(m_all_win.profit_factor)
    assert m_all_win.avg_win == 50.0
    assert m_all_win.avg_loss == 0.0
    assert m_all_win.max_drawdown_pct == 0.0

    # Mixed trades with drawdown: peak 12_000 -> trough 9_600 => 20% drawdown
    mixed = [
        winners[0],
        SimulatedTrade(
            symbol="AAPL",
            entry_date="2025-01-04",
            exit_date="2025-01-05",
            entry_price=100.0,
            exit_price=95.0,
            shares=5,
            stop_price=95.0,
            take_profit_price=105.0,
            pnl=-25.0,
            return_pct=-0.05,
            exit_reason="STOP_LOSS",
        ),
    ]
    curve = [
        ("2025-01-02", 12_000.0),
        ("2025-01-03", 9_600.0),
        ("2025-01-04", 10_025.0),
    ]
    m_mixed = compute_backtest_metrics(mixed, 10_000.0, curve)
    assert m_mixed.total_trades == 2
    assert m_mixed.winning_trades == 1
    assert m_mixed.losing_trades == 1
    assert m_mixed.win_rate == pytest.approx(0.5)
    assert m_mixed.profit_factor == pytest.approx(2.0)
    assert m_mixed.avg_win == pytest.approx(50.0)
    assert m_mixed.avg_loss == pytest.approx(25.0)
    assert m_mixed.expectancy_per_trade == pytest.approx(12.5)
    assert m_mixed.max_drawdown_pct == pytest.approx(0.20)
    assert m_mixed.compounded_pnl == pytest.approx(25.0)
    assert m_mixed.total_return_pct == pytest.approx(0.0025)


# ---------------------------------------------------------------------------
# 3. BacktestEngine end-to-end tests
# ---------------------------------------------------------------------------


def _setup_fast_config(**overrides: object) -> BacktestConfig:
    """Helper config with compact indicator windows for clean synthetic tests."""
    defaults: dict[str, object] = {
        "initial_capital": 10_000.0,
        "atr_stop_multiplier": 1.5,
        "atr_take_profit_multiplier": 2.5,
        "max_hold_bars": 5,
        "slippage_pct": 0.0,
        "sma_period": 30,
        "ema_period": 10,
        "rsi_period": 14,
        "rsi_entry_max": 70.0,
        "min_relative_volume": 1.2,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)  # type: ignore[arg-type]


def test_engine_take_profit_exit() -> None:
    """Engine enters on volume surge above SMA/EMA and exits at TAKE_PROFIT."""
    df = _make_base_uptrend_df(n_bars=60, base_price=100.0, daily_step=0.05)
    # Trigger entry at even bar 34 with volume surge (2.5x avg)
    entry_idx = 34
    df.iloc[entry_idx, df.columns.get_loc("Volume")] = 2_500_000.0
    # Next bar hits high enough to trigger take_profit
    df.iloc[entry_idx + 1, df.columns.get_loc("High")] = float(df.iloc[entry_idx]["Close"]) + 10.0

    config = _setup_fast_config()
    guardian = RiskGuardian(GuardianConfig(max_asset_exposure_pct=0.20))
    engine = BacktestEngine(guardian=guardian, config=config)

    report = engine.run({"AAPL": df})
    assert isinstance(report, BacktestReport)
    assert len(report.trades) >= 1
    first_trade = report.trades[0]
    assert first_trade.symbol == "AAPL"
    assert first_trade.exit_reason == "TAKE_PROFIT"
    assert first_trade.pnl > 0.0
    assert report.metrics.final_equity > config.initial_capital


def test_engine_trailing_stop_and_stop_loss_exits() -> None:
    """Engine ratchets trailing stop upward on gains and exits on pullback."""
    df = _make_base_uptrend_df(n_bars=50, base_price=100.0, daily_step=0.05)
    entry_idx = 32
    df.iloc[entry_idx, df.columns.get_loc("Volume")] = 2_500_000.0
    base_close = float(df.iloc[entry_idx]["Close"])

    # Bar +1: rallies strongly without hitting wide take-profit -> ratchets stop upward
    df.iloc[entry_idx + 1, df.columns.get_loc("Close")] = base_close + 4.0
    df.iloc[entry_idx + 1, df.columns.get_loc("High")] = base_close + 4.5
    df.iloc[entry_idx + 1, df.columns.get_loc("Low")] = base_close + 0.5
    df.iloc[entry_idx + 1, df.columns.get_loc("Open")] = base_close + 0.5

    # Bar +2: drops below ratcheted trailing stop (still above original entry!)
    df.iloc[entry_idx + 2, df.columns.get_loc("Open")] = base_close + 3.5
    df.iloc[entry_idx + 2, df.columns.get_loc("High")] = base_close + 3.6
    df.iloc[entry_idx + 2, df.columns.get_loc("Low")] = base_close + 1.0
    df.iloc[entry_idx + 2, df.columns.get_loc("Close")] = base_close + 1.2

    config = _setup_fast_config(
        atr_stop_multiplier=1.5,
        atr_take_profit_multiplier=20.0,  # Wide TP so trailing stop triggers first
        max_hold_bars=10,
    )
    engine = BacktestEngine(config=config)
    report = engine.run({"AAPL": df})

    assert len(report.trades) >= 1
    trade = report.trades[0]
    assert trade.exit_reason == "TRAILING_STOP"
    assert trade.stop_price > trade.entry_price
    assert trade.pnl > 0.0

    # Now test initial STOP_LOSS when bar drops immediately after entry
    df_sl = _make_base_uptrend_df(n_bars=45, base_price=100.0, daily_step=0.05)
    df_sl.iloc[entry_idx, df_sl.columns.get_loc("Volume")] = 2_500_000.0
    sl_entry_close = float(df_sl.iloc[entry_idx]["Close"])
    df_sl.iloc[entry_idx + 1, df_sl.columns.get_loc("Open")] = sl_entry_close - 0.2
    df_sl.iloc[entry_idx + 1, df_sl.columns.get_loc("Low")] = sl_entry_close - 10.0
    df_sl.iloc[entry_idx + 1, df_sl.columns.get_loc("Close")] = sl_entry_close - 8.0

    report_sl = engine.run({"AAPL": df_sl})
    assert len(report_sl.trades) >= 1
    assert report_sl.trades[0].exit_reason == "STOP_LOSS"
    assert report_sl.trades[0].pnl < 0.0


def test_engine_time_exit_and_end_of_data() -> None:
    """Engine exits sideways positions at max_hold_bars and closes remaining at END_OF_DATA."""
    df = _make_base_uptrend_df(n_bars=48, base_price=100.0, daily_step=0.01)
    # Keep volume below 1.2x except one entry bar at index 32
    df.iloc[32, df.columns.get_loc("Volume")] = 2_500_000.0
    # Flatten bars 33..47 around bar 32 close so neither SL nor TP hits
    anchor = float(df.iloc[32]["Close"])
    for idx in range(33, 48):
        df.iloc[idx, df.columns.get_loc("Open")] = anchor
        df.iloc[idx, df.columns.get_loc("High")] = anchor + 0.1
        df.iloc[idx, df.columns.get_loc("Low")] = anchor - 0.1
        df.iloc[idx, df.columns.get_loc("Close")] = anchor

    config = _setup_fast_config(
        atr_stop_multiplier=5.0,
        atr_take_profit_multiplier=10.0,
        max_hold_bars=4,
    )
    engine = BacktestEngine(config=config)
    report = engine.run({"AAPL": df})

    assert len(report.trades) == 1
    assert report.trades[0].exit_reason == "TIME_EXIT"

    # Now test END_OF_DATA when remaining bars < max_hold_bars
    df_eod = df.iloc[:35].copy()
    report_eod = engine.run({"AAPL": df_eod})
    assert len(report_eod.trades) == 1
    assert report_eod.trades[0].exit_reason == "END_OF_DATA"


def test_engine_compounding_across_consecutive_wins() -> None:
    """Realized profits compound into larger available_capital and position size on next trade."""
    df = _make_base_uptrend_df(n_bars=70, base_price=50.0, daily_step=0.05)
    df["Volume"] = 1_000_000.0

    # Trade 1 at even bar 34 -> hits TP on bar 35
    t1 = 34
    c1 = float(df.iloc[t1]["Close"])
    df.iloc[t1, df.columns.get_loc("Volume")] = 3_000_000.0
    df.iloc[t1 + 1, df.columns.get_loc("High")] = c1 + 25.0

    # Trade 2 at even bar 44 (price stays above SMA/EMA, RSI <= 70)
    t2 = 44
    df.iloc[t2, df.columns.get_loc("Volume")] = 3_000_000.0
    df.iloc[t2 + 1, df.columns.get_loc("High")] = float(df.iloc[t2]["Close"]) + 60.0

    config = _setup_fast_config(
        initial_capital=10_000.0,
        atr_stop_multiplier=1.0,
        atr_take_profit_multiplier=15.0,
        max_hold_bars=5,
        slippage_pct=0.0,
    )
    # High risk_per_trade so asset exposure cap (20% of compounding equity) governs shares
    guardian = RiskGuardian(
        GuardianConfig(
            risk_per_trade_pct=0.05,
            atr_multiplier=1.0,
            max_asset_exposure_pct=0.50,
            max_sector_exposure_pct=0.80,
        )
    )
    engine = BacktestEngine(guardian=guardian, config=config)
    report = engine.run({"AAPL": df})

    tp_trades = [t for t in report.trades if t.exit_reason == "TAKE_PROFIT"]
    assert len(tp_trades) >= 2
    first_win, second_win = tp_trades[0], tp_trades[1]
    assert first_win.pnl > 0.0
    assert second_win.pnl > 0.0
    # Second trade received larger capital allocation due to compounding realized profit
    assert (second_win.shares * second_win.entry_price) > (
        first_win.shares * first_win.entry_price
    )
    assert report.metrics.final_equity == pytest.approx(
        config.initial_capital + sum(t.pnl for t in report.trades)
    )


def test_engine_guardian_vetoes_high_vix_correlation_and_circuit_breaker() -> None:
    """RiskGuardian blocks entries when VIX > 30, correlation > 0.75, or drawdown breached."""
    df_a = _make_base_uptrend_df(n_bars=50, base_price=100.0, daily_step=0.05)
    df_b = _make_base_uptrend_df(n_bars=50, base_price=100.0, daily_step=0.05)
    df_a.iloc[34, df_a.columns.get_loc("Volume")] = 3_000_000.0
    df_b.iloc[34, df_b.columns.get_loc("Volume")] = 3_000_000.0

    config = _setup_fast_config()
    engine = BacktestEngine(guardian=RiskGuardian(), config=config)

    # 1. VIX > 30 vetoes every entry signal
    vix_high = pd.Series(35.0, index=df_a.index)
    report_vix = engine.run({"AAPL": df_a}, vix_series=vix_high)
    assert report_vix.trades == []
    assert report_vix.vetoed_entries_count >= 1

    # 2. Correlation > 0.75 blocks second correlated symbol on same bar
    vix_ok = pd.Series(18.0, index=df_a.index)
    report_corr = engine.run(
        {"AAPL": df_a, "MSFT": df_b},
        vix_series=vix_ok,
        sector_map={"AAPL": "Tech", "MSFT": "Tech"},
        correlation_matrix={("AAPL", "MSFT"): 0.85},
    )
    entered_symbols = {t.symbol for t in report_corr.trades}
    assert entered_symbols == {"AAPL"}
    assert report_corr.vetoed_entries_count >= 1

    # 3. Circuit breaker: open position drops sharply (>3% portfolio drawdown) same bar as D signal
    df_c = _make_base_uptrend_df(n_bars=50, base_price=100.0, daily_step=0.05)
    df_d = _make_base_uptrend_df(n_bars=50, base_price=100.0, daily_step=0.05)
    df_c.iloc[32, df_c.columns.get_loc("Volume")] = 3_000_000.0
    # Bar 34: C crashes 30% (position value is 20% of equity -> ~6% equity drawdown >= 3%)
    c_crash = float(df_c.iloc[32]["Close"]) * 0.70
    df_c.iloc[34, df_c.columns.get_loc("Open")] = c_crash
    df_c.iloc[34, df_c.columns.get_loc("High")] = c_crash
    df_c.iloc[34, df_c.columns.get_loc("Low")] = c_crash
    df_c.iloc[34, df_c.columns.get_loc("Close")] = c_crash
    # Bar 34: D triggers entry signal, should be vetoed by circuit breaker
    df_d.iloc[34, df_d.columns.get_loc("Volume")] = 3_000_000.0

    cb_guardian = RiskGuardian(
        GuardianConfig(
            risk_per_trade_pct=0.05,
            atr_multiplier=1.0,
            max_asset_exposure_pct=0.20,
            max_daily_drawdown_pct=0.03,
        )
    )
    cb_engine = BacktestEngine(guardian=cb_guardian, config=config)
    report_cb = cb_engine.run({"SYMC": df_c, "SYMD": df_d}, vix_series=vix_ok)
    assert "SYMD" not in {t.symbol for t in report_cb.trades}
    assert report_cb.vetoed_entries_count >= 1
