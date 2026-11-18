"""Unit tests for Stage-1 Universe screening, market data validation, and FMP cache."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockSnapshotRequest

from src.storage import NY_TIMEZONE, TradingAuditStore
from src.universe import (
    FmpApiError,
    FmpClient,
    FmpRateLimitError,
    MarketDataClient,
    UniverseConfig,
    UniverseScreener,
    calculate_atr,
    calculate_sma,
    validate_cross_price,
)


def _make_ohlcv_df(
    closes: list[float],
    spread: float = 2.0,
) -> pd.DataFrame:
    """Build a deterministic OHLCV DataFrame for yfinance mocks."""
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    close_arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": close_arr,
            "High": close_arr + spread / 2.0,
            "Low": close_arr - spread / 2.0,
            "Close": close_arr,
            "Volume": np.full(len(closes), 1_000_000.0),
        },
        index=dates,
    )


def _make_mock_snapshot(
    symbol: str = "AAPL",
    trade_price: float = 190.0,
    bid_price: float = 189.90,
    ask_price: float = 190.10,
    volume: float = 2_500_000.0,
    ts: datetime | None = None,
) -> SimpleNamespace:
    timestamp = ts or datetime(2026, 11, 18, 10, 30, tzinfo=NY_TIMEZONE)
    return SimpleNamespace(
        symbol=symbol,
        latest_trade=SimpleNamespace(price=trade_price, timestamp=timestamp),
        latest_quote=SimpleNamespace(
            bid_price=bid_price,
            ask_price=ask_price,
            timestamp=timestamp,
        ),
        daily_bar=SimpleNamespace(close=trade_price, volume=volume, timestamp=timestamp),
        previous_daily_bar=SimpleNamespace(close=trade_price, volume=volume, timestamp=timestamp),
    )


def _make_http_response(payload: object, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _make_fmp_http_get(
    fcf: float = 10_000_000_000.0,
    debt_to_equity: float = 0.75,
    earnings_dates: list[str] | None = None,
) -> MagicMock:
    dates = earnings_dates if earnings_dates is not None else ["2026-12-01", "2027-02-28"]

    def _side_effect(url: str, params: dict | None = None, timeout: float = 10.0) -> MagicMock:
        if "cash-flow-statement" in url:
            return _make_http_response([{"freeCashFlow": fcf}])
        if "ratios" in url:
            return _make_http_response([{"debtEquityRatio": debt_to_equity}])
        if "earning_calendar" in url:
            return _make_http_response([{"date": d} for d in dates])
        raise AssertionError(f"Unexpected FMP URL: {url}")

    return MagicMock(side_effect=_side_effect)


# ------------------------------------------------------------------
# Alpaca IEX Snapshot & Spread Tests
# ------------------------------------------------------------------


def test_alpaca_iex_snapshot_parsing_and_spread_calculation() -> None:
    mock_alpaca = MagicMock()
    snapshot = _make_mock_snapshot(
        symbol="AAPL",
        trade_price=100.0,
        bid_price=99.90,
        ask_price=100.10,
        volume=1_250_000.0,
    )
    mock_alpaca.get_stock_snapshot.return_value = {"AAPL": snapshot}

    client = MarketDataClient(alpaca_client=mock_alpaca)
    quote = client.get_quote("aapl")

    mock_alpaca.get_stock_snapshot.assert_called_once()
    req_arg = mock_alpaca.get_stock_snapshot.call_args.args[0]
    assert isinstance(req_arg, StockSnapshotRequest)
    assert req_arg.symbol_or_symbols == "AAPL"
    assert req_arg.feed == DataFeed.IEX

    assert quote.symbol == "AAPL"
    assert quote.price == pytest.approx(100.0)
    assert quote.bid_price == pytest.approx(99.90)
    assert quote.ask_price == pytest.approx(100.10)
    # spread = (100.10 - 99.90) / 100.00 = 0.002
    assert quote.spread_pct == pytest.approx(0.002)
    assert quote.volume == pytest.approx(1_250_000.0)
    assert quote.source == "alpaca_iex"
    assert quote.timestamp.tzinfo == NY_TIMEZONE


def test_alpaca_snapshot_handles_zero_bid_ask_without_zero_division() -> None:
    mock_alpaca = MagicMock()
    snapshot = _make_mock_snapshot(
        symbol="MSFT",
        trade_price=400.0,
        bid_price=0.0,
        ask_price=0.0,
        volume=900_000.0,
    )
    mock_alpaca.get_stock_snapshot.return_value = {"MSFT": snapshot}

    client = MarketDataClient(alpaca_client=mock_alpaca)
    quote = client.get_quote("MSFT")

    assert quote.price == pytest.approx(400.0)
    assert quote.spread_pct == 0.0


# ------------------------------------------------------------------
# Cross-Price 3-Sigma Outlier Detection Tests
# ------------------------------------------------------------------


def test_validate_cross_price_passes_within_3_sigma() -> None:
    # 2% diff / 1.5% std = 1.333σ <= 3.0σ
    res = validate_cross_price(
        symbol="AAPL",
        alpaca_price=102.0,
        yfinance_price=100.0,
        return_std=0.015,
        sigma_threshold=3.0,
    )
    assert res.is_valid is True
    assert res.pct_diff == pytest.approx(0.02)
    assert res.z_score == pytest.approx(0.02 / 0.015)
    assert res.alert_reason is None


def test_validate_cross_price_blocks_above_3_sigma() -> None:
    # 6% diff / 1.5% std = 4.0σ > 3.0σ
    res = validate_cross_price(
        symbol="AAPL",
        alpaca_price=106.0,
        yfinance_price=100.0,
        return_std=0.015,
        sigma_threshold=3.0,
    )
    assert res.is_valid is False
    assert res.pct_diff == pytest.approx(0.06)
    assert res.z_score == pytest.approx(4.0)
    assert res.alert_reason is not None
    assert "3.00σ" in res.alert_reason


@pytest.mark.parametrize(
    ("alpaca_price", "yfinance_price", "return_std", "expected_valid"),
    [
        (100.0, 100.0, 0.0, True),
        (102.0, 100.0, 0.0, True),   # 2% <= fallback 3%
        (105.0, 100.0, 0.0, False),  # 5% > fallback 3%
        (105.0, 100.0, None, False),
        (0.0, 100.0, 0.02, False),
        (100.0, 0.0, 0.02, False),
        (-10.0, 100.0, 0.02, False),
        (float("nan"), 100.0, 0.02, False),
    ],
)
def test_validate_cross_price_zero_std_and_zero_price_safety(
    alpaca_price: float,
    yfinance_price: float,
    return_std: float | None,
    expected_valid: bool,
) -> None:
    res = validate_cross_price(
        symbol="NVDA",
        alpaca_price=alpaca_price,
        yfinance_price=yfinance_price,
        return_std=return_std,
        sigma_threshold=3.0,
        fallback_pct_threshold=0.03,
    )
    assert res.is_valid is expected_valid


# ------------------------------------------------------------------
# SMA_200, ATR(14), and ^VIX Tests
# ------------------------------------------------------------------


def test_calculate_sma_and_atr_exact_and_validation_guards() -> None:
    closes = [float(i) for i in range(1, 205)]
    sma_200 = calculate_sma(closes, period=200)
    # Trailing 200 elements are 5..204 -> mean is (5 + 204) / 2 = 104.5
    assert sma_200 == pytest.approx(104.5)

    with pytest.raises(ValueError, match="Insufficient"):
        calculate_sma([10.0] * 50, period=200)

    # Constant 2.0 daily range with zero close-to-close gaps -> ATR(14) == 2.0
    n = 30
    flat_closes = [100.0] * n
    flat_highs = [101.0] * n
    flat_lows = [99.0] * n
    assert calculate_atr(flat_highs, flat_lows, flat_closes, period=14) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="equal lengths"):
        calculate_atr([101.0] * 15, [99.0] * 14, [100.0] * 15, period=14)


def test_market_data_client_yfinance_indicators_and_vix() -> None:
    closes = [100.0 + i * 0.25 for i in range(220)]
    stock_df = _make_ohlcv_df(closes, spread=2.0)
    vix_df = _make_ohlcv_df([17.8, 18.25], spread=0.5)

    def _ticker_factory(sym: str) -> MagicMock:
        ticker = MagicMock()
        if sym == "^VIX":
            ticker.history.return_value = vix_df
        else:
            ticker.history.return_value = stock_df
        return ticker

    client = MarketDataClient(ticker_factory=_ticker_factory)
    vix = client.get_vix()
    assert vix == pytest.approx(18.25)

    indicators = client.get_technical_indicators("AAPL")
    assert indicators.symbol == "AAPL"
    assert indicators.current_price == pytest.approx(closes[-1])
    assert indicators.sma_200 == pytest.approx(float(np.mean(closes[-200:])))
    assert indicators.atr_14 == pytest.approx(2.0)
    assert indicators.above_sma_200 is True
    assert indicators.vix == pytest.approx(18.25)


# ------------------------------------------------------------------
# FMP Cache & 250 req/day Quota Enforcement Tests
# ------------------------------------------------------------------


def test_fmp_cache_hit_miss_and_ny_day_rollover(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 11, 18, 10, 0, tzinfo=NY_TIMEZONE)
    mock_http = _make_fmp_http_get(
        fcf=5_500_000_000.0,
        debt_to_equity=0.42,
        earnings_dates=["2026-12-10"],
    )

    with TradingAuditStore(db_path) as store:
        fmp = FmpClient(store=store, api_key="test-key", http_get=mock_http)

        # 1. Cache miss -> makes 3 outbound FMP requests and caches result
        first = fmp.get_fundamentals("aapl", now=now)
        assert first.symbol == "AAPL"
        assert first.free_cash_flow == pytest.approx(5_500_000_000.0)
        assert first.debt_to_equity == pytest.approx(0.42)
        assert first.earnings_dates == ["2026-12-10"]
        assert first.from_cache is False
        assert mock_http.call_count == 3
        assert store.get_fmp_requests_today(now=now) == 3

        # 2. Same-day cache hit -> zero outbound HTTP requests
        second = fmp.get_fundamentals("AAPL", now=now + timedelta(hours=2))
        assert second.from_cache is True
        assert second.free_cash_flow == pytest.approx(5_500_000_000.0)
        assert second.debt_to_equity == pytest.approx(0.42)
        assert second.earnings_dates == ["2026-12-10"]
        assert mock_http.call_count == 3
        assert store.get_fmp_requests_today(now=now) == 3

        # 3. Next NY day -> cache expires, hits FMP again
        next_day = now + timedelta(days=1)
        third = fmp.get_fundamentals("AAPL", now=next_day)
        assert third.from_cache is False
        assert mock_http.call_count == 6
        assert store.get_fmp_requests_today(now=next_day) == 3


def test_fmp_daily_rate_limit_enforcement_250_requests(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 11, 18, 11, 0, tzinfo=NY_TIMEZONE)
    mock_http = _make_fmp_http_get()

    with TradingAuditStore(db_path) as store:
        for i in range(250):
            store.record_fmp_request("ratios", symbol=f"SYM{i}", now=now)

        fmp = FmpClient(
            store=store,
            api_key="test-key",
            http_get=mock_http,
            daily_limit=250,
        )

        with pytest.raises(FmpRateLimitError, match="250/250"):
            fmp.get_fundamentals("MSFT", now=now)

        assert mock_http.call_count == 0


def test_fmp_rate_limit_stops_mid_fetch_when_quota_reaches_250(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 11, 18, 11, 0, tzinfo=NY_TIMEZONE)
    mock_http = _make_fmp_http_get()

    with TradingAuditStore(db_path) as store:
        for _ in range(249):
            store.record_fmp_request("ratios", symbol="SEED", now=now)

        fmp = FmpClient(store=store, api_key="test-key", http_get=mock_http, daily_limit=250)
        with pytest.raises(FmpRateLimitError):
            fmp.get_fundamentals("NVDA", now=now)

        # Only 1 request was allowed (249 -> 250), second endpoint blocked before HTTP call
        assert mock_http.call_count == 1
        assert store.get_fmp_requests_today(now=now) == 250


def test_fmp_raises_api_error_on_empty_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 11, 18, 11, 0, tzinfo=NY_TIMEZONE)
    empty_http = MagicMock(return_value=_make_http_response([]))

    with TradingAuditStore(db_path) as store:
        fmp = FmpClient(store=store, api_key="test-key", http_get=empty_http)
        with pytest.raises(FmpApiError):
            fmp.get_fundamentals("BAD", now=now)


# ------------------------------------------------------------------
# UniverseScreener Full Integration Tests
# ------------------------------------------------------------------


def _build_screener(
    tmp_path: Path,
    *,
    alpaca_price: float = 150.0,
    bid_price: float = 149.95,
    ask_price: float = 150.05,
    volume: float = 2_000_000.0,
    yf_closes: list[float] | None = None,
    vix_level: float = 18.0,
    fcf: float = 2_000_000_000.0,
    debt_to_equity: float = 0.6,
    config: UniverseConfig | None = None,
) -> tuple[UniverseScreener, TradingAuditStore, MagicMock, MagicMock]:
    closes = yf_closes or ([100.0 + i * 0.25 for i in range(199)] + [150.0])
    stock_df = _make_ohlcv_df(closes, spread=2.0)
    vix_df = _make_ohlcv_df([vix_level], spread=0.2)

    mock_alpaca = MagicMock()
    mock_alpaca.get_stock_snapshot.return_value = {
        "AAPL": _make_mock_snapshot(
            symbol="AAPL",
            trade_price=alpaca_price,
            bid_price=bid_price,
            ask_price=ask_price,
            volume=volume,
        ),
    }

    yf_calls: list[str] = []

    def _ticker_factory(sym: str) -> MagicMock:
        yf_calls.append(sym)
        ticker = MagicMock()
        ticker.history.return_value = vix_df if sym == "^VIX" else stock_df
        return ticker

    market_data = MarketDataClient(
        alpaca_client=mock_alpaca,
        ticker_factory=_ticker_factory,
    )
    mock_http = _make_fmp_http_get(fcf=fcf, debt_to_equity=debt_to_equity)
    store = TradingAuditStore(tmp_path / "screener_audit.db")
    fmp_client = FmpClient(store=store, api_key="test-key", http_get=mock_http)
    screener = UniverseScreener(market_data=market_data, fmp_client=fmp_client, config=config)
    return screener, store, mock_alpaca, mock_http


def test_universe_screener_approves_healthy_liquid_uptrend_asset(tmp_path: Path) -> None:
    now = datetime(2026, 11, 18, 10, 30, tzinfo=NY_TIMEZONE)
    screener, store, _, _ = _build_screener(tmp_path)
    try:
        result = screener.screen_symbol("AAPL", now=now)
        assert result.passed is True
        assert result.rejection_reasons == []
        assert result.symbol == "AAPL"
        assert result.quote is not None and result.quote.spread_pct < 0.005
        assert result.validation is not None and result.validation.is_valid is True
        assert result.indicators is not None and result.indicators.above_sma_200 is True
        assert result.fundamentals is not None and result.fundamentals.free_cash_flow > 0
    finally:
        store.close()


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        ({"alpaca_price": 210.0}, "Cross-price discrepancy"),
        ({"volume": 100_000.0}, "Insufficient liquidity volume"),
        ({"bid_price": 148.0, "ask_price": 152.0}, "Excessive bid/ask spread"),
        ({"fcf": -50_000_000.0}, "free cash flow"),
        ({"fcf": 0.0}, "free cash flow"),
        ({"debt_to_equity": 2.75}, "Excessive debt-to-equity"),
        (
            {
                "alpaca_price": 90.0,
                "bid_price": 89.95,
                "ask_price": 90.05,
                "yf_closes": [120.0] * 199 + [90.0],
            },
            "Downtrend",
        ),
    ],
)
def test_universe_screener_rejects_each_disqualifying_condition(
    tmp_path: Path,
    overrides: dict,
    expected_fragment: str,
) -> None:
    now = datetime(2026, 11, 18, 10, 30, tzinfo=NY_TIMEZONE)
    screener, store, _, _ = _build_screener(tmp_path, **overrides)
    try:
        result = screener.screen_symbol("AAPL", now=now)
        assert result.passed is False
        assert any(expected_fragment in reason for reason in result.rejection_reasons)
    finally:
        store.close()


def test_universe_screener_collects_multiple_rejections_and_batches_vix(tmp_path: Path) -> None:
    now = datetime(2026, 11, 18, 10, 30, tzinfo=NY_TIMEZONE)
    screener, store, mock_alpaca, _ = _build_screener(
        tmp_path,
        alpaca_price=190.0,
        bid_price=145.0,
        ask_price=155.0,
        volume=10_000.0,
        fcf=-1.0,
        debt_to_equity=5.0,
        yf_closes=[160.0] * 200,
    )
    try:
        # Add MSFT snapshot so screen_universe handles multiple tickers
        mock_alpaca.get_stock_snapshot.side_effect = lambda req: {
            req.symbol_or_symbols: _make_mock_snapshot(
                symbol=req.symbol_or_symbols,
                trade_price=190.0,
                bid_price=145.0,
                ask_price=155.0,
                volume=10_000.0,
            )
        }
        results = screener.screen_universe(["AAPL", "MSFT"], now=now)
        assert len(results) == 2
        for res in results:
            assert res.passed is False
            assert len(res.rejection_reasons) >= 4
    finally:
        store.close()
