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
    FundamentalData,
    MarketDataClient,
    PriceValidationResult,
    TechnicalIndicators,
    UniverseConfig,
    UniverseScreener,
    UniverseScreeningResult,
    calculate_atr,
    calculate_avg_volume,
    calculate_ema,
    calculate_rsi,
    calculate_sma,
    compute_return_correlation_matrix,
    select_top_candidates,
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
            return _make_http_response([{"debtToEquityRatio": debt_to_equity}])
        if "earnings" in url or "earning_calendar" in url:
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
    assert indicators.latest_volume == pytest.approx(1_000_000.0)
    assert indicators.avg_volume_20d == pytest.approx(1_000_000.0)
    assert calculate_avg_volume([100.0] * 20 + [500.0], period=20, exclude_latest=True) == pytest.approx(100.0)
    assert calculate_avg_volume([], period=20) == 0.0


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

        secret_resp = MagicMock()
        secret_resp.raise_for_status.side_effect = RuntimeError(
            "402 Client Error: Payment Required for url: https://financialmodelingprep.com/stable/ratios?apikey=SUPERSECRET123&symbol=BAD"
        )
        fmp_err = FmpClient(
            store=store,
            api_key="SUPERSECRET123",
            http_get=MagicMock(return_value=secret_resp),
        )
        with pytest.raises(FmpApiError) as exc_info:
            fmp_err.get_fundamentals("BAD2", now=now)
        assert "SUPERSECRET123" not in str(exc_info.value)
        assert "apikey=***" in str(exc_info.value)


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
        ({"volume": 15_000.0}, "Insufficient liquidity volume"),
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


# ------------------------------------------------------------------
# Batch Quotes, EMA(20), RSI(14), and Candidate Ranking Tests
# ------------------------------------------------------------------


def test_get_quotes_batch_fetches_all_symbols_in_single_request() -> None:
    mock_alpaca = MagicMock()
    mock_alpaca.get_stock_snapshot.return_value = {
        "AAPL": _make_mock_snapshot("AAPL", trade_price=190.0, bid_price=189.9, ask_price=190.1),
        "MSFT": _make_mock_snapshot("MSFT", trade_price=420.0, bid_price=419.8, ask_price=420.2),
    }

    client = MarketDataClient(alpaca_client=mock_alpaca)
    assert client.get_quotes_batch([]) == {}
    mock_alpaca.get_stock_snapshot.assert_not_called()

    quotes = client.get_quotes_batch(["aapl", "MSFT", "AAPL"])
    mock_alpaca.get_stock_snapshot.assert_called_once()
    req_arg = mock_alpaca.get_stock_snapshot.call_args.args[0]
    assert isinstance(req_arg, StockSnapshotRequest)
    assert req_arg.symbol_or_symbols == ["AAPL", "MSFT"]
    assert req_arg.feed == DataFeed.IEX

    assert set(quotes.keys()) == {"AAPL", "MSFT"}
    assert quotes["AAPL"].price == pytest.approx(190.0)
    assert quotes["MSFT"].price == pytest.approx(420.0)


def test_calculate_ema_and_rsi_accuracy_and_zero_loss_safety() -> None:
    # Exact EMA(3) over [10, 11, 12, 13]: SMA(3)=11.0, k=0.5 -> 13*0.5 + 11*0.5 = 12.0
    assert calculate_ema([10.0, 11.0, 12.0, 13.0], period=3) == pytest.approx(12.0)
    with pytest.raises(ValueError, match="Insufficient"):
        calculate_ema([10.0, 11.0], period=3)

    # Monotonic uptrend (zero losses) -> RSI == 100.0 without ZeroDivisionError
    assert calculate_rsi([100.0 + i for i in range(20)], period=14) == pytest.approx(100.0)
    # Monotonic downtrend (zero gains) -> RSI == 0.0
    assert calculate_rsi([100.0 - i for i in range(20)], period=14) == pytest.approx(0.0)
    # Flat closes -> RSI == 50.0
    assert calculate_rsi([100.0] * 20, period=14) == pytest.approx(50.0)

    with pytest.raises(ValueError, match="Insufficient"):
        calculate_rsi([100.0] * 14, period=14)


def test_market_data_client_populates_ema20_and_rsi14() -> None:
    closes = [100.0 + i * 0.5 for i in range(210)]
    stock_df = _make_ohlcv_df(closes, spread=2.0)
    vix_df = _make_ohlcv_df([17.5], spread=0.2)

    def _ticker_factory(sym: str) -> MagicMock:
        ticker = MagicMock()
        ticker.history.return_value = vix_df if sym == "^VIX" else stock_df
        return ticker

    client = MarketDataClient(ticker_factory=_ticker_factory)
    ind = client.get_technical_indicators("NVDA")
    assert ind.ema_20 == pytest.approx(calculate_ema(closes, period=20))
    assert ind.rsi_14 == pytest.approx(100.0)


def test_select_top_candidates_filters_and_ranks_deterministically() -> None:
    def _candidate(
        symbol: str,
        *,
        passed: bool = True,
        price: float = 110.0,
        ema_20: float = 100.0,
        latest_vol: float = 1_500_000.0,
        avg_vol: float = 1_000_000.0,
        rsi_14: float = 55.0,
    ) -> UniverseScreeningResult:
        return UniverseScreeningResult(
            symbol=symbol,
            passed=passed,
            rejection_reasons=[] if passed else ["Failed"],
            validation=PriceValidationResult(
                symbol=symbol,
                alpaca_price=price,
                yfinance_price=price,
                z_score=0.0,
                pct_diff=0.0,
                is_valid=True,
            ),
            indicators=TechnicalIndicators(
                symbol=symbol,
                current_price=price,
                sma_200=90.0,
                atr_14=2.0,
                above_sma_200=True,
                vix=18.0,
                latest_volume=latest_vol,
                avg_volume_20d=avg_vol,
                ema_20=ema_20,
                rsi_14=rsi_14,
            ),
            fundamentals=FundamentalData(
                symbol=symbol,
                free_cash_flow=1_000_000.0,
                debt_to_equity=0.5,
            ),
        )

    candidates = [
        _candidate("FAIL", passed=False, latest_vol=3_000_000.0),
        _candidate("OVERBOUGHT", rsi_14=78.0, latest_vol=1_200_000.0),
        _candidate("BELOW_EMA", price=98.0, ema_20=100.0, latest_vol=1_500_000.0),
        _candidate("TOP1", rsi_14=56.0, latest_vol=2_500_000.0),
        _candidate("TOP2", rsi_14=52.0, latest_vol=1_500_000.0),
    ]

    selected = select_top_candidates(candidates, max_candidates=2)
    assert [c.symbol for c in selected] == ["TOP1", "TOP2"]
    assert select_top_candidates(candidates, max_candidates=0) == []


def test_compute_return_correlation_matrix_and_zero_std_safety() -> None:
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    rng = np.random.default_rng(7)
    base_rets = rng.normal(0.001, 0.015, size=40)
    prices_a = 100.0 * np.cumprod(1.0 + base_rets)
    prices_b = 150.0 * np.cumprod(1.0 + base_rets * 0.95 + rng.normal(0.0, 0.001, size=40))
    prices_c = 80.0 * np.cumprod(1.0 - base_rets)
    flat_prices = np.full(40, 50.0)

    histories = {
        "AAPL": pd.DataFrame({"Close": prices_a}, index=dates),
        "MSFT": pd.DataFrame({"Close": prices_b}, index=dates),
        "SH": pd.DataFrame({"Close": prices_c}, index=dates),
        "FLAT": pd.DataFrame({"Close": flat_prices}, index=dates),
    }
    matrix = compute_return_correlation_matrix(histories, lookback_days=35, min_periods=20)
    assert matrix[("AAPL", "MSFT")] > 0.90
    assert matrix[("AAPL", "SH")] < -0.90
    assert ("AAPL", "FLAT") not in matrix


def test_universe_screener_dual_iex_and_consolidated_liquidity(tmp_path: Path) -> None:
    now = datetime(2026, 11, 18, 10, 30, tzinfo=NY_TIMEZONE)
    # 250K on IEX (>= 150K min_iex_volume) with 1M consolidated 20d avg (>= 500K) passes
    screener, store, _, _ = _build_screener(tmp_path, volume=250_000.0)
    try:
        res = screener.screen_symbol("AAPL", now=now)
        assert res.passed is True
    finally:
        store.close()


def test_market_data_ttl_cache_and_morning_iex_run_rate_projection() -> None:
    from src.universe.screener import _project_intraday_iex_volume

    calls = 0

    class _FakeTicker:
        def history(self, period: str = "1y") -> pd.DataFrame:
            nonlocal calls
            calls += 1
            return pd.DataFrame(
                {"Close": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0], "Volume": [1e6, 1e6]}
            )

    md = MarketDataClient(
        ticker_factory=lambda _sym: _FakeTicker(),
        history_cache_ttl_seconds=60.0,
        vix_cache_ttl_seconds=60.0,
    )
    md.get_historical_ohlcv("AAPL", period="1y")
    md.get_historical_ohlcv("AAPL", period="1y")
    assert calls == 1

    md.get_vix()
    md.get_vix()
    assert calls == 2

    # At 09:48 ET (18m into 390m session), 63,194 IEX volume projects to ~1.37M full-day run-rate
    morning_ny = datetime(2026, 11, 20, 9, 48, tzinfo=NY_TIMEZONE)
    projected = _project_intraday_iex_volume(63_194.0, morning_ny)
    assert projected > 1_000_000.0
    # After close (16:05 ET), volume is not scaled upward
    after_close_ny = datetime(2026, 11, 20, 16, 5, tzinfo=NY_TIMEZONE)
    assert _project_intraday_iex_volume(63_194.0, after_close_ny) == pytest.approx(63_194.0)
