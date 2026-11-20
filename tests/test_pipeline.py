"""Unit and mocked end-to-end pipeline tests for notifications, clock, and orchestrator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import httpx
import pytest

from main import (
    NY_TIMEZONE,
    PipelineCycleReport,
    TradingBotOrchestrator,
    check_news_confirmation_window,
    is_market_open,
    main,
    parse_args,
)
from src.analyst import (
    AnalystConfig,
    AnalystEvaluationResult,
    AnalystReport,
    NewsItem,
)
from src.execution import (
    AlpacaExecutionClient,
    ExecutionConfig,
    ExecutionOrderResult,
    ExitEvaluationResult,
    ManagedPosition,
    PositionManager,
)
from src.guardian import GuardianConfig, RiskGuardian
from src.notifications import TelegramNotifier
from src.storage import TradingAuditStore
from src.universe import (
    FundamentalData,
    MarketQuote,
    PriceValidationResult,
    TechnicalIndicators,
    UniverseScreeningResult,
)


@dataclass(frozen=True, slots=True)
class _FakeManagedPosition:
    symbol: str
    qty: int
    entry_price: float
    current_stop_price: float
    atr_14: float
    opened_at: datetime
    highest_price: float | None = None
    stop_order_id: str | None = None
    client_order_id: str | None = None


@dataclass(frozen=True, slots=True)
class _FakeOrderResult:
    submitted: bool
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: str
    qty: int
    limit_price: float
    stop_price: float | None
    order_type: str
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class _FakeExitResult:
    symbol: str
    action: str
    should_exit: bool
    new_stop_price: float | None
    exit_reason: str
    order_result: _FakeOrderResult | None = None


def _make_screening_result(
    symbol: str = "AAPL",
    passed: bool = True,
    price: float = 200.0,
    ask_price: float = 200.10,
    atr_14: float = 3.0,
    vix: float = 18.0,
    volume: float = 2_000_000.0,
    earnings_dates: list[str] | None = None,
) -> UniverseScreeningResult:
    return UniverseScreeningResult(
        symbol=symbol,
        passed=passed,
        rejection_reasons=[] if passed else ["Stage-1 screening failed"],
        quote=MarketQuote(
            symbol=symbol,
            price=price,
            bid_price=price - 0.10,
            ask_price=ask_price,
            spread_pct=0.001,
            volume=volume,
        ),
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
            sma_200=175.0,
            atr_14=atr_14,
            above_sma_200=True,
            vix=vix,
        ),
        fundamentals=FundamentalData(
            symbol=symbol,
            free_cash_flow=10_000_000_000.0,
            debt_to_equity=0.8,
            earnings_dates=earnings_dates or [],
        ),
    )


def _make_news(published_at: datetime, source: str = "Reuters") -> NewsItem:
    return NewsItem(
        headline=f"Institutional flow update from {source}",
        summary="Strong earnings momentum confirmed.",
        source=source,
        url=f"https://example.com/{source.lower()}",
        published_at=published_at,
        provider="finnhub",
    )


def _make_analyst_result(
    symbol: str = "AAPL",
    approved: bool = True,
    confidence: float = 0.90,
    sources_count: int = 4,
    signal_id: int = 1,
) -> AnalystEvaluationResult:
    report = AnalystReport(
        ticker=symbol,
        sentiment_score=0.75,
        confidence_score=confidence,
        independent_sources_count=sources_count,
        volume_confirmed=True,
        manipulation_risk=False,
        rationale="Multi-source bullish consensus corroborated by volume.",
    )
    return AnalystEvaluationResult(
        ticker=symbol,
        approved=approved,
        rejection_reasons=[] if approved else ["confidence_score below threshold"],
        report=report,
        relative_volume=1.8,
        news_count=4,
        sources=["Reuters", "Bloomberg", "Benzinga", "WSJ"][:sources_count],
        signal_id=signal_id,
    )


# ---------------------------------------------------------------------------
# TelegramNotifier fail-safe & formatting tests
# ---------------------------------------------------------------------------


def test_telegram_notifier_missing_credentials_returns_false_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    post_spy = MagicMock()

    notifier = TelegramNotifier(http_post=post_spy)
    assert notifier.is_configured is False
    assert notifier.send_message("hello") is False
    assert notifier.notify_trade_executed("AAPL", "BUY", 10, 200.0) is False
    post_spy.assert_not_called()

    notifier_partial = TelegramNotifier(bot_token="token-only", chat_id="", http_post=post_spy)
    assert notifier_partial.send_message("hello") is False
    post_spy.assert_not_called()


def test_telegram_notifier_network_exception_and_timeout_are_failsafe() -> None:
    def failing_post(*args: object, **kwargs: object) -> object:
        raise httpx.TimeoutException("Request timed out")

    notifier = TelegramNotifier(
        bot_token="123:abc",
        chat_id="999",
        http_post=failing_post,
    )
    assert notifier.send_message("Trade alert") is False

    class FailingAsyncClient:
        def __init__(self, timeout: float = 10.0) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FailingAsyncClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> object:
            raise ConnectionError("Network unreachable")

    notifier_factory = TelegramNotifier(
        bot_token="123:abc",
        chat_id="999",
        client_factory=FailingAsyncClient,
    )
    assert asyncio.run(notifier_factory.send_message_async("Hello")) is False


def test_telegram_notifier_http_error_and_api_ok_false_return_false() -> None:
    bad_response = MagicMock()
    bad_response.status_code = 502
    bad_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Bad Gateway",
        request=MagicMock(),
        response=MagicMock(),
    )
    notifier_http_err = TelegramNotifier(
        bot_token="123:abc",
        chat_id="999",
        http_post=lambda *a, **kw: bad_response,
    )
    assert notifier_http_err.send_message("Test") is False

    api_false_response = MagicMock()
    api_false_response.status_code = 200
    api_false_response.raise_for_status.return_value = None
    api_false_response.json.return_value = {"ok": False, "description": "Bad Request"}
    notifier_api_err = TelegramNotifier(
        bot_token="123:abc",
        chat_id="999",
        http_post=lambda *a, **kw: api_false_response,
    )
    assert notifier_api_err.send_message("Test") is False


def test_telegram_notifier_success_and_message_formatting() -> None:
    captured: list[tuple[str, dict[str, object]]] = []

    class FakeAsyncClient:
        def __init__(self, timeout: float = 10.0) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> object:
            captured.append((url, json))
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"ok": True}
            return resp

    notifier = TelegramNotifier(
        bot_token="bot-token-1",
        chat_id="-10042",
        client_factory=FakeAsyncClient,
    )

    assert notifier.notify_trade_executed(
        symbol="aapl",
        side="buy",
        qty=25,
        limit_price=200.50,
        stop_price=194.50,
        dry_run=True,
    ) is True
    assert notifier.notify_position_closed(
        symbol="NVDA",
        action="EMERGENCY_EXIT",
        qty=10,
        limit_price=125.00,
        reason="Sudden 5-min drop",
    ) is True
    assert notifier.notify_guardian_veto(
        symbol="MSFT",
        veto_reasons=["VIX 34.50 exceeds maximum allowed threshold 30.00"],
        vix=34.5,
    ) is True
    assert notifier.notify_circuit_breaker(
        drawdown_pct=0.035,
        starting_equity=100_000.0,
        daily_pnl=-3500.0,
    ) is True

    assert len(captured) == 4
    url0, payload0 = captured[0]
    assert url0 == "https://api.telegram.org/botbot-token-1/sendMessage"
    assert payload0["chat_id"] == "-10042"
    assert "[DRY-RUN] TRADE EXECUTED (LIMIT)" in str(payload0["text"])
    assert "AAPL" in str(payload0["text"])
    assert "$200.50" in str(payload0["text"])
    assert "$194.50" in str(payload0["text"])

    assert "POSITION CLOSED (EMERGENCY_EXIT)" in str(captured[1][1]["text"])
    assert "GUARDIAN VETO" in str(captured[2][1]["text"])
    assert "VIX: 34.50" in str(captured[2][1]["text"])
    assert "CIRCUIT BREAKER TRIGGERED" in str(captured[3][1]["text"])
    assert "3.50%" in str(captured[3][1]["text"])


# ---------------------------------------------------------------------------
# Market hours (America/New_York) & timezone conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dt", "expected"),
    [
        # Regular Wednesday in EST (UTC-5)
        (datetime(2026, 11, 18, 9, 29, 59, tzinfo=NY_TIMEZONE), False),
        (datetime(2026, 11, 18, 9, 30, 0, tzinfo=NY_TIMEZONE), True),
        (datetime(2026, 11, 18, 12, 15, 0, tzinfo=NY_TIMEZONE), True),
        (datetime(2026, 11, 18, 15, 59, 59, tzinfo=NY_TIMEZONE), True),
        (datetime(2026, 11, 18, 16, 0, 0, tzinfo=NY_TIMEZONE), False),
        # Weekends (Saturday & Sunday)
        (datetime(2026, 11, 21, 12, 0, 0, tzinfo=NY_TIMEZONE), False),
        (datetime(2026, 11, 22, 12, 0, 0, tzinfo=NY_TIMEZONE), False),
        # UTC conversion in standard time (Nov 18, 2026: EST = UTC-5)
        (datetime(2026, 11, 18, 14, 29, 0, tzinfo=timezone.utc), False),
        (datetime(2026, 11, 18, 14, 30, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 11, 18, 21, 0, 0, tzinfo=timezone.utc), False),
        # Naive datetimes interpreted as America/New_York
        (datetime(2026, 11, 18, 9, 30, 0), True),
        (datetime(2026, 11, 18, 16, 0, 0), False),
        # Other timezone converted to America/New_York (London UTC+0 in Nov)
        (datetime(2026, 11, 18, 15, 0, 0, tzinfo=ZoneInfo("Europe/London")), True),
    ],
)
def test_is_market_open_boundaries_and_timezones(dt: datetime, expected: bool) -> None:
    assert is_market_open(dt) is expected


# ---------------------------------------------------------------------------
# News confirmation delay window (2-5 min) tests
# ---------------------------------------------------------------------------


def test_news_confirmation_window_enforces_2_to_5_minute_delay() -> None:
    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)

    # No news -> allowed
    ok, _ = check_news_confirmation_window([], now=now_ny)
    assert ok is True

    # Breaking news 90 seconds ago (< 2 min) -> deferred
    recent_news = [_make_news(now_ny - timedelta(seconds=90), source="Reuters")]
    ok, reason = check_news_confirmation_window(recent_news, now=now_ny)
    assert ok is False
    assert "confirmation window active" in reason.lower()

    # Mixed older + fresh breaking news -> latest item governs -> deferred
    mixed_news = [
        _make_news(now_ny - timedelta(minutes=20), source="Bloomberg"),
        _make_news(now_ny - timedelta(seconds=119), source="Benzinga"),
    ]
    ok, _ = check_news_confirmation_window(mixed_news, now=now_ny)
    assert ok is False

    # Exactly 2.0 minutes old -> allowed
    confirmed_2m = [_make_news(now_ny - timedelta(minutes=2), source="Reuters")]
    ok, _ = check_news_confirmation_window(confirmed_2m, now=now_ny)
    assert ok is True

    # 4 minutes old -> allowed
    confirmed_4m = [_make_news(now_ny - timedelta(minutes=4), source="WSJ")]
    ok, _ = check_news_confirmation_window(confirmed_4m, now=now_ny)
    assert ok is True


# ---------------------------------------------------------------------------
# End-to-end mocked pipeline tests
# ---------------------------------------------------------------------------


def _build_mock_orchestrator(
    audit_store: TradingAuditStore,
    screening_result: UniverseScreeningResult,
    news_items: list[NewsItem],
    analyst_result: AnalystEvaluationResult,
    guardian_config: GuardianConfig | None = None,
    execution_result: _FakeOrderResult | None = None,
    exit_result: _FakeExitResult | None = None,
) -> tuple[TradingBotOrchestrator, dict[str, MagicMock]]:
    screener = MagicMock()
    screener.screen_symbol.return_value = screening_result

    news_aggregator = MagicMock()
    news_aggregator.fetch_news.return_value = news_items

    analyst = MagicMock()
    analyst.config = AnalystConfig()
    analyst.evaluate.return_value = analyst_result

    guardian = RiskGuardian(config=guardian_config or GuardianConfig())

    execution_client = MagicMock()
    execution_client.submit_limit_buy.return_value = execution_result or _FakeOrderResult(
        submitted=True,
        client_order_id="cid-aapl-1",
        broker_order_id="bid-aapl-1",
        symbol=screening_result.symbol,
        side="BUY",
        qty=100,
        limit_price=200.10,
        stop_price=194.10,
        order_type="limit",
        status="accepted",
        reason="Limit order submitted",
    )

    position_manager = MagicMock()
    position_manager.evaluate_and_manage.return_value = exit_result or _FakeExitResult(
        symbol="MSFT",
        action="HOLD",
        should_exit=False,
        new_stop_price=395.0,
        exit_reason="Trailing stop held",
        order_result=None,
    )

    notifier = MagicMock(spec=TelegramNotifier)
    notifier.notify_trade_executed.return_value = True
    notifier.notify_position_closed.return_value = True
    notifier.notify_guardian_veto.return_value = True
    notifier.notify_circuit_breaker.return_value = True

    orchestrator = TradingBotOrchestrator(
        screener=screener,
        news_aggregator=news_aggregator,
        analyst=analyst,
        guardian=guardian,
        execution_client=execution_client,
        position_manager=position_manager,
        audit_store=audit_store,
        notifier=notifier,
    )
    mocks = {
        "screener": screener,
        "news_aggregator": news_aggregator,
        "analyst": analyst,
        "execution_client": execution_client,
        "position_manager": position_manager,
        "notifier": notifier,
    }
    return orchestrator, mocks


def test_pipeline_happy_path_live_and_dry_run() -> None:
    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)
    confirmed_news = [
        _make_news(now_ny - timedelta(minutes=4), source="Reuters"),
        _make_news(now_ny - timedelta(minutes=6), source="Bloomberg"),
        _make_news(now_ny - timedelta(minutes=10), source="Benzinga"),
    ]
    store = TradingAuditStore(":memory:")

    # 1) Live-mocked run with open position exit alert
    exit_order = _FakeOrderResult(
        submitted=True,
        client_order_id="cid-exit-msft",
        broker_order_id="bid-exit-msft",
        symbol="MSFT",
        side="SELL",
        qty=15,
        limit_price=402.0,
        stop_price=None,
        order_type="limit",
        status="accepted",
        reason="Time-window exit",
    )
    exit_eval = _FakeExitResult(
        symbol="MSFT",
        action="TIME_EXIT",
        should_exit=True,
        new_stop_price=None,
        exit_reason="Max holding window reached",
        order_result=exit_order,
    )
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL", vix=18.5),
        news_items=confirmed_news,
        analyst_result=_make_analyst_result("AAPL", approved=True, confidence=0.88, sources_count=4),
        exit_result=exit_eval,
    )
    open_pos = _FakeManagedPosition(
        symbol="MSFT",
        qty=15,
        entry_price=400.0,
        current_stop_price=392.0,
        atr_14=4.0,
        opened_at=now_ny - timedelta(hours=2),
    )

    report = orchestrator.run_cycle(
        symbols=["AAPL"],
        now=now_ny,
        dry_run=False,
        open_positions=[open_pos],
        current_prices={"MSFT": 402.5, "AAPL": 200.0},
        bid_prices={"MSFT": 402.0},
    )

    assert isinstance(report, PipelineCycleReport)
    assert report.skipped is False
    assert report.market_open is True
    assert len(report.exit_results) == 1
    assert len(report.ticker_outcomes) == 1

    outcome = report.ticker_outcomes[0]
    assert outcome.approved is True
    assert outcome.status == "executed"
    assert outcome.stage_reached == "execution"
    assert outcome.guardian_decision is not None
    assert outcome.guardian_decision.approved is True

    # Verify Limit order submitted & Telegram alerts sent
    mocks["execution_client"].submit_limit_buy.assert_called_once()
    call_kwargs = mocks["execution_client"].submit_limit_buy.call_args.kwargs
    assert call_kwargs["symbol"] == "AAPL"
    assert call_kwargs["qty"] == outcome.guardian_decision.recommended_shares
    assert call_kwargs["dry_run"] is False

    mocks["notifier"].notify_position_closed.assert_called_once()
    mocks["notifier"].notify_trade_executed.assert_called_once()
    mocks["notifier"].notify_guardian_veto.assert_not_called()

    # Verify SQLite Audit log recorded approved decision
    decisions = store.get_guardian_decisions(symbol="AAPL")
    assert len(decisions) == 1
    assert decisions[0].approved is True
    assert decisions[0].vix == 18.5

    # 2) Dry-run simulation mode
    dry_store = TradingAuditStore(":memory:")
    dry_exec_result = _FakeOrderResult(
        submitted=False,
        client_order_id="cid-dry-aapl",
        broker_order_id=None,
        symbol="AAPL",
        side="BUY",
        qty=100,
        limit_price=200.10,
        stop_price=194.10,
        order_type="limit",
        status="dry_run",
        reason="Dry-run simulation",
    )
    dry_orch, dry_mocks = _build_mock_orchestrator(
        audit_store=dry_store,
        screening_result=_make_screening_result("AAPL", vix=18.5),
        news_items=confirmed_news,
        analyst_result=_make_analyst_result("AAPL", approved=True),
        execution_result=dry_exec_result,
    )
    dry_report = dry_orch.run_cycle(symbols=["AAPL"], now=now_ny, dry_run=True)
    assert dry_report.ticker_outcomes[0].approved is True
    assert dry_report.ticker_outcomes[0].status == "dry_run_executed"
    assert dry_mocks["execution_client"].submit_limit_buy.call_args.kwargs["dry_run"] is True
    assert dry_mocks["notifier"].notify_trade_executed.call_args.kwargs["dry_run"] is True


def test_guardian_vix_veto_overrides_gemini_approval() -> None:
    now_ny = datetime(2026, 11, 18, 11, 0, 0, tzinfo=NY_TIMEZONE)
    confirmed_news = [_make_news(now_ny - timedelta(minutes=5))]
    store = TradingAuditStore(":memory:")

    # Gemini approves strongly (0.90 confidence, 4 sources), but VIX = 35 > 30
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL", vix=35.0),
        news_items=confirmed_news,
        analyst_result=_make_analyst_result("AAPL", approved=True, confidence=0.90, sources_count=4),
    )

    report = orchestrator.run_cycle(symbols=["AAPL"], now=now_ny)
    outcome = report.ticker_outcomes[0]

    assert outcome.approved is False
    assert outcome.status == "vetoed_guardian"
    assert outcome.stage_reached == "guardian"
    assert any("VIX 35.00 exceeds" in reason for reason in outcome.reasons)

    # Order execution MUST NOT be called
    mocks["execution_client"].submit_limit_buy.assert_not_called()
    mocks["notifier"].notify_trade_executed.assert_not_called()
    mocks["notifier"].notify_guardian_veto.assert_called_once()
    mocks["notifier"].notify_circuit_breaker.assert_not_called()

    # Veto MUST be persisted in SQLite
    decisions = store.get_guardian_decisions(symbol="AAPL", approved=False)
    assert len(decisions) == 1
    assert decisions[0].approved is False
    assert decisions[0].vix == 35.0
    assert any("VIX 35.00" in r for r in decisions[0].veto_reasons)


def test_guardian_circuit_breaker_veto_overrides_gemini_and_alerts_telegram() -> None:
    now_ny = datetime(2026, 11, 18, 11, 0, 0, tzinfo=NY_TIMEZONE)
    confirmed_news = [_make_news(now_ny - timedelta(minutes=5))]
    store = TradingAuditStore(":memory:")

    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("NVDA", vix=19.0),
        news_items=confirmed_news,
        analyst_result=_make_analyst_result("NVDA", approved=True, confidence=0.92, sources_count=4),
    )

    # 4% daily loss >= 3% circuit breaker threshold
    report = orchestrator.run_cycle(
        symbols=["NVDA"],
        now=now_ny,
        starting_equity=100_000.0,
        daily_pnl=-4_000.0,
    )
    outcome = report.ticker_outcomes[0]
    assert outcome.approved is False
    assert outcome.status == "vetoed_guardian"
    assert any("circuit breaker" in r.lower() for r in outcome.reasons)

    mocks["execution_client"].submit_limit_buy.assert_not_called()
    mocks["notifier"].notify_circuit_breaker.assert_called_once()
    mocks["notifier"].notify_guardian_veto.assert_called_once()

    cb_kwargs = mocks["notifier"].notify_circuit_breaker.call_args.kwargs
    assert cb_kwargs["drawdown_pct"] == pytest.approx(0.04)
    assert cb_kwargs["starting_equity"] == 100_000.0
    assert cb_kwargs["daily_pnl"] == -4_000.0

    decisions = store.get_guardian_decisions(symbol="NVDA", approved=False)
    assert len(decisions) == 1
    assert decisions[0].approved is False


def test_guardian_earnings_blackout_veto_overrides_gemini() -> None:
    now_ny = datetime(2026, 11, 18, 10, 0, 0, tzinfo=NY_TIMEZONE)
    confirmed_news = [_make_news(now_ny - timedelta(minutes=5))]
    store = TradingAuditStore(":memory:")

    # Earnings next day (2026-11-19 -> 14h ahead <= 24h blackout)
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result(
            "AAPL",
            vix=16.0,
            earnings_dates=["2026-11-19"],
        ),
        news_items=confirmed_news,
        analyst_result=_make_analyst_result("AAPL", approved=True, confidence=0.90),
    )

    report = orchestrator.run_cycle(symbols=["AAPL"], now=now_ny)
    outcome = report.ticker_outcomes[0]
    assert outcome.approved is False
    assert outcome.status == "vetoed_guardian"
    assert any("blackout" in r.lower() for r in outcome.reasons)
    mocks["execution_client"].submit_limit_buy.assert_not_called()
    mocks["notifier"].notify_guardian_veto.assert_called_once()


def test_news_confirmation_window_defers_before_analyst_and_execution() -> None:
    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)
    breaking_news = [_make_news(now_ny - timedelta(seconds=45), source="Reuters")]
    store = TradingAuditStore(":memory:")

    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL"),
        news_items=breaking_news,
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )

    report = orchestrator.run_cycle(symbols=["AAPL"], now=now_ny)
    outcome = report.ticker_outcomes[0]
    assert outcome.approved is False
    assert outcome.stage_reached == "news_confirmation"
    assert outcome.status == "deferred_news_window"
    mocks["analyst"].evaluate.assert_not_called()
    mocks["execution_client"].submit_limit_buy.assert_not_called()


def test_closed_market_aborts_cycle_without_calling_stages() -> None:
    pre_market_ny = datetime(2026, 11, 18, 8, 45, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(":memory:")
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL"),
        news_items=[],
        analyst_result=_make_analyst_result("AAPL"),
    )

    report = orchestrator.run_cycle(
        symbols=["AAPL", "MSFT"],
        now=pre_market_ny,
        enforce_market_hours=True,
    )

    assert report.skipped is True
    assert report.market_open is False
    assert report.ticker_outcomes == []
    mocks["screener"].screen_symbol.assert_not_called()
    mocks["news_aggregator"].fetch_news.assert_not_called()
    mocks["analyst"].evaluate.assert_not_called()
    mocks["position_manager"].evaluate_and_manage.assert_not_called()
    mocks["execution_client"].submit_limit_buy.assert_not_called()


def test_cli_parse_args_and_main_loop_interrupt(tmp_path: object) -> None:
    args = parse_args(
        ["--tickers", "AAPL", "NVDA", "--dry-run", "--ignore-market-hours", "--interval-seconds", "15"]
    )
    assert args.tickers == ["AAPL", "NVDA"]
    assert args.dry_run is True
    assert args.ignore_market_hours is True
    assert args.loop is False
    assert args.interval_seconds == 15.0

    mock_orch = MagicMock(spec=TradingBotOrchestrator)
    mock_orch.run_cycle.return_value = PipelineCycleReport(
        timestamp=datetime(2026, 11, 18, 12, 0, tzinfo=NY_TIMEZONE),
        market_open=True,
        skipped=False,
        dry_run=True,
    )

    # Single shot
    exit_code = main(
        argv=["--tickers", "AAPL", "--dry-run", "--ignore-market-hours"],
        orchestrator=mock_orch,
    )
    assert exit_code == 0
    mock_orch.run_cycle.assert_called_once_with(
        symbols=["AAPL"],
        dry_run=True,
        enforce_market_hours=False,
    )

    # Loop stops cleanly on KeyboardInterrupt
    mock_orch.run_cycle.reset_mock()
    sleep_calls = 0

    def interrupt_on_second_sleep(seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        assert seconds == 5.0
        if sleep_calls >= 2:
            raise KeyboardInterrupt

    exit_code_loop = main(
        argv=["--tickers", "AAPL", "--loop", "--interval-seconds", "5", "--ignore-market-hours"],
        orchestrator=mock_orch,
        sleep_fn=interrupt_on_second_sleep,
    )
    assert exit_code_loop == 0
    assert mock_orch.run_cycle.call_count == 2


def test_pipeline_with_real_execution_client_and_position_manager_dry_run() -> None:
    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)
    confirmed_news = [_make_news(now_ny - timedelta(minutes=5), source="Reuters")]
    store = TradingAuditStore(":memory:")

    exec_client = AlpacaExecutionClient(
        audit_store=store,
        config=ExecutionConfig(paper=True),
    )
    pos_manager = PositionManager(execution_client=exec_client, audit_store=store)

    screener = MagicMock()
    screener.screen_symbol.return_value = _make_screening_result("AAPL", vix=17.0)

    news_aggregator = MagicMock()
    news_aggregator.fetch_news.return_value = confirmed_news

    analyst = MagicMock()
    analyst.config = AnalystConfig()
    analyst.evaluate.return_value = _make_analyst_result("AAPL", approved=True)

    notifier = MagicMock(spec=TelegramNotifier)
    notifier.notify_trade_executed.return_value = True
    notifier.notify_position_closed.return_value = True

    orchestrator = TradingBotOrchestrator(
        screener=screener,
        news_aggregator=news_aggregator,
        analyst=analyst,
        guardian=RiskGuardian(),
        execution_client=exec_client,
        position_manager=pos_manager,
        audit_store=store,
        notifier=notifier,
    )

    flash_crash_pos = ManagedPosition(
        symbol="NVDA",
        qty=10,
        entry_price=100.0,
        current_stop_price=95.0,
        atr_14=2.0,
        opened_at=now_ny - timedelta(minutes=3),
    )

    report = orchestrator.run_cycle(
        symbols=["AAPL"],
        now=now_ny,
        dry_run=True,
        open_positions=[flash_crash_pos],
        current_prices={"NVDA": 96.0, "AAPL": 200.0},
        bid_prices={"NVDA": 95.9},
    )

    assert len(report.exit_results) == 1
    assert isinstance(report.exit_results[0], ExitEvaluationResult)
    assert report.exit_results[0].action == "EMERGENCY_EXIT"
    assert report.exit_results[0].should_exit is True
    notifier.notify_position_closed.assert_called_once()

    assert len(report.ticker_outcomes) == 1
    outcome = report.ticker_outcomes[0]
    assert outcome.approved is True
    assert outcome.status == "dry_run_executed"
    assert isinstance(outcome.order_result, ExecutionOrderResult)
    assert outcome.order_result.status == "dry_run"
    notifier.notify_trade_executed.assert_called_once()

    # Verify both orders (emergency sell + dry-run buy) were audited in SQLite
    orders = store.get_orders()
    assert len(orders) == 2


def test_intraday_run_rate_volume_projection_used_by_orchestrator() -> None:
    from main import estimate_intraday_run_rate_volume

    # At 12:45 ET (195 min into 390-min session = 50% elapsed), 30M volume projects to 60M
    midday_ny = datetime(2026, 11, 18, 12, 45, 0, tzinfo=NY_TIMEZONE)
    assert estimate_intraday_run_rate_volume(30_000_000.0, now=midday_ny) == pytest.approx(60_000_000.0)

    # Off-hours returns raw volume without scaling
    after_hours = datetime(2026, 11, 18, 18, 0, 0, tzinfo=NY_TIMEZONE)
    assert estimate_intraday_run_rate_volume(30_000_000.0, now=after_hours) == pytest.approx(30_000_000.0)

    screening = _make_screening_result("AAPL", vix=18.5)
    screening = UniverseScreeningResult(
        symbol="AAPL",
        passed=True,
        rejection_reasons=[],
        quote=screening.quote,
        validation=screening.validation,
        indicators=TechnicalIndicators(
            symbol="AAPL",
            current_price=200.0,
            sma_200=180.0,
            atr_14=3.0,
            above_sma_200=True,
            vix=18.5,
            latest_volume=30_000_000.0,
            avg_volume_20d=40_000_000.0,
        ),
        fundamentals=screening.fundamentals,
    )
    store = TradingAuditStore(":memory:")
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=screening,
        news_items=[_make_news(midday_ny - timedelta(minutes=5))],
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )

    orchestrator.run_cycle(symbols=["AAPL"], now=midday_ny, dry_run=True)
    eval_kwargs = mocks["analyst"].evaluate.call_args.kwargs
    assert eval_kwargs["current_volume"] == pytest.approx(60_000_000.0)
    assert eval_kwargs["avg_volume_20d"] == pytest.approx(40_000_000.0)


def test_orchestrator_passes_bracket_take_profit_price() -> None:
    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(":memory:")
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL", price=200.0, ask_price=200.10, atr_14=4.0),
        news_items=[_make_news(now_ny - timedelta(minutes=5), source="Reuters")],
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )

    report = orchestrator.run_cycle(symbols=["AAPL"], now=now_ny, dry_run=False)
    assert report.ticker_outcomes[0].approved is True
    call_kwargs = mocks["execution_client"].submit_limit_buy.call_args.kwargs
    # 200.10 + 2.0 * 4.0 = 208.10
    assert call_kwargs["take_profit_price"] == pytest.approx(208.10)


def test_orchestrator_sync_account_compounds_equity_and_positions() -> None:
    from src.execution import AccountSnapshot
    from src.guardian import PortfolioPosition

    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(":memory:")
    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL", price=200.0, ask_price=200.0, atr_14=2.0),
        news_items=[_make_news(now_ny - timedelta(minutes=5), source="Reuters")],
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )

    # Compounded account grew from $10,000 -> $25,000; open position in MSFT shouldn't block AAPL
    mocks["execution_client"].get_account_snapshot.return_value = AccountSnapshot(
        equity=25_000.0,
        cash=20_000.0,
        buying_power=45_000.0,
        last_equity=24_000.0,
        daily_pnl=1_000.0,
    )
    mocks["execution_client"].get_open_positions.return_value = [
        PortfolioPosition(symbol="MSFT", sector="Technology", market_value=5_000.0, shares=12, avg_price=415.0)
    ]

    report = orchestrator.run_cycle(
        symbols=["AAPL"],
        now=now_ny,
        dry_run=False,
        starting_equity=10_000.0,
        sync_account=True,
    )
    outcome = report.ticker_outcomes[0]
    assert outcome.approved is True
    assert outcome.guardian_decision is not None
    # Compounded capital = 24,000 + 1,000 = 25,000; 35% Technology cap ($8,750) minus synced MSFT ($5,000) = $3,750 -> 18 shares
    # (Whereas unsynced starting_equity=10,000 would cap at 20% = $2,000 -> 10 shares.)
    assert outcome.guardian_decision.recommended_shares == 18

    # If synced position already holds AAPL, RiskGuardian vetoes duplicate entry
    mocks["execution_client"].get_open_positions.return_value = [
        PortfolioPosition(symbol="AAPL", sector="Technology", market_value=4_000.0, shares=20, avg_price=200.0)
    ]
    dup_report = orchestrator.run_cycle(
        symbols=["AAPL"],
        now=now_ny,
        dry_run=False,
        sync_account=True,
    )
    assert dup_report.ticker_outcomes[0].approved is False
    assert dup_report.ticker_outcomes[0].status == "vetoed_guardian"


def test_orchestrator_limits_ai_calls_to_max_candidates_for_ai() -> None:
    now_ny = datetime(2026, 11, 18, 10, 30, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(":memory:")

    screening_map = {
        "LOW": UniverseScreeningResult(
            symbol="LOW",
            passed=True,
            indicators=TechnicalIndicators(
                symbol="LOW",
                current_price=95.0,
                sma_200=80.0,
                atr_14=2.0,
                above_sma_200=True,
                vix=18.0,
                latest_volume=900_000.0,
                avg_volume_20d=1_000_000.0,
                ema_20=100.0,
                rsi_14=75.0,
            ),
            quote=MarketQuote("LOW", 95.0, 94.9, 95.1, 0.002, 1_000_000.0),
            fundamentals=FundamentalData("LOW", 1e9, 0.5),
        ),
        "TOP1": UniverseScreeningResult(
            symbol="TOP1",
            passed=True,
            indicators=TechnicalIndicators(
                symbol="TOP1",
                current_price=150.0,
                sma_200=120.0,
                atr_14=2.5,
                above_sma_200=True,
                vix=18.0,
                latest_volume=2_400_000.0,
                avg_volume_20d=1_000_000.0,
                ema_20=140.0,
                rsi_14=56.0,
            ),
            quote=MarketQuote("TOP1", 150.0, 149.9, 150.1, 0.001, 2_400_000.0),
            fundamentals=FundamentalData("TOP1", 2e9, 0.4),
        ),
        "TOP2": UniverseScreeningResult(
            symbol="TOP2",
            passed=True,
            indicators=TechnicalIndicators(
                symbol="TOP2",
                current_price=210.0,
                sma_200=180.0,
                atr_14=3.0,
                above_sma_200=True,
                vix=18.0,
                latest_volume=1_600_000.0,
                avg_volume_20d=1_000_000.0,
                ema_20=200.0,
                rsi_14=54.0,
            ),
            quote=MarketQuote("TOP2", 210.0, 209.9, 210.1, 0.001, 1_600_000.0),
            fundamentals=FundamentalData("TOP2", 2e9, 0.4),
        ),
        "FAIL": _make_screening_result("FAIL", passed=False),
    }

    orchestrator, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=screening_map["TOP1"],
        news_items=[_make_news(now_ny - timedelta(minutes=5))],
        analyst_result=_make_analyst_result("TOP1", approved=True),
    )
    orchestrator.max_candidates_for_ai = 2
    mocks["screener"].screen_symbol.side_effect = lambda sym, **kw: screening_map[sym]
    mocks["analyst"].evaluate.side_effect = lambda **kw: _make_analyst_result(kw["symbol"], approved=True)

    report = orchestrator.run_cycle(
        symbols=["LOW", "TOP1", "FAIL", "TOP2"],
        now=now_ny,
        dry_run=True,
    )

    assert [o.symbol for o in report.ticker_outcomes] == ["LOW", "TOP1", "FAIL", "TOP2"]
    assert report.ticker_outcomes[0].status == "filtered_max_candidates"
    assert report.ticker_outcomes[2].status == "rejected_universe"
    assert report.ticker_outcomes[1].approved is True
    assert report.ticker_outcomes[3].approved is True

    fetched_symbols = [call.args[0] for call in mocks["news_aggregator"].fetch_news.call_args_list]
    evaluated_symbols = [call.kwargs["symbol"] for call in mocks["analyst"].evaluate.call_args_list]
    assert fetched_symbols == ["TOP1", "TOP2"]
    assert evaluated_symbols == ["TOP1", "TOP2"]


def test_cli_backtest_flag_and_new_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json
    import sys
    from types import ModuleType, SimpleNamespace
    import pandas as pd

    cfg_path = tmp_path / "strategy.json"
    cfg_path.write_text(
        json.dumps(
            {
                "tickers": ["AAPL", "MSFT"],
                "atr_stop_multiplier": 2.0,
                "atr_take_profit_multiplier": 2.0,
                "max_hold_bars": 10,
                "rsi_entry_max": 60.0,
                "min_relative_volume": 1.2,
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--tickers",
            "AAPL",
            "NVDA",
            "--config",
            str(cfg_path),
            "--backtest",
            "--backtest-period",
            "6mo",
            "--initial-capital",
            "25000",
            "--max-candidates",
            "3",
            "--sync-account",
        ]
    )
    assert args.backtest is True
    assert args.backtest_period == "6mo"
    assert args.initial_capital == 25000.0
    assert args.max_candidates == 3
    assert args.sync_account is True

    fake_backtest = ModuleType("src.backtest")
    mock_store = MagicMock()
    mock_store.get_or_fetch_candles.return_value = pd.DataFrame({"Close": [100.0, 105.0]})

    mock_engine = MagicMock()
    mock_engine.run.return_value = SimpleNamespace(
        metrics=SimpleNamespace(
            total_trades=12,
            win_rate=0.625,
            profit_factor=2.15,
            max_drawdown_pct=0.045,
            total_return_pct=0.184,
            initial_capital=25000.0,
            final_equity=29600.0,
            compounded_pnl=4600.0,
        ),
        vetoed_entries_count=3,
    )

    fake_backtest.BacktestConfig = MagicMock(return_value="cfg")
    fake_backtest.ParquetCandleStore = MagicMock(return_value=mock_store)
    fake_backtest.BacktestEngine = MagicMock(return_value=mock_engine)
    monkeypatch.setitem(sys.modules, "src.backtest", fake_backtest)

    code = main(
        [
            "--tickers",
            "aapl",
            "msft",
            "--config",
            str(cfg_path),
            "--backtest",
            "--backtest-period",
            "1y",
            "--initial-capital",
            "25000",
        ]
    )
    assert code == 0
    mock_store.get_or_fetch_candles.assert_any_call("AAPL", period="1y")
    mock_store.get_or_fetch_candles.assert_any_call("MSFT", period="1y")
    fake_backtest.BacktestConfig.assert_called_once_with(
        initial_capital=25000.0,
        atr_stop_multiplier=2.0,
        atr_take_profit_multiplier=2.0,
        max_hold_bars=10,
        rsi_entry_max=60.0,
        min_relative_volume=1.2,
    )
    mock_engine.run.assert_called_once()

    captured = capsys.readouterr().out
    assert "Total Trades" in captured
    assert "Win Rate" in captured
    assert "Profit Factor" in captured
    assert "Max Drawdown" in captured
    assert "Total Return" in captured
    assert "Final Equity" in captured


def test_process_lock_prevents_concurrent_instances_and_loop_backoff(tmp_path: Path) -> None:
    from main import ProcessLock, ProcessLockError

    lock_file = tmp_path / "bot.lock"
    with ProcessLock(lock_file):
        with pytest.raises(ProcessLockError, match="already running"):
            with ProcessLock(lock_file):
                pass

    # After release, lock can be acquired cleanly again
    with ProcessLock(lock_file):
        assert lock_file.exists()

    # Loop recovers from transient exception with backoff and Telegram alert
    mock_orch = MagicMock(spec=TradingBotOrchestrator)
    mock_orch.notifier = MagicMock()
    mock_orch.run_cycle.side_effect = [
        RuntimeError("transient network drop"),
        PipelineCycleReport(
            timestamp=datetime(2026, 11, 18, 12, 0, tzinfo=NY_TIMEZONE),
            market_open=True,
            skipped=False,
            dry_run=True,
        ),
    ]
    recorded_sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded_sleeps.append(seconds)
        if len(recorded_sleeps) >= 2:
            raise KeyboardInterrupt

    exit_code = main(
        argv=["--tickers", "AAPL", "--loop", "--interval-seconds", "10", "--ignore-market-hours"],
        orchestrator=mock_orch,
        sleep_fn=fake_sleep,
    )
    assert exit_code == 0
    assert recorded_sleeps == [10.0, 10.0]
    mock_orch.notifier.send_message.assert_called_once()


def test_orchestrator_auto_resolves_correlation_and_fomc_blackout(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd
    from src.backtest import ParquetCandleStore
    from src.guardian import PortfolioPosition

    now_ny = datetime(2026, 12, 9, 11, 0, 0, tzinfo=NY_TIMEZONE)
    confirmed_news = [_make_news(now_ny - timedelta(minutes=5), source="Reuters")]

    # 1) FOMC decision at 14:00 ET same day vetoes even when Gemini approves
    store_fed = TradingAuditStore(tmp_path / "fed.db")
    orch_fed, mocks_fed = _build_mock_orchestrator(
        audit_store=store_fed,
        screening_result=_make_screening_result("AAPL", vix=17.0),
        news_items=confirmed_news,
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )
    orch_fed.default_fed_dates = (datetime(2026, 12, 9, 14, 0, 0, tzinfo=NY_TIMEZONE),)
    fed_report = orch_fed.run_cycle(symbols=["AAPL"], now=now_ny, dry_run=False)
    assert fed_report.ticker_outcomes[0].approved is False
    assert fed_report.ticker_outcomes[0].status == "vetoed_guardian"
    assert any("Fed" in r for r in fed_report.ticker_outcomes[0].reasons)
    mocks_fed["execution_client"].submit_limit_buy.assert_not_called()

    # 2) Cached Parquet candles drive automatic correlation_matrix (> 0.75 veto)
    candle_store = ParquetCandleStore(cache_dir=tmp_path / "candles")
    dates = pd.date_range("2026-08-01", periods=45, freq="B")
    rets = np.random.default_rng(42).normal(0.001, 0.015, size=45)
    candle_store.save_candles("AAPL", pd.DataFrame({"Close": 100.0 * np.cumprod(1.0 + rets)}, index=dates))
    candle_store.save_candles("QQQ", pd.DataFrame({"Close": 200.0 * np.cumprod(1.0 + rets * 0.99)}, index=dates))

    now_corr = datetime(2026, 11, 18, 11, 0, 0, tzinfo=NY_TIMEZONE)
    store_corr = TradingAuditStore(tmp_path / "corr.db")
    orch_corr, mocks_corr = _build_mock_orchestrator(
        audit_store=store_corr,
        screening_result=_make_screening_result("AAPL", vix=17.0),
        news_items=[_make_news(now_corr - timedelta(minutes=5), source="Reuters")],
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )
    orch_corr._candle_store = candle_store

    corr_report = orch_corr.run_cycle(
        symbols=["AAPL"],
        now=now_corr,
        portfolio_positions=[
            PortfolioPosition(symbol="QQQ", sector="Broad Market", market_value=10_000.0, shares=20, avg_price=500.0)
        ],
    )
    assert corr_report.ticker_outcomes[0].approved is False
    assert corr_report.ticker_outcomes[0].status == "vetoed_guardian"
    assert any("Correlation" in r and "QQQ" in r for r in corr_report.ticker_outcomes[0].reasons)
    mocks_corr["execution_client"].submit_limit_buy.assert_not_called()


def test_orchestrator_skips_gemini_call_when_deterministic_prefilter_fails(tmp_path: Path) -> None:
    now_ny = datetime(2026, 11, 19, 11, 30, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "prefilter.db")
    screening = _make_screening_result("XOM", vix=18.0)
    # 2 sources (< 3) and relative volume 0.84 (< 1.20)
    news_items = [
        _make_news(now_ny - timedelta(minutes=10), source="Reuters"),
        _make_news(now_ny - timedelta(minutes=20), source="Bloomberg"),
    ]
    orch, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=screening,
        news_items=news_items,
        analyst_result=_make_analyst_result("XOM", approved=True),
    )
    orch.skip_ai_on_prefilter_fail = True

    report = orch.run_cycle(
        symbols=["XOM"],
        now=now_ny,
        dry_run=True,
        current_volumes={"XOM": 840_000.0},
        avg_volumes_20d={"XOM": 1_000_000.0},
    )

    assert len(report.ticker_outcomes) == 1
    outcome = report.ticker_outcomes[0]
    assert outcome.approved is False
    assert outcome.stage_reached == "analyst_prefilter"
    assert outcome.status == "rejected_analyst_prefilter"
    assert any("relative_volume 0.84" in r for r in outcome.reasons)
    assert any("independent_sources_count 2 < 3" in r for r in outcome.reasons)
    mocks["analyst"].evaluate.assert_not_called()
    mocks["execution_client"].submit_limit_buy.assert_not_called()


def test_orchestrator_prefilter_preserves_top_candidate_slots_for_volume_confirmed_tickers(
    tmp_path: Path,
) -> None:
    now_ny = datetime(2026, 11, 19, 11, 30, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "slots.db")
    screening_map = {
        "LOWVOL": _make_screening_result("LOWVOL", vix=18.0),
        "HIGHVOL": _make_screening_result("HIGHVOL", vix=18.0),
    }
    news_items = [
        _make_news(now_ny - timedelta(minutes=10), source="Reuters"),
        _make_news(now_ny - timedelta(minutes=15), source="Bloomberg"),
        _make_news(now_ny - timedelta(minutes=20), source="Benzinga"),
    ]
    orch, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=screening_map["LOWVOL"],
        news_items=news_items,
        analyst_result=_make_analyst_result("HIGHVOL", approved=True),
    )
    orch.max_candidates_for_ai = 1
    orch.skip_ai_on_prefilter_fail = True
    mocks["screener"].screen_symbol.side_effect = lambda sym, **kw: screening_map[sym]
    mocks["analyst"].evaluate.side_effect = lambda **kw: _make_analyst_result(
        kw["symbol"], approved=True
    )

    report = orch.run_cycle(
        symbols=["LOWVOL", "HIGHVOL"],
        now=now_ny,
        dry_run=True,
        current_volumes={"LOWVOL": 800_000.0, "HIGHVOL": 1_500_000.0},
        avg_volumes_20d={"LOWVOL": 1_000_000.0, "HIGHVOL": 1_000_000.0},
    )

    by_sym = {o.symbol: o for o in report.ticker_outcomes}
    assert by_sym["LOWVOL"].status == "rejected_analyst_prefilter"
    assert by_sym["HIGHVOL"].approved is True
    assert [c.kwargs["symbol"] for c in mocks["analyst"].evaluate.call_args_list] == ["HIGHVOL"]


def test_orchestrator_skips_ai_for_held_symbols_and_earnings_blackout_and_fast_sigterm(
    tmp_path: Path,
) -> None:
    import os
    import signal
    import threading
    import time

    from main import _run_loop, parse_args
    from src.guardian import PortfolioPosition

    now_ny = datetime(2026, 11, 19, 12, 0, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "skip_held_cal.db")

    held_screening = _make_screening_result("DIS", vix=20.0)
    earnings_screening = UniverseScreeningResult(
        symbol="WMT",
        passed=True,
        quote=MarketQuote("WMT", 100.0, 99.9, 100.1, 0.002, 1_500_000.0),
        indicators=TechnicalIndicators(
            symbol="WMT",
            current_price=100.0,
            sma_200=90.0,
            atr_14=2.0,
            above_sma_200=True,
            vix=20.0,
            latest_volume=2_000_000.0,
            avg_volume_20d=1_000_000.0,
            ema_20=98.0,
            rsi_14=50.0,
        ),
        fundamentals=FundamentalData(
            symbol="WMT",
            free_cash_flow=5e9,
            debt_to_equity=0.5,
            earnings_dates=["2026-11-19"],
        ),
    )
    fresh_screening = _make_screening_result("KO", vix=20.0)
    smap = {"DIS": held_screening, "WMT": earnings_screening, "KO": fresh_screening}
    news_items = [
        _make_news(now_ny - timedelta(minutes=10), source="Reuters"),
        _make_news(now_ny - timedelta(minutes=15), source="Bloomberg"),
        _make_news(now_ny - timedelta(minutes=20), source="Benzinga"),
    ]
    orch, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=fresh_screening,
        news_items=news_items,
        analyst_result=_make_analyst_result("KO", approved=True),
    )
    orch.skip_ai_on_prefilter_fail = True
    mocks["screener"].screen_symbol.side_effect = lambda sym, **kw: smap[sym]

    report = orch.run_cycle(
        symbols=["DIS", "WMT", "KO"],
        now=now_ny,
        dry_run=True,
        portfolio_positions=[
            PortfolioPosition(
                symbol="DIS",
                sector="Communication Services",
                market_value=10_000.0,
                shares=100.0,
                avg_price=100.0,
            )
        ],
        current_volumes={"DIS": 2e6, "WMT": 2e6, "KO": 2e6},
        avg_volumes_20d={"DIS": 1e6, "WMT": 1e6, "KO": 1e6},
    )

    by_sym = {o.symbol: o for o in report.ticker_outcomes}
    assert by_sym["DIS"].status == "skipped_already_held"
    assert by_sym["WMT"].status == "skipped_calendar_blackout"
    assert by_sym["KO"].approved is True
    assert [c.kwargs["symbol"] for c in mocks["analyst"].evaluate.call_args_list] == ["KO"]

    # Verify _run_loop wakes up immediately (< 1.5s) on SIGTERM even with interval_seconds=60
    fake_bot = MagicMock()
    fake_bot.notifier = MagicMock()
    fake_bot.run_cycle.return_value = PipelineCycleReport(
        timestamp=now_ny,
        market_open=True,
        dry_run=True,
    )
    args = parse_args(["--loop", "--interval-seconds", "60", "--dry-run"])

    def _send_sigterm() -> None:
        time.sleep(0.15)
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=_send_sigterm, daemon=True)
    t0 = time.monotonic()
    t.start()
    rc = _run_loop(fake_bot, ["KO"], args, sleep_fn=time.sleep)
    elapsed = time.monotonic() - t0
    t.join(timeout=2.0)

    assert rc == 0
    assert elapsed < 2.0
    store.close()


def test_orchestrator_cash_only_guard_prevents_margin_and_skips_scanning(tmp_path: Path) -> None:
    from src.execution.models import AccountSnapshot

    now_ny = datetime(2026, 11, 20, 10, 15, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "cash_guard.db")
    orch, mocks = _build_mock_orchestrator(
        audit_store=store,
        screening_result=_make_screening_result("AAPL", vix=18.0),
        news_items=[_make_news(now_ny - timedelta(minutes=10), source="Reuters")],
        analyst_result=_make_analyst_result("AAPL", approved=True),
    )
    # Simulate account with negative/exhausted cash (margin available, cash < $100)
    mocks["execution_client"].get_account_snapshot.return_value = AccountSnapshot(
        equity=100_000.0,
        cash=-5_000.0,
        buying_power=50_000.0,
        last_equity=100_000.0,
        daily_pnl=0.0,
    )
    mocks["execution_client"].get_open_positions.return_value = []
    mocks["execution_client"].reconcile_managed_positions.return_value = []

    report = orch.run_cycle(
        symbols=["AAPL", "MSFT"],
        now=now_ny,
        dry_run=False,
        sync_account=True,
    )

    assert len(report.ticker_outcomes) == 2
    assert all(o.status == "skipped_insufficient_cash" for o in report.ticker_outcomes)
    mocks["screener"].screen_symbol.assert_not_called()
    mocks["news_aggregator"].fetch_news.assert_not_called()
    mocks["analyst"].evaluate.assert_not_called()
    mocks["execution_client"].submit_limit_buy.assert_not_called()
    store.close()
