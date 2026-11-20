"""Phase 3 chaos and interruption-resilience tests for already-held positions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from main import ProcessLock, ProcessLockError, TradingBotOrchestrator
from src.execution import AlpacaExecutionClient, ExecutionConfig, ManagedPosition, PositionManager
from src.execution.models import NY_TIMEZONE
from src.storage import TradingAuditStore
from src.universe.models import MarketQuote


def test_cold_restart_with_wiped_db_does_not_false_trigger_flash_crash(tmp_path: Path) -> None:
    """Recovering an underwater swing position on a fresh DB must NOT panic-sell as flash crash."""
    now_ny = datetime(2026, 11, 20, 10, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "fresh_vps.db")
    mock_broker = MagicMock()
    # Pre-existing underwater position (-3.5% vs entry), no local SQLite history, no resting stop
    mock_broker.get_all_positions.return_value = [
        SimpleNamespace(symbol="AMD", qty="11", avg_entry_price="482.16", market_value="5117.75"),
    ]
    mock_broker.get_orders.return_value = []
    mock_broker.submit_order.return_value = SimpleNamespace(
        id="recovered-gtc-stop-1", status="accepted"
    )

    client = AlpacaExecutionClient(
        audit_store=store,
        trading_client=mock_broker,
        config=ExecutionConfig(trailing_atr_multiplier=2.5),
        paper=True,
    )
    reconciled = client.reconcile_managed_positions(now=now_ny, ensure_protective_stops=True)
    assert len(reconciled) == 1
    pos = reconciled[0]
    assert pos.stop_order_id == "recovered-gtc-stop-1"
    assert (now_ny - pos.opened_at) >= timedelta(minutes=15)

    # Evaluate at -3.5% below entry (above stop_price): must HOLD, NOT EMERGENCY_EXIT
    pm = PositionManager(execution_client=client, audit_store=store)
    res = pm.evaluate_and_manage(
        pos,
        current_price=465.25,
        bid_price=465.00,
        now=now_ny,
        dry_run=False,
    )
    assert res.action == "HOLD"
    assert res.should_exit is False
    store.close()


def test_manage_open_positions_isolates_single_ticker_network_failure(tmp_path: Path) -> None:
    """A transient quote/API error on one held position must not block managing remaining positions."""
    now_ny = datetime(2026, 11, 20, 11, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "isolation.db")

    mock_screener = MagicMock()
    def _quote_side_effect(sym: str) -> MarketQuote:
        if sym == "AAPL":
            raise ConnectionError("Simulated Alpaca network drop on AAPL")
        if sym == "KO":
            return MarketQuote("KO", 102.0, 101.95, 102.05, 0.001, 2_000_000.0)
        return MarketQuote("DIS", 110.0, 109.90, 110.10, 0.001, 2_000_000.0)
    mock_screener.market_data.get_quote.side_effect = _quote_side_effect

    mock_exec = MagicMock()
    mock_exec.audit_store = store
    mock_exec.config = ExecutionConfig()
    mock_exec.submit_limit_sell.return_value = SimpleNamespace(
        submitted=True,
        client_order_id="sell-dis-1",
        broker_order_id="brk-dis-1",
        symbol="DIS",
        side="SELL",
        qty=50.0,
        limit_price=109.57,
        stop_price=None,
        order_type="limit",
        status="accepted",
        reason="Trailing stop triggered",
    )
    pm = PositionManager(execution_client=mock_exec, audit_store=store)
    mock_notifier = MagicMock()

    orch = TradingBotOrchestrator(
        screener=mock_screener,
        execution_client=mock_exec,
        position_manager=pm,
        audit_store=store,
        notifier=mock_notifier,
    )

    positions = [
        ManagedPosition("AAPL", 10, 350.0, 335.0, 5.0, now_ny - timedelta(hours=3)),
        ManagedPosition("KO", 100, 95.0, 92.0, 2.0, now_ny - timedelta(hours=3), stop_order_id="stop-ko"),
        ManagedPosition("DIS", 50, 120.0, 115.0, 2.5, now_ny - timedelta(hours=3), stop_order_id="stop-dis"),
    ]

    results = orch._manage_open_positions(
        open_positions=positions,
        now_ny=now_ny,
        dry_run=False,
        current_prices=None,
        bid_prices=None,
    )

    by_sym = {r.symbol: r for r in results}
    assert "AAPL" not in by_sym  # Failed position logged & alerted without aborting loop
    assert by_sym["KO"].action == "TRAIL_UPDATE"
    assert by_sym["KO"].new_stop_price == pytest.approx(98.0)
    assert by_sym["DIS"].action == "TRAILING_STOP_EXIT"
    assert by_sym["DIS"].should_exit is True
    mock_exec.update_stop_order.assert_called_once()
    mock_exec.submit_limit_sell.assert_called_once()
    mock_notifier.send_message.assert_called_once()
    mock_notifier.notify_position_closed.assert_called_once()
    store.close()


def test_stale_stop_order_failure_still_persists_local_trailing_stop_and_clears_stale_id(
    tmp_path: Path,
) -> None:
    """If broker rejects stop replacement with 404/canceled, keep local stop and self-heal ID."""
    now_ny = datetime(2026, 11, 20, 11, 30, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "stale_stop.db")
    mock_exec = MagicMock()
    mock_exec.audit_store = store
    mock_exec.config = ExecutionConfig()
    mock_exec.update_stop_order.side_effect = RuntimeError("404 order not found / already canceled")

    pm = PositionManager(execution_client=mock_exec, audit_store=store)
    pos = ManagedPosition(
        symbol="JNJ",
        qty=73,
        entry_price=266.74,
        current_stop_price=254.54,
        atr_14=5.0,
        opened_at=now_ny - timedelta(hours=20),
        highest_price=266.74,
        stop_order_id="stale-stop-id-999",
    )

    res = pm.evaluate_and_manage(
        pos,
        current_price=275.0,
        bid_price=274.9,
        now=now_ny,
        dry_run=False,
    )
    assert res.action == "TRAIL_UPDATE"
    assert res.new_stop_price == pytest.approx(265.0)
    assert pos.stop_order_id is None  # Cleared so next reconcile auto-attaches fresh GTC stop
    persisted = store.get_position_state("JNJ")
    assert persisted is not None
    assert persisted.current_stop_price == pytest.approx(265.0)
    assert persisted.highest_price == pytest.approx(275.0)
    assert persisted.stop_order_id is None
    store.close()


def test_reconcile_cleans_closed_positions_and_updates_partial_fills_plus_process_lock(
    tmp_path: Path,
) -> None:
    """Verify broker-closed positions are purged, partial fills shrink qty, and ProcessLock blocks races."""
    now_ny = datetime(2026, 11, 20, 12, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "reconcile_sync.db")
    store.upsert_position_state(
        symbol="KO", qty=208.0, entry_price=95.92, current_stop_price=92.77,
        atr_14=1.8, opened_at=now_ny - timedelta(days=1), highest_price=97.25,
        stop_order_id="stop-ko", updated_at=now_ny - timedelta(hours=1),
    )
    store.upsert_position_state(
        symbol="V", qty=49.0, entry_price=403.80, current_stop_price=385.85,
        atr_14=9.0, opened_at=now_ny - timedelta(days=1), highest_price=407.02,
        stop_order_id="stop-v", updated_at=now_ny - timedelta(hours=1),
    )

    mock_broker = MagicMock()
    # KO closed overnight (take-profit filled); V partially reduced from 49 -> 25 shares
    mock_broker.get_all_positions.return_value = [
        SimpleNamespace(symbol="V", qty="25", avg_entry_price="403.80", market_value="10150.00"),
    ]
    mock_broker.get_orders.return_value = [
        SimpleNamespace(id="stop-v", symbol="V", side="sell", order_type="stop_limit", stop_price="385.85"),
    ]

    client = AlpacaExecutionClient(audit_store=store, trading_client=mock_broker, paper=True)
    reconciled = client.reconcile_managed_positions(now=now_ny, ensure_protective_stops=True)
    assert [p.symbol for p in reconciled] == ["V"]
    assert reconciled[0].qty == pytest.approx(25.0)
    assert store.get_position_state("KO") is None
    v_state = store.get_position_state("V")
    assert v_state is not None and v_state.qty == pytest.approx(25.0)

    # Verify duplicate restart race is blocked by ProcessLock
    lock_path = tmp_path / "bot.lock"
    with ProcessLock(lock_path):
        with pytest.raises(ProcessLockError):
            with ProcessLock(lock_path):
                pass
    store.close()


def test_email_notifier_html_good_news_protective_exit_and_veto_dedup() -> None:
    from src.notifications import CompositeNotifier, EmailNotifier

    sent_messages: list[tuple[str, list[str], str]] = []

    class _FakeSmtp:
        def __init__(self, host: str, port: int, timeout: float = 15.0) -> None:
            self.host = host
            self.port = port

        def __enter__(self) -> _FakeSmtp:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def login(self, user: str, password: str) -> None:
            assert user == "bot@example.com"
            assert password == "abcd-efgh-ijkl-mnop"

        def sendmail(self, from_addr: str, to_addrs: list[str], raw_msg: str) -> None:
            sent_messages.append((from_addr, to_addrs, raw_msg))

    notifier = EmailNotifier(
        smtp_user="bot@example.com",
        smtp_app_password="abcd-efgh-ijkl-mnop",
        recipient_email="user@example.com",
        smtp_factory=_FakeSmtp,
    )
    composite = CompositeNotifier(email_notifier=notifier)

    # 1) Buy -> Good News green card
    import email as email_pkg

    def _decode_mime(raw: str) -> str:
        parsed = email_pkg.message_from_string(raw)
        return "\n".join(
            part.get_payload(decode=True).decode("utf-8")
            for part in parsed.walk()
            if part.get_payload(decode=True) is not None
        )

    assert composite.notify_trade_executed(
        symbol="JNJ", side="BUY", qty=73, limit_price=266.74, stop_price=253.13
    ) is True
    assert len(sent_messages) == 1
    assert "16a34a" in _decode_mime(sent_messages[-1][2])  # Green good-news header

    # 2) Profitable exit -> Good News realized profit card (+$402.23 / +2.07%)
    assert composite.notify_position_closed(
        symbol="JNJ",
        action="TRAILING_STOP_EXIT",
        qty=73,
        limit_price=272.25,
        reason="Trailing stop locked profit",
        entry_price=266.74,
    ) is True
    assert len(sent_messages) == 2
    assert "402.23" in _decode_mime(sent_messages[-1][2])

    # 3) Controlled loss exit -> Yellow protective card
    assert composite.notify_position_closed(
        symbol="DIS",
        action="TIME_EXIT",
        qty=100,
        limit_price=119.00,
        reason="Time window expired",
        entry_price=121.29,
    ) is True
    assert len(sent_messages) == 3
    assert "ca8a04" in _decode_mime(sent_messages[-1][2])

    # 4) Guardian veto deduplicates repeated same-day vetoes per symbol
    assert composite.notify_guardian_veto("WMT", ["Earnings in 0.00h"], vix=24.5) is True
    assert composite.notify_guardian_veto("WMT", ["Earnings in 0.00h"], vix=24.5) is False
    assert len(sent_messages) == 4


def test_position_manager_software_take_profit_exit(tmp_path: Path) -> None:
    now_ny = datetime(2026, 11, 20, 14, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "tp_exit.db")
    mock_exec = MagicMock()
    mock_exec.audit_store = store
    mock_exec.config = ExecutionConfig(
        trailing_atr_multiplier=2.5,
        take_profit_atr_multiplier=5.0,
        enable_take_profit_exit=True,
    )
    mock_exec.submit_limit_sell.return_value = SimpleNamespace(
        submitted=True,
        client_order_id="tp-ko-1",
        broker_order_id="brk-tp-ko-1",
        symbol="KO",
        side="SELL",
        qty=200.0,
        limit_price=104.68,
        stop_price=None,
        order_type="limit",
        status="accepted",
        reason="Take-profit target reached",
    )

    pm = PositionManager(
        execution_client=mock_exec,
        audit_store=store,
        config=mock_exec.config,
    )
    pos = ManagedPosition(
        symbol="KO",
        qty=200.0,
        entry_price=95.00,
        current_stop_price=92.00,
        atr_14=2.00,
        opened_at=now_ny - timedelta(hours=24),
        stop_order_id="gtc-stop-ko",
    )
    # Target = 95.00 + 5.0 * 2.00 = 105.00
    res = pm.evaluate_and_manage(
        pos,
        current_price=105.20,
        bid_price=105.00,
        now=now_ny,
        dry_run=False,
    )
    assert res.action == "TAKE_PROFIT_EXIT"
    assert res.should_exit is True
    mock_exec.submit_limit_sell.assert_called_once()
    store.close()


def test_position_manager_peak_exhaustion_and_breakeven_lock(tmp_path: Path) -> None:
    now_ny = datetime(2026, 11, 20, 14, 30, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "peak_exit.db")
    mock_exec = MagicMock()
    mock_exec.audit_store = store
    mock_exec.config = ExecutionConfig(
        trailing_atr_multiplier=2.5,
        take_profit_atr_multiplier=5.0,
        enable_take_profit_exit=False,
        breakeven_trigger_pct=0.05,
        peak_activation_pct=0.05,
        peak_pullback_pct=0.012,
    )
    mock_exec.submit_limit_sell.return_value = SimpleNamespace(
        submitted=True,
        client_order_id="peak-v-1",
        broker_order_id="brk-peak-v-1",
        symbol="V",
        side="SELL",
        qty=50.0,
        limit_price=324.50,
        stop_price=None,
        order_type="limit",
        status="accepted",
        reason="Peak exhaustion detected",
    )

    pm = PositionManager(
        execution_client=mock_exec,
        audit_store=store,
        config=mock_exec.config,
    )
    pos = ManagedPosition(
        symbol="V",
        qty=50.0,
        entry_price=310.00,
        current_stop_price=292.00,
        atr_14=15.00,
        opened_at=now_ny - timedelta(hours=48),
        highest_price=310.00,
        stop_order_id="gtc-stop-v",
    )

    # 1) Surge to +5.5% ($327.05) with wide ATR ($15 * 2.5 = $37.50) -> breakeven lock raises stop to $310.62
    res_ratchet = pm.evaluate_and_manage(
        pos,
        current_price=327.05,
        bid_price=327.00,
        now=now_ny,
        dry_run=False,
    )
    assert res_ratchet.action == "TRAIL_UPDATE"
    assert res_ratchet.new_stop_price == round(310.00 * 1.002, 2)
    assert pos.highest_price == 327.05

    # 2) Pull back 1.4% from peak ($327.05 -> $322.40) while still +4.0% above entry -> PEAK_EXHAUSTION_EXIT
    res_peak = pm.evaluate_and_manage(
        pos,
        current_price=322.40,
        bid_price=322.30,
        now=now_ny + timedelta(minutes=5),
        dry_run=False,
    )
    assert res_peak.action == "PEAK_EXHAUSTION_EXIT"
    assert res_peak.should_exit is True
    mock_exec.submit_limit_sell.assert_called_once()
    store.close()


def test_broker_closed_position_reconciliation_and_daily_summary_email(
    tmp_path: Path,
) -> None:
    now_ny = datetime(2026, 11, 20, 15, 0, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "broker_close.db")
    store.upsert_position_state(
        symbol="JNJ",
        qty=73.0,
        entry_price=266.74,
        current_stop_price=266.96,
        atr_14=3.34,
        opened_at=now_ny - timedelta(hours=20),
        highest_price=275.31,
        stop_order_id="stop-jnj-1",
    )

    mock_trading_client = MagicMock()
    mock_trading_client.get_all_positions.return_value = []  # Closed at broker
    mock_trading_client.get_orders.side_effect = [
        [],  # status=OPEN
        [
            SimpleNamespace(
                symbol="JNJ",
                side="sell",
                status="filled",
                type="stop_limit",
                filled_avg_price="267.10",
            )
        ],  # status=ALL
    ]

    exec_client = AlpacaExecutionClient(
        audit_store=store,
        trading_client=mock_trading_client,
    )
    reconciled = exec_client.reconcile_managed_positions(now=now_ny)
    assert reconciled == []
    assert len(exec_client.last_broker_closed_positions) == 1
    closed_info = exec_client.last_broker_closed_positions[0]
    assert closed_info["symbol"] == "JNJ"
    assert closed_info["exit_price"] == 267.10
    assert store.get_open_position_states() == []

    # Verify daily summary HTML email generation
    from src.notifications import EmailNotifier

    sent_messages: list[tuple[str, str, str]] = []

    class _FakeSMTP:
        def __init__(self, host: str, port: int, timeout: float = 15.0) -> None:
            pass

        def __enter__(self) -> _FakeSMTP:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def starttls(self) -> None:
            pass

        def login(self, user: str, pwd: str) -> None:
            pass

        def sendmail(self, sender: str, recipients: list[str], msg_str: str) -> None:
            sent_messages.append((sender, recipients[0], msg_str))

    email_notifier = EmailNotifier(
        smtp_user="sergio5453@gmail.com",
        smtp_app_password="app-password",
        recipient_email="serggi.art@gmail.com",
        smtp_factory=_FakeSMTP,  # type: ignore[arg-type]
    )
    assert email_notifier.notify_daily_summary(
        equity=100_987.91,
        daily_pnl=985.59,
        cash=30_000.00,
        open_positions=[SimpleNamespace(symbol="JNJ", shares=73)],
    ) is True
    assert len(sent_messages) == 1
    store.close()


def test_cli_close_symbols_submits_limit_sell_and_clears_state(tmp_path: Path) -> None:
    from main import main

    now_ny = datetime(2026, 11, 20, 15, 30, tzinfo=NY_TIMEZONE)
    store = TradingAuditStore(tmp_path / "cli_close.db")
    store.upsert_position_state(
        symbol="AAPL",
        qty=55.0,
        entry_price=296.45,
        current_stop_price=286.63,
        atr_14=5.61,
        opened_at=now_ny - timedelta(hours=5),
    )

    mock_bot = MagicMock()
    mock_bot.audit_store = store
    mock_bot.execution_client.get_open_positions.return_value = [
        SimpleNamespace(
            symbol="AAPL",
            shares=55.0,
            market_value=16390.0,
            avg_price=296.45,
        )
    ]
    mock_bot.screener.market_data.get_quote.return_value = SimpleNamespace(
        price=298.10,
        bid_price=298.00,
    )
    mock_bot.execution_client.submit_limit_sell.return_value = SimpleNamespace(
        limit_price=297.11,
    )

    rc = main(["--close-symbols", "AAPL"], orchestrator=mock_bot)
    assert rc == 0
    mock_bot.execution_client.submit_limit_sell.assert_called_once()
    assert store.get_open_position_states() == []
    store.close()
