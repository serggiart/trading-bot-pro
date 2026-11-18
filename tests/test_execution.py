"""Unit tests for limit-only order execution and dynamic exit management."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, ReplaceOrderRequest

from src.execution import (
    NY_TIMEZONE,
    AccountSnapshot,
    AlpacaExecutionClient,
    ExecutionConfig,
    ManagedPosition,
    PositionManager,
    calculate_limit_buy_price,
    calculate_limit_sell_price,
    compute_trailing_stop,
    is_flash_crash_triggered,
    is_time_window_expired,
)
from src.guardian import PortfolioPosition
from src.storage import TradingAuditStore


def _make_mock_trading_client(order_id: str = "brk-123", status: str = "accepted") -> MagicMock:
    client = MagicMock()
    client.submit_order.return_value = SimpleNamespace(id=order_id, status=status)
    client.replace_order_by_id.return_value = SimpleNamespace(id=order_id, status="replaced")
    return client


# ---------------------------------------------------------------------------
# 1. Static AST Safety Check (No MarketOrderRequest / close_position)
# ---------------------------------------------------------------------------


def test_execution_never_uses_market_orders_or_close_position() -> None:
    """Verify via AST that MarketOrderRequest and close_position never appear in src/execution/*.py."""
    execution_dir = Path(__file__).resolve().parents[1] / "src" / "execution"
    py_files = sorted(execution_dir.glob("*.py"))
    assert py_files, "Expected Python modules in src/execution/"

    forbidden_identifiers = {"MarketOrderRequest", "close_position"}

    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_identifiers, (
                    f"Forbidden identifier '{node.id}' found in {py_file.name}"
                )
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_identifiers, (
                    f"Forbidden attribute '{node.attr}' found in {py_file.name}"
                )
            elif isinstance(node, ast.alias):
                assert node.name not in forbidden_identifiers, (
                    f"Forbidden import '{node.name}' found in {py_file.name}"
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for forbidden in forbidden_identifiers:
                    assert forbidden not in node.value, (
                        f"Forbidden string reference '{forbidden}' found in {py_file.name}"
                    )


# ---------------------------------------------------------------------------
# 2. Slippage Helpers & ALPACA_PAPER Configuration
# ---------------------------------------------------------------------------


def test_slippage_price_calculations_and_validation() -> None:
    assert calculate_limit_buy_price(100.0, 0.001) == 100.10
    assert calculate_limit_sell_price(100.0, 0.003) == 99.70
    assert calculate_limit_sell_price(100.0, 0.015) == 98.50
    # Sub-cent protection
    assert calculate_limit_sell_price(0.01, 0.5) == 0.01

    with pytest.raises(ValueError):
        calculate_limit_buy_price(0.0, 0.001)
    with pytest.raises(ValueError):
        calculate_limit_buy_price(100.0, -0.01)
    with pytest.raises(ValueError):
        calculate_limit_sell_price(-10.0, 0.003)
    with pytest.raises(ValueError):
        calculate_limit_sell_price(100.0, 1.0)


def test_alpaca_paper_defaults_to_true_and_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    client_default = AlpacaExecutionClient(trading_client=_make_mock_trading_client())
    assert client_default.paper is True
    assert client_default.config.paper is True

    monkeypatch.setenv("ALPACA_PAPER", "false")
    client_live_false = AlpacaExecutionClient(trading_client=_make_mock_trading_client())
    assert client_live_false.paper is False

    monkeypatch.setenv("ALPACA_PAPER", "0")
    client_live_zero = AlpacaExecutionClient(trading_client=_make_mock_trading_client())
    assert client_live_zero.paper is False

    monkeypatch.setenv("ALPACA_PAPER", "true")
    client_paper_true = AlpacaExecutionClient(trading_client=_make_mock_trading_client())
    assert client_paper_true.paper is True


# ---------------------------------------------------------------------------
# 3. Limit Buy / Sell / Stop Replacement & Audit Logging
# ---------------------------------------------------------------------------


def test_submit_limit_buy_oto_and_audit_logging(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    store = TradingAuditStore(db_path=db_path)
    mock_broker = _make_mock_trading_client(order_id="alpaca-buy-1", status="accepted")
    client = AlpacaExecutionClient(audit_store=store, trading_client=mock_broker)

    result = client.submit_limit_buy(
        symbol="aapl",
        qty=10,
        ask_price=150.00,
        stop_loss_price=144.00,
        client_order_id="cid-buy-1",
    )

    assert result.submitted is True
    assert result.symbol == "AAPL"
    assert result.side == "BUY"
    assert result.order_type == "limit"
    assert result.limit_price == 150.15
    assert result.stop_price == 144.00
    assert result.broker_order_id == "alpaca-buy-1"
    assert result.status == "accepted"

    mock_broker.submit_order.assert_called_once()
    submitted_req = mock_broker.submit_order.call_args.kwargs["order_data"]
    assert isinstance(submitted_req, LimitOrderRequest)
    assert submitted_req.side == OrderSide.BUY
    assert submitted_req.time_in_force == TimeInForce.DAY
    assert submitted_req.order_class == OrderClass.OTO
    assert submitted_req.limit_price == 150.15
    assert submitted_req.stop_loss is not None
    assert submitted_req.stop_loss.stop_price == 144.00
    assert submitted_req.take_profit is None

    orders = store.get_orders(symbol="AAPL")
    assert len(orders) == 1
    assert orders[0].client_order_id == "cid-buy-1"
    assert orders[0].broker_order_id == "alpaca-buy-1"
    assert orders[0].limit_price == 150.15
    assert orders[0].stop_price == 144.00
    assert orders[0].created_at.tzinfo == NY_TIMEZONE
    store.close()


def test_submit_limit_buy_bracket_with_take_profit(tmp_path: Path) -> None:
    store = TradingAuditStore(db_path=tmp_path / "audit.db")
    mock_broker = _make_mock_trading_client(order_id="alpaca-bracket-1")
    client = AlpacaExecutionClient(audit_store=store, trading_client=mock_broker)

    result = client.submit_limit_buy(
        symbol="MSFT",
        qty=5,
        ask_price=400.00,
        stop_loss_price=388.00,
        take_profit_price=420.00,
        client_order_id="cid-bracket-1",
    )

    assert result.submitted is True
    submitted_req = mock_broker.submit_order.call_args.kwargs["order_data"]
    assert isinstance(submitted_req, LimitOrderRequest)
    assert submitted_req.order_class == OrderClass.BRACKET
    assert submitted_req.stop_loss is not None
    assert submitted_req.stop_loss.stop_price == 388.00
    assert submitted_req.take_profit is not None
    assert submitted_req.take_profit.limit_price == 420.00
    store.close()


def test_submit_limit_buy_dry_run_skips_broker_and_logs_dry_run(tmp_path: Path) -> None:
    store = TradingAuditStore(db_path=tmp_path / "audit.db")
    mock_broker = _make_mock_trading_client()
    client = AlpacaExecutionClient(audit_store=store, trading_client=mock_broker)

    result = client.submit_limit_buy(
        symbol="NVDA",
        qty=4,
        ask_price=120.00,
        stop_loss_price=114.00,
        client_order_id="cid-dry-1",
        dry_run=True,
    )

    assert result.submitted is False
    assert result.status == "dry_run"
    assert result.broker_order_id is None
    assert result.limit_price == 120.12
    mock_broker.submit_order.assert_not_called()

    orders = store.get_orders(symbol="NVDA")
    assert len(orders) == 1
    assert orders[0].status == "dry_run"
    assert orders[0].broker_order_id is None
    store.close()


def test_submit_limit_buy_validation_rejects_invalid_stop_or_qty() -> None:
    client = AlpacaExecutionClient(trading_client=_make_mock_trading_client())

    with pytest.raises(ValueError):
        client.submit_limit_buy("AAPL", qty=0, ask_price=100.0, stop_loss_price=95.0)
    with pytest.raises(ValueError):
        client.submit_limit_buy("AAPL", qty=1, ask_price=100.0, stop_loss_price=0.0)
    with pytest.raises(ValueError):
        client.submit_limit_buy("AAPL", qty=1, ask_price=100.0, stop_loss_price=100.10)
    with pytest.raises(ValueError):
        client.submit_limit_buy(
            "AAPL", qty=1, ask_price=100.0, stop_loss_price=95.0, take_profit_price=99.0
        )


def test_update_stop_order_calls_replace_and_respects_dry_run() -> None:
    mock_broker = _make_mock_trading_client()
    client = AlpacaExecutionClient(trading_client=mock_broker)

    assert client.update_stop_order("stop-1", 104.567, dry_run=True) is False
    mock_broker.replace_order_by_id.assert_not_called()

    assert client.update_stop_order("stop-1", 104.567, dry_run=False) is True
    mock_broker.replace_order_by_id.assert_called_once()
    kwargs = mock_broker.replace_order_by_id.call_args.kwargs
    assert kwargs["order_id"] == "stop-1"
    assert isinstance(kwargs["order_data"], ReplaceOrderRequest)
    assert kwargs["order_data"].stop_price == 104.57


# ---------------------------------------------------------------------------
# 4. Trailing Stop Ratchet Invariant (NEVER moves down)
# ---------------------------------------------------------------------------


def test_compute_trailing_stop_ratchet_invariant() -> None:
    # Price rises -> stop ratchets up
    stop_up = compute_trailing_stop(
        entry_price=100.0,
        current_price=110.0,
        current_stop=96.0,
        atr_14=2.0,
        atr_multiplier=2.0,
    )
    assert stop_up == 106.0

    # Price dips -> stop NEVER moves down
    stop_after_dip = compute_trailing_stop(
        entry_price=100.0,
        current_price=102.0,
        current_stop=stop_up,
        atr_14=2.0,
        atr_multiplier=2.0,
        highest_price=110.0,
    )
    assert stop_after_dip == 106.0

    # Even if current_stop > peak - atr*mult, current_stop wins
    stop_high_floor = compute_trailing_stop(
        entry_price=100.0,
        current_price=101.0,
        current_stop=99.0,
        atr_14=2.0,
        atr_multiplier=2.0,
    )
    assert stop_high_floor == 99.0

    # atr_14 <= 0 returns current_stop safely
    assert compute_trailing_stop(100.0, 120.0, 96.0, atr_14=0.0) == 96.0
    assert compute_trailing_stop(100.0, 120.0, 96.0, atr_14=-1.5) == 96.0


def test_position_manager_trailing_ratchet_and_stop_exit(tmp_path: Path) -> None:
    store = TradingAuditStore(db_path=tmp_path / "audit.db")
    mock_broker = _make_mock_trading_client(order_id="sell-stop-1")
    client = AlpacaExecutionClient(
        audit_store=store,
        trading_client=mock_broker,
        config=ExecutionConfig(trailing_atr_multiplier=2.0),
    )
    manager = PositionManager(execution_client=client, audit_store=store)

    opened_at = datetime(2026, 5, 1, 10, 0, tzinfo=NY_TIMEZONE)
    pos = ManagedPosition(
        symbol="AAPL",
        qty=10,
        entry_price=100.0,
        current_stop_price=96.0,
        atr_14=2.0,
        opened_at=opened_at,
        stop_order_id="broker-stop-1",
    )

    # Step 1: Price rises to 110 (+10%) after 30 minutes -> trail updates to 106.0
    res_up = manager.evaluate_and_manage(
        pos,
        current_price=110.0,
        now=opened_at + timedelta(minutes=30),
    )
    assert res_up.action == "TRAIL_UPDATE"
    assert res_up.should_exit is False
    assert res_up.new_stop_price == 106.0
    assert pos.current_stop_price == 106.0
    assert pos.highest_price == 110.0
    mock_broker.replace_order_by_id.assert_called_once()

    # Step 2: Price dips to 107.5 -> stop MUST remain 106.0 and action is HOLD
    res_dip = manager.evaluate_and_manage(
        pos,
        current_price=107.5,
        now=opened_at + timedelta(minutes=45),
    )
    assert res_dip.action == "HOLD"
    assert res_dip.should_exit is False
    assert res_dip.new_stop_price == 106.0
    assert pos.current_stop_price == 106.0
    assert mock_broker.replace_order_by_id.call_count == 1

    # Step 3: Price hits trailing stop (<= 106.0) -> TRAILING_STOP_EXIT limit sell
    res_exit = manager.evaluate_and_manage(
        pos,
        current_price=105.5,
        bid_price=105.40,
        now=opened_at + timedelta(minutes=60),
    )
    assert res_exit.action == "TRAILING_STOP_EXIT"
    assert res_exit.should_exit is True
    assert res_exit.order_result is not None
    assert res_exit.order_result.side == "SELL"
    assert res_exit.order_result.order_type == "limit"
    assert res_exit.order_result.limit_price == calculate_limit_sell_price(105.40, 0.003)

    submitted_req = mock_broker.submit_order.call_args.kwargs["order_data"]
    assert isinstance(submitted_req, LimitOrderRequest)
    assert submitted_req.side == OrderSide.SELL
    store.close()


# ---------------------------------------------------------------------------
# 5. 5-Minute Flash Crash Guard
# ---------------------------------------------------------------------------


def test_flash_crash_triggers_within_5_minutes_and_logs_anomaly(tmp_path: Path) -> None:
    store = TradingAuditStore(db_path=tmp_path / "audit.db")
    mock_broker = _make_mock_trading_client(order_id="emg-sell-1")
    client = AlpacaExecutionClient(
        audit_store=store,
        trading_client=mock_broker,
        config=ExecutionConfig(trailing_atr_multiplier=2.0),
    )
    manager = PositionManager(execution_client=client, audit_store=store)

    opened_at = datetime(2026, 5, 1, 9, 35, tzinfo=NY_TIMEZONE)
    pos = ManagedPosition(
        symbol="TSLA",
        qty=5,
        entry_price=200.0,
        current_stop_price=190.0,
        atr_14=4.0,
        opened_at=opened_at,
    )

    # 4 minutes later, price plunges 4% to 192.0 (even above stop 190.0)
    res = manager.evaluate_and_manage(
        pos,
        current_price=192.0,
        bid_price=191.80,
        now=opened_at + timedelta(minutes=4),
    )

    assert res.action == "EMERGENCY_EXIT"
    assert res.should_exit is True
    assert res.order_result is not None
    assert res.order_result.submitted is True
    assert res.order_result.side == "SELL"
    assert res.order_result.limit_price == calculate_limit_sell_price(191.80, 0.015)

    submitted_req = mock_broker.submit_order.call_args.kwargs["order_data"]
    assert isinstance(submitted_req, LimitOrderRequest)
    assert submitted_req.side == OrderSide.SELL

    # Verify anomaly logged in both orders and guardian_decisions tables
    orders = store.get_orders(symbol="TSLA")
    assert len(orders) == 1
    assert orders[0].metadata["anomaly"] == "flash_crash"
    assert orders[0].metadata["exit_type"] == "EMERGENCY_EXIT"

    decisions = store.get_guardian_decisions(symbol="TSLA")
    assert len(decisions) == 1
    assert decisions[0].action == "EMERGENCY_EXIT"
    assert decisions[0].approved is False
    assert "Flash crash guard triggered" in decisions[0].veto_reasons[0]
    store.close()


def test_flash_crash_does_not_trigger_after_5_minutes_and_normalizes_tz() -> None:
    opened_ny = datetime(2026, 5, 1, 9, 35, tzinfo=NY_TIMEZONE)
    # 5 minutes + 1 second expressed in UTC
    after_window_utc = (opened_ny + timedelta(minutes=5, seconds=1)).astimezone(timezone.utc)
    within_window_utc = (opened_ny + timedelta(minutes=5)).astimezone(timezone.utc)

    assert (
        is_flash_crash_triggered(
            entry_price=100.0,
            current_price=96.0,
            opened_at=opened_ny,
            now=within_window_utc,
        )
        is True
    )
    assert (
        is_flash_crash_triggered(
            entry_price=100.0,
            current_price=96.0,
            opened_at=opened_ny,
            now=after_window_utc,
        )
        is False
    )

    # Verify PositionManager does NOT emit EMERGENCY_EXIT after 5 minutes
    mock_broker = _make_mock_trading_client()
    client = AlpacaExecutionClient(trading_client=mock_broker)
    manager = PositionManager(execution_client=client)
    pos = ManagedPosition(
        symbol="AMD",
        qty=10,
        entry_price=100.0,
        current_stop_price=94.0,
        atr_14=3.0,
        opened_at=opened_ny,
    )
    res = manager.evaluate_and_manage(pos, current_price=96.0, now=after_window_utc)
    assert res.action == "HOLD"
    assert res.should_exit is False


# ---------------------------------------------------------------------------
# 6. Time-Window Exit (Opportunity Cost Guard)
# ---------------------------------------------------------------------------


def test_time_window_exit_closes_stagnant_and_holds_profitable(tmp_path: Path) -> None:
    opened_at = datetime(2026, 5, 1, 10, 0, tzinfo=NY_TIMEZONE)

    # Before 72h -> never expired
    assert (
        is_time_window_expired(
            entry_price=100.0,
            current_price=100.2,
            opened_at=opened_at,
            now=opened_at + timedelta(hours=71, minutes=59),
        )
        is False
    )
    # At 72h with 0.5% return (< 1.0%) -> expired
    assert (
        is_time_window_expired(
            entry_price=100.0,
            current_price=100.5,
            opened_at=opened_at,
            now=opened_at + timedelta(hours=72),
        )
        is True
    )
    # At 72h with 1.0% return (>= 1.0%) -> hold
    assert (
        is_time_window_expired(
            entry_price=100.0,
            current_price=101.0,
            opened_at=opened_at,
            now=opened_at + timedelta(hours=72),
        )
        is False
    )

    store = TradingAuditStore(db_path=tmp_path / "audit.db")
    mock_broker = _make_mock_trading_client(order_id="time-sell-1")
    client = AlpacaExecutionClient(
        audit_store=store,
        trading_client=mock_broker,
        config=ExecutionConfig(trailing_atr_multiplier=2.0),
    )
    manager = PositionManager(execution_client=client, audit_store=store)

    stagnant_pos = ManagedPosition(
        symbol="KO",
        qty=20,
        entry_price=100.0,
        current_stop_price=95.0,
        atr_14=2.0,
        opened_at=opened_at,
    )
    res_exit = manager.evaluate_and_manage(
        stagnant_pos,
        current_price=100.40,
        now=opened_at + timedelta(hours=73),
    )
    assert res_exit.action == "TIME_EXIT"
    assert res_exit.should_exit is True
    assert res_exit.order_result is not None
    assert res_exit.order_result.side == "SELL"
    assert res_exit.order_result.limit_price == calculate_limit_sell_price(100.40, 0.003)

    profitable_pos = ManagedPosition(
        symbol="KO",
        qty=20,
        entry_price=100.0,
        current_stop_price=97.5,
        atr_14=2.0,
        opened_at=opened_at,
    )
    res_hold = manager.evaluate_and_manage(
        profitable_pos,
        current_price=101.50,
        now=opened_at + timedelta(hours=73),
    )
    assert res_hold.action == "HOLD"
    assert res_hold.should_exit is False
    store.close()


# ---------------------------------------------------------------------------
# 7. Broker Account Snapshot & Open Positions Sync
# ---------------------------------------------------------------------------


def test_get_account_snapshot_and_open_positions_offline_and_mocked() -> None:
    offline_client = AlpacaExecutionClient()
    assert offline_client.get_account_snapshot() is None
    assert offline_client.get_open_positions() == []

    mock_broker = _make_mock_trading_client()
    mock_broker.get_account.return_value = SimpleNamespace(
        equity="10525.50",
        cash="4200.00",
        buying_power="14725.50",
        last_equity="10000.00",
    )
    mock_broker.get_all_positions.return_value = [
        SimpleNamespace(
            symbol="aapl",
            qty="10",
            market_value="2000.00",
            avg_entry_price="195.00",
        ),
        SimpleNamespace(
            symbol="MSFT",
            qty="5",
            market_value="2100.00",
            avg_entry_price="410.00",
        ),
    ]

    client = AlpacaExecutionClient(trading_client=mock_broker)
    snapshot = client.get_account_snapshot()
    assert isinstance(snapshot, AccountSnapshot)
    assert snapshot.equity == pytest.approx(10525.50)
    assert snapshot.cash == pytest.approx(4200.00)
    assert snapshot.buying_power == pytest.approx(14725.50)
    assert snapshot.last_equity == pytest.approx(10000.00)
    assert snapshot.daily_pnl == pytest.approx(525.50)

    positions = client.get_open_positions(sector_map={"AAPL": "Technology"}, default_sector="Equities")
    assert len(positions) == 2
    assert all(isinstance(p, PortfolioPosition) for p in positions)
    assert positions[0].symbol == "AAPL"
    assert positions[0].sector == "Technology"
    assert positions[0].shares == pytest.approx(10.0)
    assert positions[0].market_value == pytest.approx(2000.0)
    assert positions[0].avg_price == pytest.approx(195.0)
    assert positions[1].symbol == "MSFT"
    assert positions[1].sector == "Equities"

    cfg = ExecutionConfig()
    assert cfg.take_profit_atr_multiplier == pytest.approx(2.0)
    assert cfg.trailing_atr_multiplier == pytest.approx(2.0)
    with pytest.raises(ValueError, match="take_profit_atr_multiplier"):
        ExecutionConfig(take_profit_atr_multiplier=-0.5)


def test_crash_safe_position_reconciliation_from_broker_and_sqlite(tmp_path: Path) -> None:
    store = TradingAuditStore(tmp_path / "reconcile.db")
    opened_at = datetime(2026, 11, 18, 10, 0, tzinfo=NY_TIMEZONE)

    # Simulate persisted trailing-stop high-watermark + stale closed symbol before crash
    store.upsert_position_state(
        symbol="AAPL",
        qty=10.0,
        entry_price=200.0,
        current_stop_price=203.50,
        atr_14=2.5,
        opened_at=opened_at,
        highest_price=208.50,
        stop_order_id="stop-aapl-1",
        client_order_id="buy-aapl-1",
    )
    store.upsert_position_state(
        symbol="STALE",
        qty=5.0,
        entry_price=50.0,
        current_stop_price=48.0,
        atr_14=1.0,
        opened_at=opened_at,
    )

    mock_broker = MagicMock()
    mock_broker.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="10", avg_entry_price="200.00"),
        SimpleNamespace(symbol="MSFT", qty="4", avg_entry_price="400.00"),
    ]
    mock_broker.get_orders.return_value = [
        SimpleNamespace(
            id="stop-aapl-1",
            symbol="AAPL",
            side="sell",
            order_type="stop",
            stop_price="202.00",
        ),
        SimpleNamespace(
            id="stop-msft-1",
            symbol="MSFT",
            side="sell",
            order_type="stop",
            stop_price="392.00",
        ),
    ]

    client = AlpacaExecutionClient(audit_store=store, trading_client=mock_broker)
    now_ny = opened_at + timedelta(hours=4)
    reconciled = client.reconcile_managed_positions(atr_lookup={"MSFT": 4.0}, now=now_ny)

    assert [p.symbol for p in reconciled] == ["AAPL", "MSFT"]
    aapl_pos, msft_pos = reconciled
    # Preserves higher persisted trailing stop (203.50 > broker 202.00) and original opened_at
    assert aapl_pos.current_stop_price == pytest.approx(203.50)
    assert aapl_pos.highest_price == pytest.approx(208.50)
    assert aapl_pos.opened_at == opened_at
    assert aapl_pos.stop_order_id == "stop-aapl-1"

    # Reconstructs MSFT from broker stop order + atr_lookup
    assert msft_pos.current_stop_price == pytest.approx(392.00)
    assert msft_pos.atr_14 == pytest.approx(4.0)
    assert msft_pos.stop_order_id == "stop-msft-1"

    # Stale closed symbol was purged from SQLite position_state
    assert store.get_position_state("STALE") is None
    assert store.get_position_state("MSFT") is not None


def test_submit_limit_buy_fill_gated_state_and_sell_cancels_bracket_legs(
    tmp_path: Path,
) -> None:
    store = TradingAuditStore(tmp_path / "bracket_legs.db")
    mock_broker = MagicMock()

    # 1) Unfilled accepted buy must NOT write position_state prematurely
    mock_broker.submit_order.return_value = SimpleNamespace(
        id="broker-buy-accepted",
        status="accepted",
        legs=[],
    )
    client = AlpacaExecutionClient(audit_store=store, trading_client=mock_broker)
    unfilled_res = client.submit_limit_buy(
        symbol="AAPL",
        qty=10,
        ask_price=200.0,
        stop_loss_price=194.0,
        take_profit_price=208.0,
        metadata={"atr_14": 3.0},
    )
    assert unfilled_res.status == "accepted"
    assert store.get_position_state("AAPL") is None

    # Once Alpaca later fills it, reconcile_managed_positions recovers atr_14 & stop from orders
    mock_broker.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="10", avg_entry_price="200.10"),
    ]
    mock_broker.get_orders.return_value = [
        SimpleNamespace(id="leg-stop-1", symbol="AAPL", side="sell", order_type="stop", stop_price="194.00"),
        SimpleNamespace(id="leg-tp-1", symbol="AAPL", side="sell", order_type="limit", limit_price="208.00"),
        SimpleNamespace(id="other-sym-1", symbol="NVDA", side="sell", order_type="stop", stop_price="140.00"),
    ]
    reconciled = client.reconcile_managed_positions()
    assert len(reconciled) == 1
    assert reconciled[0].atr_14 == pytest.approx(3.0)
    assert reconciled[0].current_stop_price == pytest.approx(194.0)
    assert reconciled[0].stop_order_id == "leg-stop-1"
    assert store.get_orders(symbol="AAPL")[0].status == "filled"

    # 2) Filled buy immediately records position_state with child stop leg ID
    mock_broker.submit_order.return_value = SimpleNamespace(
        id="broker-buy-filled",
        status="filled",
        legs=[SimpleNamespace(id="child-stop-99", order_type="stop")],
    )
    client.submit_limit_buy(
        symbol="MSFT",
        qty=5,
        ask_price=400.0,
        stop_loss_price=390.0,
        metadata={"atr_14": 5.0},
    )
    msft_state = store.get_position_state("MSFT")
    assert msft_state is not None
    assert msft_state.stop_order_id == "child-stop-99"

    # 3) submit_limit_sell cancels only AAPL's open bracket legs before placing Limit sell
    mock_broker.submit_order.return_value = SimpleNamespace(
        id="broker-sell-filled",
        status="filled",
    )
    sell_res = client.submit_limit_sell(
        symbol="AAPL",
        qty=10,
        reference_price=198.0,
        reason="TIME_EXIT",
    )
    assert sell_res.submitted is True
    cancelled_ids = [call.args[0] for call in mock_broker.cancel_order_by_id.call_args_list]
    assert cancelled_ids == ["leg-stop-1", "leg-tp-1"]
    assert store.get_position_state("AAPL") is None
