"""Unit tests for the local SQLite persistence layer in src/storage/."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src.storage import (
    DEFAULT_DB_PATH,
    DEFAULT_FMP_DAILY_LIMIT,
    NY_TIMEZONE,
    AiSignalRecord,
    AuditDatabase,
    DuplicateOrderError,
    FmpCacheEntry,
    GuardianDecisionRecord,
    OrderRecord,
    TradingAuditStore,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide an isolated temporary SQLite path so data/trading_audit.db is never touched."""
    return tmp_path / "nested" / "test_trading_audit.db"


@pytest.fixture
def store(db_path: Path) -> TradingAuditStore:
    """Provide a fresh TradingAuditStore backed by a temporary file."""
    return TradingAuditStore(db_path=db_path)


def test_schema_initialization_wal_and_idempotent_reopen(db_path: Path) -> None:
    """Creating and reopening the DB initializes all tables, indexes, and WAL mode cleanly."""
    assert DEFAULT_DB_PATH == Path("data/trading_audit.db")
    assert AuditDatabase is TradingAuditStore

    first_store = TradingAuditStore(db_path=db_path)
    assert db_path.exists()

    with sqlite3.connect(db_path) as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode.lower() == "wal"

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "guardian_decisions",
            "fmp_cache",
            "fmp_api_usage",
            "ai_signals",
            "orders",
        }.issubset(tables)

        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert {
            "idx_guardian_decisions_symbol_decided",
            "idx_guardian_decisions_approved_decided",
            "idx_fmp_cache_cache_date",
            "idx_fmp_cache_expires_at",
            "idx_fmp_api_usage_date",
            "idx_fmp_api_usage_symbol_date",
            "idx_ai_signals_symbol_created",
            "idx_orders_symbol_status_created",
            "idx_orders_status_created",
        }.issubset(indexes)

    decision_id = first_store.log_guardian_decision(
        symbol="AAPL",
        action="BUY",
        approved=False,
        veto_reasons=["VIX > 30"],
    )

    # Reopening an existing database must be idempotent and preserve existing rows.
    reopened_store = TradingAuditStore(db_path=db_path)
    decisions = reopened_store.get_guardian_decisions()
    assert len(decisions) == 1
    assert decisions[0].id == decision_id
    assert decisions[0].veto_reasons == ["VIX > 30"]


def test_transaction_rollback_on_error(store: TradingAuditStore) -> None:
    """Exceptions inside store.transaction() roll back uncommitted writes."""
    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO fmp_api_usage (endpoint, symbol, request_date, requested_at)
                VALUES (?, ?, ?, ?)
                """,
                ("/profile", "AAPL", "2026-11-18", "2026-11-18T10:00:00-05:00"),
            )
            raise RuntimeError("abort")

    assert store.get_fmp_requests_today(
        now=datetime(2026, 11, 18, 10, 5, tzinfo=NY_TIMEZONE)
    ) == 0


def test_guardian_decisions_logging_and_filtering(store: TradingAuditStore) -> None:
    """Guardian vetoes and approvals preserve exact veto reasons, sizing, metadata, and NY timestamps."""
    t1_utc = datetime(2026, 11, 18, 14, 30, tzinfo=UTC)  # 09:30 America/New_York
    t2_ny = datetime(2026, 11, 18, 10, 15, tzinfo=NY_TIMEZONE)
    t3_ny = datetime(2026, 11, 18, 11, 0, tzinfo=NY_TIMEZONE)

    veto_id = store.log_guardian_decision(
        symbol="aapl",
        action="buy",
        approved=False,
        veto_reasons=["VIX > 30", "Earnings within 24h"],
        vix=32.4,
        recommended_shares=0.0,
        recommended_notional=0.0,
        metadata={"earnings_hours_away": 12.5, "circuit_breaker": False},
        decided_at=t1_utc,
    )
    approved_id = store.log_guardian_decision(
        symbol="MSFT",
        action="BUY",
        approved=True,
        veto_reasons=[],
        vix=18.2,
        recommended_shares=15.0,
        recommended_notional=6450.0,
        metadata={"atr": 4.25, "kelly_fraction": 0.25},
        decided_at=t2_ny,
    )
    store.log_guardian_decision(
        symbol="AAPL",
        action="BUY",
        approved=True,
        veto_reasons=[],
        vix=17.9,
        recommended_shares=10.0,
        recommended_notional=2300.0,
        decided_at=t3_ny,
    )

    all_records = store.get_guardian_decisions()
    assert len(all_records) == 3
    assert all(isinstance(rec, GuardianDecisionRecord) for rec in all_records)
    # Ordered newest first
    assert [rec.symbol for rec in all_records] == ["AAPL", "MSFT", "AAPL"]

    vetoed = store.get_guardian_decisions(approved=False)
    assert len(vetoed) == 1
    assert vetoed[0].id == veto_id
    assert vetoed[0].symbol == "AAPL"
    assert vetoed[0].action == "BUY"
    assert vetoed[0].approved is False
    assert vetoed[0].veto_reasons == ["VIX > 30", "Earnings within 24h"]
    assert vetoed[0].vix == pytest.approx(32.4)
    assert vetoed[0].metadata == {"earnings_hours_away": 12.5, "circuit_breaker": False}
    assert vetoed[0].decided_at.tzinfo == NY_TIMEZONE
    assert vetoed[0].decided_at == t1_utc.astimezone(NY_TIMEZONE)

    msft_approved = store.get_guardian_decisions(symbol="msft", approved=True)
    assert len(msft_approved) == 1
    assert msft_approved[0].id == approved_id
    assert msft_approved[0].recommended_shares == pytest.approx(15.0)
    assert msft_approved[0].recommended_notional == pytest.approx(6450.0)

    limited = store.get_guardian_decisions(symbol="AAPL", limit=1)
    assert len(limited) == 1
    assert limited[0].approved is True


def test_fmp_cache_hit_upsert_ttl_and_ny_day_rollover(store: TradingAuditStore) -> None:
    """FMP cache supports upserts, invalidates on TTL expiration, and invalidates on NY day rollover."""
    morning_ny = datetime(2026, 11, 18, 9, 30, tzinfo=NY_TIMEZONE)

    # Cache miss when empty
    assert store.get_fmp_cache("NVDA", "fundamentals", now=morning_ny) is None

    # Upsert initial payload
    created = store.upsert_fmp_cache(
        symbol="nvda",
        data_type="fundamentals",
        payload={"free_cash_flow": 50_000_000_000, "debt_to_equity": 0.41},
        earnings_date=date(2026, 11, 25),
        ttl_hours=4,
        now=morning_ny,
    )
    assert isinstance(created, FmpCacheEntry)
    assert created.symbol == "NVDA"
    assert created.cache_date == "2026-11-18"
    assert created.earnings_date == "2026-11-25"

    # Cache hit before TTL and within same NY day
    hit = store.get_fmp_cache(
        "NVDA", "fundamentals", now=morning_ny + timedelta(hours=3, minutes=59)
    )
    assert hit is not None
    assert hit.payload["free_cash_flow"] == 50_000_000_000

    # Cache miss at exact TTL boundary (now >= expires_at)
    assert (
        store.get_fmp_cache("NVDA", "fundamentals", now=morning_ny + timedelta(hours=4))
        is None
    )

    # Upsert updates existing row rather than duplicating
    late_night_ny = datetime(2026, 11, 18, 23, 30, tzinfo=NY_TIMEZONE)
    store.upsert_fmp_cache(
        symbol="NVDA",
        data_type="fundamentals",
        payload={"free_cash_flow": 55_000_000_000, "debt_to_equity": 0.39},
        earnings_date="2026-11-25",
        ttl_hours=24,
        now=late_night_ny,
    )
    same_day_hit = store.get_fmp_cache(
        "NVDA", "fundamentals", now=datetime(2026, 11, 18, 23, 59, tzinfo=NY_TIMEZONE)
    )
    assert same_day_hit is not None
    assert same_day_hit.payload["free_cash_flow"] == 55_000_000_000

    # NY day rollover invalidates cache even when TTL (24h) has NOT elapsed!
    next_day_ny = datetime(2026, 11, 19, 0, 5, tzinfo=NY_TIMEZONE)
    assert store.get_fmp_cache("NVDA", "fundamentals", now=next_day_ny) is None

    # UTC timestamp corresponding to 23:55 NY vs 00:05 NY next day (EST = UTC-5)
    utc_before_ny_midnight = datetime(2026, 11, 19, 4, 55, tzinfo=UTC)  # 23:55 NY Nov 18
    assert store.get_fmp_cache("NVDA", "fundamentals", now=utc_before_ny_midnight) is not None
    utc_after_ny_midnight = datetime(2026, 11, 19, 5, 5, tzinfo=UTC)  # 00:05 NY Nov 19
    assert store.get_fmp_cache("NVDA", "fundamentals", now=utc_after_ny_midnight) is None


def test_purge_expired_cache(store: TradingAuditStore) -> None:
    """purge_expired_cache deletes TTL-expired and prior-day entries while keeping fresh ones."""
    day1 = datetime(2026, 11, 18, 22, 0, tzinfo=NY_TIMEZONE)
    day2_morning = datetime(2026, 11, 19, 8, 0, tzinfo=NY_TIMEZONE)

    # Prior-day entry (stale on day 2 despite 24h TTL)
    store.upsert_fmp_cache("AAPL", "cash_flow", {"ocf": 100}, ttl_hours=24, now=day1)
    # Day 2 entry with short TTL that expires before day2_check
    store.upsert_fmp_cache("TSLA", "cash_flow", {"ocf": 20}, ttl_hours=1, now=day2_morning)
    # Day 2 entry still valid at day2_check
    store.upsert_fmp_cache("MSFT", "cash_flow", {"ocf": 90}, ttl_hours=12, now=day2_morning)

    day2_check = day2_morning + timedelta(hours=2)  # 10:00 NY on Nov 19
    purged = store.purge_expired_cache(now=day2_check)
    assert purged == 2

    assert store.get_fmp_cache("AAPL", "cash_flow", now=day2_check) is None
    assert store.get_fmp_cache("TSLA", "cash_flow", now=day2_check) is None
    valid = store.get_fmp_cache("MSFT", "cash_flow", now=day2_check)
    assert valid is not None
    assert valid.payload == {"ocf": 90}

    # Purging again when nothing else is expired returns 0
    assert store.purge_expired_cache(now=day2_check) == 0


def test_fmp_daily_rate_limit_tracking_and_ny_reset(store: TradingAuditStore) -> None:
    """FMP request quota tracks 250/day per America/New_York date and resets at NY midnight."""
    assert DEFAULT_FMP_DAILY_LIMIT == 250
    day1_morning = datetime(2026, 11, 18, 9, 0, tzinfo=NY_TIMEZONE)

    assert store.get_fmp_requests_today(now=day1_morning) == 0
    assert store.can_make_fmp_request(now=day1_morning) is True

    for idx in range(249):
        rec_id = store.record_fmp_request(
            endpoint="/api/v3/cash-flow-statement",
            symbol="AAPL" if idx % 2 == 0 else None,
            now=day1_morning + timedelta(seconds=idx),
        )
        assert rec_id > 0

    assert store.get_fmp_requests_today(now=day1_morning) == 249
    assert store.can_make_fmp_request(daily_limit=250, now=day1_morning) is True

    # 250th request exhausts daily quota
    store.record_fmp_request(
        endpoint="/api/v3/earning_calendar",
        symbol="NVDA",
        now=datetime(2026, 11, 18, 23, 50, tzinfo=NY_TIMEZONE),
    )
    assert store.get_fmp_requests_today(now=day1_morning) == 250
    assert store.can_make_fmp_request(daily_limit=250, now=day1_morning) is False

    # Same instant expressed in UTC (Nov 19 04:50 UTC == Nov 18 23:50 EST) is STILL day 1
    still_day1_utc = datetime(2026, 11, 19, 4, 59, tzinfo=UTC)
    assert store.get_fmp_requests_today(now=still_day1_utc) == 250
    assert store.can_make_fmp_request(daily_limit=250, now=still_day1_utc) is False

    # Next NY day (Nov 19 00:01 EST == Nov 19 05:01 UTC) resets quota to 0
    next_ny_day_utc = datetime(2026, 11, 19, 5, 1, tzinfo=UTC)
    assert store.get_fmp_requests_today(now=next_ny_day_utc) == 0
    assert store.can_make_fmp_request(daily_limit=250, now=next_ny_day_utc) is True


def test_ai_signals_logging_and_querying(store: TradingAuditStore) -> None:
    """AI Analyst signals persist sentiment, confidence, consensus, sources, and threshold flag."""
    t1 = datetime(2026, 11, 18, 9, 45, tzinfo=NY_TIMEZONE)
    t2 = datetime(2026, 11, 18, 10, 0, tzinfo=NY_TIMEZONE)

    id1 = store.log_ai_signal(
        symbol="spy",
        sentiment="BULLISH",
        confidence_score=0.62,
        consensus_count=2,
        sources=["finnhub", "alpaca_news"],
        rationale="Only 2 confirming sources; below 0.70 threshold.",
        passed_threshold=False,
        created_at=t1,
    )
    id2 = store.log_ai_signal(
        symbol="SPY",
        sentiment="bullish",
        confidence_score=0.84,
        consensus_count=4,
        sources=["finnhub", "alpaca_news", "volume_spike", "sector_flow"],
        rationale="Strong institutional volume confirmation across 4 sources.",
        passed_threshold=True,
        created_at=t2,
    )
    store.log_ai_signal(
        symbol="QQQ",
        sentiment="neutral",
        confidence_score=0.51,
        consensus_count=1,
        sources=["finnhub"],
        rationale="Mixed headlines.",
        passed_threshold=False,
        created_at=t2,
    )

    spy_signals = store.get_ai_signals(symbol="SPY")
    assert len(spy_signals) == 2
    assert all(isinstance(sig, AiSignalRecord) for sig in spy_signals)
    assert [sig.id for sig in spy_signals] == [id2, id1]

    latest = spy_signals[0]
    assert latest.sentiment == "bullish"
    assert latest.confidence_score == pytest.approx(0.84)
    assert latest.consensus_count == 4
    assert latest.sources == ["finnhub", "alpaca_news", "volume_spike", "sector_flow"]
    assert latest.passed_threshold is True
    assert latest.created_at == t2

    limited = store.get_ai_signals(limit=2)
    assert len(limited) == 2


def test_orders_logging_duplicate_prevention_status_updates_and_queries(
    store: TradingAuditStore,
) -> None:
    """Orders enforce unique client_order_id, support status updates, and filter by symbol/status."""
    t_create = datetime(2026, 11, 18, 9, 35, tzinfo=NY_TIMEZONE)
    t_fill = datetime(2026, 11, 18, 9, 36, tzinfo=NY_TIMEZONE)

    order_id = store.log_order(
        client_order_id="bot-aapl-001",
        symbol="aapl",
        side="buy",
        qty=12.0,
        order_type="limit",
        limit_price=228.50,
        stop_price=224.00,
        status="pending",
        metadata={"trailing_stop_pct": 0.02},
        created_at=t_create,
    )
    assert order_id > 0

    # Duplicate client_order_id must be rejected deterministically
    with pytest.raises(DuplicateOrderError):
        store.log_order(
            client_order_id="bot-aapl-001",
            symbol="AAPL",
            side="BUY",
            qty=12.0,
            limit_price=228.50,
        )

    store.log_order(
        client_order_id="bot-msft-001",
        symbol="MSFT",
        side="BUY",
        qty=5.0,
        order_type="limit",
        limit_price=420.00,
        status="submitted",
        broker_order_id="alpaca-999",
        created_at=t_create + timedelta(seconds=10),
    )

    # Update status and assign broker_order_id
    updated = store.update_order_status(
        client_order_id="bot-aapl-001",
        status="submitted",
        broker_order_id="alpaca-123",
        updated_at=t_fill,
    )
    assert updated is True

    # Subsequent update without broker_order_id preserves existing broker_order_id
    updated_again = store.update_order_status(
        client_order_id="bot-aapl-001",
        status="filled",
        updated_at=t_fill + timedelta(seconds=15),
    )
    assert updated_again is True

    # Unknown client_order_id returns False
    assert (
        store.update_order_status(client_order_id="does-not-exist", status="cancelled")
        is False
    )

    filled_aapl = store.get_orders(symbol="AAPL", status="filled")
    assert len(filled_aapl) == 1
    rec = filled_aapl[0]
    assert isinstance(rec, OrderRecord)
    assert rec.client_order_id == "bot-aapl-001"
    assert rec.broker_order_id == "alpaca-123"
    assert rec.side == "BUY"
    assert rec.order_type == "limit"
    assert rec.qty == pytest.approx(12.0)
    assert rec.limit_price == pytest.approx(228.50)
    assert rec.stop_price == pytest.approx(224.00)
    assert rec.status == "filled"
    assert rec.metadata == {"trailing_stop_pct": 0.02}
    assert rec.created_at == t_create
    assert rec.updated_at == t_fill + timedelta(seconds=15)

    submitted_orders = store.get_orders(status="submitted")
    assert [o.client_order_id for o in submitted_orders] == ["bot-msft-001"]

    all_orders = store.get_orders(limit=10)
    assert [o.client_order_id for o in all_orders] == ["bot-msft-001", "bot-aapl-001"]


def test_in_memory_store_and_validation_guards() -> None:
    """In-memory store works across transactions and rejects invalid inputs."""
    with TradingAuditStore(":memory:") as mem_store:
        mem_store.record_fmp_request("/quote", symbol="SPY")
        assert mem_store.get_fmp_requests_today() == 1

        with pytest.raises(ValueError, match="symbol"):
            mem_store.log_guardian_decision(symbol="  ", action="BUY", approved=False)

        with pytest.raises(ValueError, match="ttl_hours"):
            mem_store.upsert_fmp_cache("AAPL", "profile", {}, ttl_hours=-1)

        with pytest.raises(ValueError, match="limit"):
            mem_store.get_orders(limit=0)


def test_position_state_persistence_and_cleanup(store: TradingAuditStore) -> None:
    """position_state upserts, preserves stop_order_id on partial updates, and deletes cleanly."""
    opened = datetime(2026, 11, 18, 9, 35, tzinfo=NY_TIMEZONE)
    rec = store.upsert_position_state(
        symbol="aapl",
        qty=10.0,
        entry_price=200.0,
        current_stop_price=195.0,
        atr_14=2.5,
        opened_at=opened,
        highest_price=201.0,
        stop_order_id="stop-1",
        client_order_id="cid-1",
    )
    assert rec.symbol == "AAPL"
    assert rec.current_stop_price == pytest.approx(195.0)
    assert rec.stop_order_id == "stop-1"

    # Upsert without stop_order_id preserves existing stop_order_id via COALESCE
    updated = store.upsert_position_state(
        symbol="AAPL",
        qty=10.0,
        entry_price=200.0,
        current_stop_price=198.0,
        atr_14=2.5,
        opened_at=opened,
        highest_price=204.0,
    )
    assert updated.current_stop_price == pytest.approx(198.0)
    assert updated.highest_price == pytest.approx(204.0)
    assert updated.stop_order_id == "stop-1"

    states = store.get_open_position_states()
    assert len(states) == 1
    assert states[0].symbol == "AAPL"
    assert store.get_position_state("aapl") is not None

    assert store.delete_position_state("AAPL") is True
    assert store.delete_position_state("AAPL") is False
    assert store.get_position_state("AAPL") is None
