"""SQLite persistence layer for trading audit trails and FMP daily quota/cache."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.storage.models import (
    NY_TIMEZONE,
    AiSignalRecord,
    FmpCacheEntry,
    GuardianDecisionRecord,
    OrderRecord,
    PositionStateRecord,
)

DEFAULT_DB_PATH = Path("data/trading_audit.db")
DEFAULT_FMP_DAILY_LIMIT = 250


class DuplicateOrderError(sqlite3.IntegrityError):
    """Raised when attempting to log an order with an existing client_order_id."""


def _ensure_ny_datetime(dt: datetime | None = None) -> datetime:
    """Return a timezone-aware datetime normalized to America/New_York."""
    if dt is None:
        return datetime.now(NY_TIMEZONE)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=NY_TIMEZONE)
    return dt.astimezone(NY_TIMEZONE)


def _parse_ny_datetime(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp string into an America/New_York datetime."""
    parsed = datetime.fromisoformat(raw)
    return _ensure_ny_datetime(parsed)


def _normalize_symbol(symbol: str) -> str:
    """Normalize and validate an equity/index ticker symbol."""
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must be a non-empty string")
    return cleaned


def _normalize_optional_symbol(symbol: str | None) -> str | None:
    """Normalize an optional ticker symbol when provided."""
    if symbol is None:
        return None
    return _normalize_symbol(symbol)


def _normalize_earnings_date(earnings_date: str | date | datetime | None) -> str | None:
    """Normalize earnings_date into an ISO YYYY-MM-DD string or None."""
    if earnings_date is None:
        return None
    if isinstance(earnings_date, datetime):
        return earnings_date.date().isoformat()
    if isinstance(earnings_date, date):
        return earnings_date.isoformat()
    cleaned = earnings_date.strip()
    if not cleaned:
        return None
    return date.fromisoformat(cleaned).isoformat()


def _validate_limit(limit: int) -> int:
    """Validate query row limits."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS guardian_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
        veto_reasons_json TEXT NOT NULL,
        vix REAL,
        recommended_shares REAL,
        recommended_notional REAL,
        metadata_json TEXT NOT NULL,
        decided_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_guardian_decisions_symbol_decided
    ON guardian_decisions (symbol, decided_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_guardian_decisions_approved_decided
    ON guardian_decisions (approved, decided_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS fmp_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        data_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        earnings_date TEXT,
        cache_date TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        UNIQUE (symbol, data_type)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fmp_cache_cache_date
    ON fmp_cache (cache_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fmp_cache_expires_at
    ON fmp_cache (expires_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS fmp_api_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint TEXT NOT NULL,
        symbol TEXT,
        request_date TEXT NOT NULL,
        requested_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fmp_api_usage_date
    ON fmp_api_usage (request_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fmp_api_usage_symbol_date
    ON fmp_api_usage (symbol, request_date)
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        consensus_count INTEGER NOT NULL,
        sources_json TEXT NOT NULL,
        rationale TEXT NOT NULL,
        passed_threshold INTEGER NOT NULL CHECK (passed_threshold IN (0, 1)),
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ai_signals_symbol_created
    ON ai_signals (symbol, created_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_order_id TEXT NOT NULL UNIQUE,
        broker_order_id TEXT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        qty REAL NOT NULL,
        limit_price REAL,
        stop_price REAL,
        status TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_symbol_status_created
    ON orders (symbol, status, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_status_created
    ON orders (status, created_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS position_state (
        symbol TEXT PRIMARY KEY,
        qty REAL NOT NULL,
        entry_price REAL NOT NULL,
        current_stop_price REAL NOT NULL,
        highest_price REAL NOT NULL,
        atr_14 REAL NOT NULL,
        opened_at TEXT NOT NULL,
        stop_order_id TEXT,
        client_order_id TEXT,
        updated_at TEXT NOT NULL
    )
    """,
)


class TradingAuditStore:
    """Deterministic SQLite audit store and FMP daily cache manager."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Initialize the store, create schema/indexes, and enable WAL for file DBs."""
        raw_path = str(db_path)
        self._is_memory = raw_path == ":memory:" or raw_path.startswith("file::memory:")
        self.db_path: str | Path = raw_path if self._is_memory else Path(db_path)
        self._memory_conn: sqlite3.Connection | None = None

        if not self._is_memory:
            assert isinstance(self.db_path, Path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_conn = self._open_connection()

        self._initialize_schema()

    def __enter__(self) -> TradingAuditStore:
        """Support usage as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close any held in-memory connection on context exit."""
        self.close()

    def close(self) -> None:
        """Close any persistent connection held by the store."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def _open_connection(self) -> sqlite3.Connection:
        """Open a configured SQLite connection with row mapping."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection inside an atomic commit/rollback transaction."""
        if self._is_memory:
            if self._memory_conn is None:
                self._memory_conn = self._open_connection()
            conn = self._memory_conn
            should_close = False
        else:
            conn = self._open_connection()
            should_close = True

        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            if should_close:
                conn.close()

    def _initialize_schema(self) -> None:
        """Create tables, indexes, and configure WAL mode idempotently."""
        if not self._is_memory:
            conn = self._open_connection()
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            finally:
                conn.close()

        with self.transaction() as conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)

    # ------------------------------------------------------------------
    # Guardian Decisions & Vetoes
    # ------------------------------------------------------------------

    def log_guardian_decision(
        self,
        symbol: str,
        action: str,
        approved: bool,
        veto_reasons: Sequence[str] | None = None,
        vix: float | None = None,
        recommended_shares: float | None = None,
        recommended_notional: float | None = None,
        metadata: dict[str, Any] | None = None,
        decided_at: datetime | None = None,
    ) -> int:
        """Persist a Guardian decision or veto and return the new record ID."""
        normalized_symbol = _normalize_symbol(symbol)
        normalized_action = action.strip().upper()
        if not normalized_action:
            raise ValueError("action must be a non-empty string")

        reasons_list = [str(reason) for reason in (veto_reasons or ())]
        metadata_dict = dict(metadata) if metadata is not None else {}
        timestamp_ny = _ensure_ny_datetime(decided_at)

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO guardian_decisions (
                    symbol,
                    action,
                    approved,
                    veto_reasons_json,
                    vix,
                    recommended_shares,
                    recommended_notional,
                    metadata_json,
                    decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_symbol,
                    normalized_action,
                    int(bool(approved)),
                    json.dumps(reasons_list),
                    vix,
                    recommended_shares,
                    recommended_notional,
                    json.dumps(metadata_dict),
                    timestamp_ny.isoformat(),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def get_guardian_decisions(
        self,
        symbol: str | None = None,
        approved: bool | None = None,
        limit: int = 100,
    ) -> list[GuardianDecisionRecord]:
        """Query Guardian decisions ordered newest-first with optional filters."""
        _validate_limit(limit)
        clauses: list[str] = []
        params: list[Any] = []

        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(_normalize_symbol(symbol))
        if approved is not None:
            clauses.append("approved = ?")
            params.append(int(bool(approved)))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id,
                    symbol,
                    action,
                    approved,
                    veto_reasons_json,
                    vix,
                    recommended_shares,
                    recommended_notional,
                    metadata_json,
                    decided_at
                FROM guardian_decisions
                {where_sql}
                ORDER BY decided_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            GuardianDecisionRecord(
                id=int(row["id"]),
                symbol=str(row["symbol"]),
                action=str(row["action"]),
                approved=bool(row["approved"]),
                veto_reasons=list(json.loads(row["veto_reasons_json"])),
                vix=float(row["vix"]) if row["vix"] is not None else None,
                recommended_shares=(
                    float(row["recommended_shares"])
                    if row["recommended_shares"] is not None
                    else None
                ),
                recommended_notional=(
                    float(row["recommended_notional"])
                    if row["recommended_notional"] is not None
                    else None
                ),
                metadata=dict(json.loads(row["metadata_json"])),
                decided_at=_parse_ny_datetime(row["decided_at"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # FMP Daily Cache & API Quota Tracking
    # ------------------------------------------------------------------

    def upsert_fmp_cache(
        self,
        symbol: str,
        data_type: str,
        payload: dict[str, Any] | list[Any],
        earnings_date: str | date | datetime | None = None,
        ttl_hours: float = 24,
        now: datetime | None = None,
    ) -> FmpCacheEntry:
        """Insert or update an FMP cache entry tied to the current America/New_York date."""
        if ttl_hours < 0:
            raise ValueError("ttl_hours must be non-negative")

        normalized_symbol = _normalize_symbol(symbol)
        normalized_type = data_type.strip().lower()
        if not normalized_type:
            raise ValueError("data_type must be a non-empty string")

        now_ny = _ensure_ny_datetime(now)
        expires_at = now_ny + timedelta(hours=ttl_hours)
        cache_date = now_ny.date().isoformat()
        normalized_earnings = _normalize_earnings_date(earnings_date)
        payload_json = json.dumps(payload)

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO fmp_cache (
                    symbol,
                    data_type,
                    payload_json,
                    earnings_date,
                    cache_date,
                    fetched_at,
                    expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, data_type) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    earnings_date = excluded.earnings_date,
                    cache_date = excluded.cache_date,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at
                """,
                (
                    normalized_symbol,
                    normalized_type,
                    payload_json,
                    normalized_earnings,
                    cache_date,
                    now_ny.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            row = conn.execute(
                """
                SELECT id, symbol, data_type, payload_json, earnings_date, cache_date, fetched_at, expires_at
                FROM fmp_cache
                WHERE symbol = ? AND data_type = ?
                """,
                (normalized_symbol, normalized_type),
            ).fetchone()

        assert row is not None
        return self._row_to_fmp_cache_entry(row)

    def get_fmp_cache(
        self,
        symbol: str,
        data_type: str,
        now: datetime | None = None,
    ) -> FmpCacheEntry | None:
        """Return a valid FMP cache entry, or None if missing, TTL-expired, or from a prior NY day."""
        normalized_symbol = _normalize_symbol(symbol)
        normalized_type = data_type.strip().lower()
        if not normalized_type:
            raise ValueError("data_type must be a non-empty string")

        now_ny = _ensure_ny_datetime(now)
        current_ny_date = now_ny.date().isoformat()

        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT id, symbol, data_type, payload_json, earnings_date, cache_date, fetched_at, expires_at
                FROM fmp_cache
                WHERE symbol = ? AND data_type = ?
                """,
                (normalized_symbol, normalized_type),
            ).fetchone()

        if row is None:
            return None

        entry = self._row_to_fmp_cache_entry(row)
        if now_ny >= entry.expires_at or entry.cache_date != current_ny_date:
            return None
        return entry

    def purge_expired_cache(self, now: datetime | None = None) -> int:
        """Delete all FMP cache entries that are TTL-expired or from a prior NY date."""
        now_ny = _ensure_ny_datetime(now)
        current_ny_date = now_ny.date().isoformat()

        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT id, cache_date, expires_at FROM fmp_cache"
            ).fetchall()

            expired_ids = [
                int(row["id"])
                for row in rows
                if row["cache_date"] != current_ny_date
                or now_ny >= _parse_ny_datetime(row["expires_at"])
            ]
            if not expired_ids:
                return 0

            placeholders = ", ".join("?" for _ in expired_ids)
            cursor = conn.execute(
                f"DELETE FROM fmp_cache WHERE id IN ({placeholders})",
                expired_ids,
            )
            return int(cursor.rowcount)

    def record_fmp_request(
        self,
        endpoint: str,
        symbol: str | None = None,
        now: datetime | None = None,
    ) -> int:
        """Log an outbound FMP API request for NY-day quota accounting and return record ID."""
        cleaned_endpoint = endpoint.strip()
        if not cleaned_endpoint:
            raise ValueError("endpoint must be a non-empty string")

        normalized_symbol = _normalize_optional_symbol(symbol)
        now_ny = _ensure_ny_datetime(now)
        request_date = now_ny.date().isoformat()

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO fmp_api_usage (endpoint, symbol, request_date, requested_at)
                VALUES (?, ?, ?, ?)
                """,
                (cleaned_endpoint, normalized_symbol, request_date, now_ny.isoformat()),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def get_fmp_requests_today(self, now: datetime | None = None) -> int:
        """Return total FMP API requests logged on the current America/New_York date."""
        now_ny = _ensure_ny_datetime(now)
        request_date = now_ny.date().isoformat()

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS request_count FROM fmp_api_usage WHERE request_date = ?",
                (request_date,),
            ).fetchone()

        assert row is not None
        return int(row["request_count"])

    def can_make_fmp_request(
        self,
        daily_limit: int = DEFAULT_FMP_DAILY_LIMIT,
        now: datetime | None = None,
    ) -> bool:
        """Return True if today's NY-date FMP request count is strictly below daily_limit."""
        if daily_limit <= 0:
            return False
        return self.get_fmp_requests_today(now=now) < daily_limit

    @staticmethod
    def _row_to_fmp_cache_entry(row: sqlite3.Row) -> FmpCacheEntry:
        """Convert a SQLite row into an FmpCacheEntry dataclass."""
        return FmpCacheEntry(
            id=int(row["id"]),
            symbol=str(row["symbol"]),
            data_type=str(row["data_type"]),
            payload=json.loads(row["payload_json"]),
            earnings_date=str(row["earnings_date"]) if row["earnings_date"] is not None else None,
            cache_date=str(row["cache_date"]),
            fetched_at=_parse_ny_datetime(row["fetched_at"]),
            expires_at=_parse_ny_datetime(row["expires_at"]),
        )

    # ------------------------------------------------------------------
    # AI Signals History
    # ------------------------------------------------------------------

    def log_ai_signal(
        self,
        symbol: str,
        sentiment: str,
        confidence_score: float,
        consensus_count: int,
        sources: Sequence[str] | None = None,
        rationale: str = "",
        passed_threshold: bool = False,
        created_at: datetime | None = None,
    ) -> int:
        """Persist an AI Analyst signal evaluation and return the new record ID."""
        normalized_symbol = _normalize_symbol(symbol)
        cleaned_sentiment = sentiment.strip().lower()
        if not cleaned_sentiment:
            raise ValueError("sentiment must be a non-empty string")

        sources_list = [str(src) for src in (sources or ())]
        timestamp_ny = _ensure_ny_datetime(created_at)

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ai_signals (
                    symbol,
                    sentiment,
                    confidence_score,
                    consensus_count,
                    sources_json,
                    rationale,
                    passed_threshold,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_symbol,
                    cleaned_sentiment,
                    float(confidence_score),
                    int(consensus_count),
                    json.dumps(sources_list),
                    rationale,
                    int(bool(passed_threshold)),
                    timestamp_ny.isoformat(),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def get_ai_signals(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[AiSignalRecord]:
        """Query AI signal records ordered newest-first with optional symbol filter."""
        _validate_limit(limit)
        clauses: list[str] = []
        params: list[Any] = []

        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(_normalize_symbol(symbol))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id,
                    symbol,
                    sentiment,
                    confidence_score,
                    consensus_count,
                    sources_json,
                    rationale,
                    passed_threshold,
                    created_at
                FROM ai_signals
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            AiSignalRecord(
                id=int(row["id"]),
                symbol=str(row["symbol"]),
                sentiment=str(row["sentiment"]),
                confidence_score=float(row["confidence_score"]),
                consensus_count=int(row["consensus_count"]),
                sources=list(json.loads(row["sources_json"])),
                rationale=str(row["rationale"]),
                passed_threshold=bool(row["passed_threshold"]),
                created_at=_parse_ny_datetime(row["created_at"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Orders History
    # ------------------------------------------------------------------

    def log_order(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
        limit_price: float | None = None,
        stop_price: float | None = None,
        status: str = "pending",
        broker_order_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> int:
        """Persist a new order record and return its ID; reject duplicate client_order_id."""
        cleaned_client_id = client_order_id.strip()
        if not cleaned_client_id:
            raise ValueError("client_order_id must be a non-empty string")

        normalized_symbol = _normalize_symbol(symbol)
        cleaned_side = side.strip().upper()
        cleaned_type = order_type.strip().lower()
        cleaned_status = status.strip().lower()
        if not cleaned_side or not cleaned_type or not cleaned_status:
            raise ValueError("side, order_type, and status must be non-empty strings")

        timestamp_ny = _ensure_ny_datetime(created_at)
        timestamp_iso = timestamp_ny.isoformat()
        metadata_dict = dict(metadata) if metadata is not None else {}

        try:
            with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO orders (
                        client_order_id,
                        broker_order_id,
                        symbol,
                        side,
                        order_type,
                        qty,
                        limit_price,
                        stop_price,
                        status,
                        metadata_json,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cleaned_client_id,
                        broker_order_id,
                        normalized_symbol,
                        cleaned_side,
                        cleaned_type,
                        float(qty),
                        limit_price,
                        stop_price,
                        cleaned_status,
                        json.dumps(metadata_dict),
                        timestamp_iso,
                        timestamp_iso,
                    ),
                )
                assert cursor.lastrowid is not None
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise DuplicateOrderError(
                f"Order with client_order_id '{cleaned_client_id}' already exists"
            ) from exc

    def update_order_status(
        self,
        client_order_id: str,
        status: str,
        broker_order_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> bool:
        """Update an order's status (and optional broker_order_id); return True if found."""
        cleaned_client_id = client_order_id.strip()
        cleaned_status = status.strip().lower()
        if not cleaned_client_id or not cleaned_status:
            raise ValueError("client_order_id and status must be non-empty strings")

        timestamp_ny = _ensure_ny_datetime(updated_at)

        with self.transaction() as conn:
            if broker_order_id is not None:
                cursor = conn.execute(
                    """
                    UPDATE orders
                    SET status = ?, broker_order_id = ?, updated_at = ?
                    WHERE client_order_id = ?
                    """,
                    (cleaned_status, broker_order_id, timestamp_ny.isoformat(), cleaned_client_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE orders
                    SET status = ?, updated_at = ?
                    WHERE client_order_id = ?
                    """,
                    (cleaned_status, timestamp_ny.isoformat(), cleaned_client_id),
                )
            return cursor.rowcount > 0

    def get_orders(
        self,
        symbol: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[OrderRecord]:
        """Query orders ordered newest-first with optional symbol and status filters."""
        _validate_limit(limit)
        clauses: list[str] = []
        params: list[Any] = []

        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(_normalize_symbol(symbol))
        if status is not None:
            clauses.append("status = ?")
            params.append(status.strip().lower())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self.transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    id,
                    client_order_id,
                    broker_order_id,
                    symbol,
                    side,
                    order_type,
                    qty,
                    limit_price,
                    stop_price,
                    status,
                    metadata_json,
                    created_at,
                    updated_at
                FROM orders
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            OrderRecord(
                id=int(row["id"]),
                client_order_id=str(row["client_order_id"]),
                broker_order_id=(
                    str(row["broker_order_id"]) if row["broker_order_id"] is not None else None
                ),
                symbol=str(row["symbol"]),
                side=str(row["side"]),
                order_type=str(row["order_type"]),
                qty=float(row["qty"]),
                limit_price=(
                    float(row["limit_price"]) if row["limit_price"] is not None else None
                ),
                stop_price=float(row["stop_price"]) if row["stop_price"] is not None else None,
                status=str(row["status"]),
                metadata=dict(json.loads(row["metadata_json"])),
                created_at=_parse_ny_datetime(row["created_at"]),
                updated_at=_parse_ny_datetime(row["updated_at"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Crash-Safe Open Position State
    # ------------------------------------------------------------------

    def upsert_position_state(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        current_stop_price: float,
        atr_14: float,
        opened_at: datetime | None = None,
        highest_price: float | None = None,
        stop_order_id: str | None = None,
        client_order_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> PositionStateRecord:
        """Insert or update crash-safe runtime state for an active open position."""
        normalized_symbol = _normalize_symbol(symbol)
        if qty <= 0 or entry_price <= 0 or current_stop_price < 0 or atr_14 < 0:
            raise ValueError("Invalid numeric values for position_state")

        opened_ny = _ensure_ny_datetime(opened_at)
        updated_ny = _ensure_ny_datetime(updated_at)
        peak = float(highest_price) if highest_price is not None else float(entry_price)

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO position_state (
                    symbol,
                    qty,
                    entry_price,
                    current_stop_price,
                    highest_price,
                    atr_14,
                    opened_at,
                    stop_order_id,
                    client_order_id,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty = excluded.qty,
                    entry_price = excluded.entry_price,
                    current_stop_price = excluded.current_stop_price,
                    highest_price = excluded.highest_price,
                    atr_14 = excluded.atr_14,
                    opened_at = excluded.opened_at,
                    stop_order_id = COALESCE(excluded.stop_order_id, position_state.stop_order_id),
                    client_order_id = COALESCE(excluded.client_order_id, position_state.client_order_id),
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_symbol,
                    float(qty),
                    float(entry_price),
                    float(current_stop_price),
                    peak,
                    float(atr_14),
                    opened_ny.isoformat(),
                    stop_order_id,
                    client_order_id,
                    updated_ny.isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM position_state WHERE symbol = ?",
                (normalized_symbol,),
            ).fetchone()

        assert row is not None
        return self._row_to_position_state(row)

    def get_position_state(self, symbol: str) -> PositionStateRecord | None:
        """Fetch persisted state for a single open position symbol if present."""
        normalized_symbol = _normalize_symbol(symbol)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM position_state WHERE symbol = ?",
                (normalized_symbol,),
            ).fetchone()
        return self._row_to_position_state(row) if row is not None else None

    def get_open_position_states(self) -> list[PositionStateRecord]:
        """Return all persisted open position states ordered by opened_at ascending."""
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM position_state ORDER BY opened_at ASC, symbol ASC"
            ).fetchall()
        return [self._row_to_position_state(row) for row in rows]

    def delete_position_state(self, symbol: str) -> bool:
        """Remove persisted open position state once closed; return True if deleted."""
        normalized_symbol = _normalize_symbol(symbol)
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM position_state WHERE symbol = ?",
                (normalized_symbol,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_position_state(row: sqlite3.Row) -> PositionStateRecord:
        return PositionStateRecord(
            symbol=str(row["symbol"]),
            qty=float(row["qty"]),
            entry_price=float(row["entry_price"]),
            current_stop_price=float(row["current_stop_price"]),
            highest_price=float(row["highest_price"]),
            atr_14=float(row["atr_14"]),
            opened_at=_parse_ny_datetime(row["opened_at"]),
            stop_order_id=(
                str(row["stop_order_id"]) if row["stop_order_id"] is not None else None
            ),
            client_order_id=(
                str(row["client_order_id"]) if row["client_order_id"] is not None else None
            ),
            updated_at=_parse_ny_datetime(row["updated_at"]),
        )


AuditDatabase = TradingAuditStore
