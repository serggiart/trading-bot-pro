"""Typed data models for local SQLite audit logging and FMP daily caching."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

NY_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class GuardianDecisionRecord:
    """Immutable audit record of an Algorithmic Guardian evaluation or veto."""

    id: int
    symbol: str
    action: str
    approved: bool
    veto_reasons: list[str]
    vix: float | None
    recommended_shares: float | None
    recommended_notional: float | None
    metadata: dict[str, Any]
    decided_at: datetime


@dataclass(frozen=True, slots=True)
class FmpCacheEntry:
    """Cached Financial Modeling Prep payload scoped to an America/New_York date."""

    id: int
    symbol: str
    data_type: str
    payload: dict[str, Any] | list[Any]
    earnings_date: str | None
    cache_date: str
    fetched_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AiSignalRecord:
    """Immutable audit record of an AI Analyst consensus & sentiment evaluation."""

    id: int
    symbol: str
    sentiment: str
    confidence_score: float
    consensus_count: int
    sources: list[str]
    rationale: str
    passed_threshold: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """Audit record tracking a limit order lifecycle from creation to terminal state."""

    id: int
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: str
    order_type: str
    qty: float
    limit_price: float | None
    stop_price: float | None
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(NY_TIMEZONE))
    updated_at: datetime = field(default_factory=lambda: datetime.now(NY_TIMEZONE))
