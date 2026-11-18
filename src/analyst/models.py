"""Typed Pydantic models and configuration for the AI Analyst module."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

NY_TIMEZONE = ZoneInfo("America/New_York")


def normalize_ny_datetime(value: datetime | int | float | str | None) -> datetime:
    """Convert an epoch timestamp, ISO string, or datetime into America/New_York."""
    if value is None:
        return datetime.now(NY_TIMEZONE)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("Timestamp must be a finite numeric epoch value")
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone(NY_TIMEZONE)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Timestamp string cannot be empty")
        if cleaned.endswith("Z"):
            cleaned = f"{cleaned[:-1]}+00:00"
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(NY_TIMEZONE)
    if isinstance(value, datetime):
        aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        return aware.astimezone(NY_TIMEZONE)
    raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")


def _validate_finite_float(value: float, field_name: str) -> float:
    """Reject NaN and infinite floating-point inputs."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


class AnalystReport(BaseModel):
    """Structured output schema returned by the Gemini AI Analyst."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(..., min_length=1)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    independent_sources_count: int = Field(..., ge=0)
    volume_confirmed: bool
    manipulation_risk: bool
    rationale: str = Field(..., min_length=1)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be a non-empty string")
        return normalized

    @field_validator("sentiment_score", "confidence_score")
    @classmethod
    def _check_finite_scores(cls, value: float, info: object) -> float:
        field_name = getattr(info, "field_name", "score")
        return _validate_finite_float(value, str(field_name))


class NewsItem(BaseModel):
    """Normalized news article collected from Finnhub or Alpaca News."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    headline: str = Field(..., min_length=1)
    summary: str = Field(default="")
    source: str = Field(..., min_length=1)
    url: str = Field(default="")
    published_at: datetime
    provider: Literal["finnhub", "alpaca"]

    @field_validator("published_at", mode="before")
    @classmethod
    def _normalize_published_at(cls, value: datetime | int | float | str) -> datetime:
        return normalize_ny_datetime(value)


class AnalystConfig(BaseModel):
    """Deterministic thresholds and model settings for GeminiAnalyst."""

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    min_independent_sources: int = Field(default=3, ge=1)
    min_relative_volume: float = Field(default=1.2, ge=0.0)
    model_name: str = Field(default="gemini-2.5-flash", min_length=1)
    lookback_hours: int = Field(default=48, ge=1)

    @field_validator("min_confidence", "min_relative_volume")
    @classmethod
    def _check_finite_thresholds(cls, value: float, info: object) -> float:
        field_name = getattr(info, "field_name", "threshold")
        return _validate_finite_float(value, str(field_name))


class AnalystEvaluationResult(BaseModel):
    """Final gated decision combining LLM report, volume verification, and SQLite audit ID."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(..., min_length=1)
    approved: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    report: AnalystReport
    relative_volume: float = Field(..., ge=0.0)
    news_count: int = Field(..., ge=0)
    sources: list[str] = Field(default_factory=list)
    signal_id: int | None = None
