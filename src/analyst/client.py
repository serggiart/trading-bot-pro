"""Gemini AI Analyst client with strict deterministic gating and SQLite audit logging."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from google import genai
from google.genai import types

from src.analyst.models import (
    AnalystConfig,
    AnalystEvaluationResult,
    AnalystReport,
    NewsItem,
)
from src.analyst.news import NewsAggregator
from src.storage import TradingAuditStore


def compute_relative_volume(current_volume: float, avg_volume_20d: float) -> float:
    """Compute current_volume / avg_volume_20d safely, returning 0.0 for invalid/<=0 inputs."""
    cur = float(current_volume)
    avg = float(avg_volume_20d)
    if not (math.isfinite(cur) and math.isfinite(avg)):
        return 0.0
    if cur <= 0.0 or avg <= 0.0:
        return 0.0
    return cur / avg


def map_sentiment_label(sentiment_score: float) -> Literal["bullish", "bearish", "neutral"]:
    """Map continuous sentiment score in [-1.0, 1.0] to a discrete SQLite audit label."""
    if sentiment_score > 0.10:
        return "bullish"
    if sentiment_score < -0.10:
        return "bearish"
    return "neutral"


def evaluate_report_gates(
    report: AnalystReport,
    relative_volume: float,
    config: AnalystConfig,
) -> list[str]:
    """Apply deterministic anti-desinformation and risk gates to an AI AnalystReport."""
    reasons: list[str] = []
    if report.confidence_score < config.min_confidence:
        reasons.append(
            f"confidence_score {report.confidence_score:.2f} < {config.min_confidence:.2f}"
        )
    if report.independent_sources_count < config.min_independent_sources:
        reasons.append(
            f"independent_sources_count {report.independent_sources_count} < "
            f"{config.min_independent_sources}"
        )
    if not report.volume_confirmed:
        reasons.append("volume_confirmed is False")
    if relative_volume < config.min_relative_volume:
        reasons.append(
            f"relative_volume {relative_volume:.2f} < {config.min_relative_volume:.2f} "
            "(price-volume divergence)"
        )
    if report.manipulation_risk:
        reasons.append("manipulation_risk flagged True")
    return reasons


def _build_analysis_prompt(
    symbol: str,
    current_volume: float,
    avg_volume_20d: float,
    relative_volume: float,
    news_items: Sequence[NewsItem],
    sources: Sequence[str],
    config: AnalystConfig,
) -> str:
    """Build a concise, deterministic prompt for Gemini structured evaluation."""
    if news_items:
        formatted_articles = "\n".join(
            (
                f"- [{item.published_at.isoformat()}] ({item.provider}/{item.source}) "
                f"{item.headline}"
                + (f" — {item.summary}" if item.summary else "")
            )
            for item in news_items
        )
    else:
        formatted_articles = "- No recent news articles found in the lookback window."

    sources_summary = ", ".join(sources) if sources else "None"
    return (
        f"Evaluate equity ticker {symbol} for trading signal quality and disinformation risk.\n"
        f"Market Volume Metrics:\n"
        f"- Current Volume: {current_volume:,.0f}\n"
        f"- 20-Day Average Volume: {avg_volume_20d:,.0f}\n"
        f"- Relative Volume: {relative_volume:.2f}x "
        f"(institutional confirmation threshold: {config.min_relative_volume:.2f}x)\n"
        f"- Total Deduplicated Articles: {len(news_items)}\n"
        f"- Distinct Sources ({len(sources)}): {sources_summary}\n\n"
        f"Recent News (America/New_York):\n"
        f"{formatted_articles}\n\n"
        f"Instructions:\n"
        f"1. Score sentiment_score in [-1.0, 1.0] and confidence_score in [0.0, 1.0].\n"
        f"2. Count independent_sources_count strictly as distinct non-syndicated sources.\n"
        f"3. Set volume_confirmed=true only if relative volume supports institutional participation.\n"
        f"4. Set manipulation_risk=true if there are signs of rumor, pump-and-dump, or "
        f"uncorroborated hype.\n"
        f"5. Provide a concise rationale."
    )


def _parse_analyst_report(response: Any) -> AnalystReport:
    """Extract and validate AnalystReport from a google-genai GenerateContentResponse."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, AnalystReport):
        return parsed
    if isinstance(parsed, dict):
        return AnalystReport.model_validate(parsed)

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return AnalystReport.model_validate_json(text)

    raise ValueError("Gemini response did not contain a parseable AnalystReport")


class GeminiAnalyst:
    """Probabilistic AI news/volume analyst wrapped in strict deterministic gates."""

    def __init__(
        self,
        audit_store: TradingAuditStore | None = None,
        news_aggregator: NewsAggregator | None = None,
        client: Any | None = None,
        config: AnalystConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize GeminiAnalyst with injectable store, news aggregator, and GenAI client."""
        self._audit_store = audit_store
        self.news_aggregator = news_aggregator or NewsAggregator()
        self.config = config or AnalystConfig()
        self._api_key = api_key

        if client is not None:
            self._client: Any | None = client
        else:
            resolved_key = api_key or os.getenv("GEMINI_API_KEY")
            self._client = genai.Client(api_key=resolved_key) if resolved_key else None

    @property
    def audit_store(self) -> TradingAuditStore:
        """Return the configured TradingAuditStore, initializing lazily if omitted."""
        if self._audit_store is None:
            self._audit_store = TradingAuditStore()
        return self._audit_store

    def _get_client(self) -> Any:
        """Return injected or lazily constructed google.genai.Client."""
        if self._client is None:
            resolved_key = self._api_key or os.getenv("GEMINI_API_KEY")
            self._client = (
                genai.Client(api_key=resolved_key) if resolved_key else genai.Client()
            )
        return self._client

    def evaluate(
        self,
        symbol: str,
        current_volume: float,
        avg_volume_20d: float,
        news_items: Sequence[NewsItem] | None = None,
        as_of: datetime | None = None,
    ) -> AnalystEvaluationResult:
        """Analyze recent news + relative volume via Gemini, gate deterministically, and log."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be a non-empty string")

        relative_volume = compute_relative_volume(current_volume, avg_volume_20d)
        if news_items is None:
            items = self.news_aggregator.fetch_news(
                normalized_symbol,
                lookback_hours=self.config.lookback_hours,
                as_of=as_of,
            )
        else:
            items = NewsAggregator.deduplicate(news_items)

        sources = NewsAggregator.distinct_sources(items)
        prompt = _build_analysis_prompt(
            symbol=normalized_symbol,
            current_volume=float(current_volume),
            avg_volume_20d=float(avg_volume_20d),
            relative_volume=relative_volume,
            news_items=items,
            sources=sources,
            config=self.config,
        )

        client = self._get_client()
        response = client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalystReport,
                temperature=0.1,
            ),
        )
        report = _parse_analyst_report(response)

        rejection_reasons = evaluate_report_gates(
            report=report,
            relative_volume=relative_volume,
            config=self.config,
        )
        approved = len(rejection_reasons) == 0

        signal_id = self.audit_store.log_ai_signal(
            symbol=normalized_symbol,
            sentiment=map_sentiment_label(report.sentiment_score),
            confidence_score=report.confidence_score,
            consensus_count=report.independent_sources_count,
            sources=sources,
            rationale=report.rationale,
            passed_threshold=approved,
        )

        return AnalystEvaluationResult(
            ticker=normalized_symbol,
            approved=approved,
            rejection_reasons=rejection_reasons,
            report=report,
            relative_volume=relative_volume,
            news_count=len(items),
            sources=sources,
            signal_id=signal_id,
        )
