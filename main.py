"""Main orchestration pipeline and CLI entrypoint for the algorithmic trading bot."""

from __future__ import annotations

import argparse
import fcntl
import inspect
import logging
import math
import signal
import threading
import time as time_module
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.analyst import (
    AnalystConfig,
    AnalystEvaluationResult,
    GeminiAnalyst,
    NewsAggregator,
    NewsItem,
    compute_relative_volume,
)
from src.config import StrategySettings, load_fed_dates, load_strategy_settings
from src.guardian import (
    GuardianDecision,
    MarketContext,
    PortfolioPosition,
    RiskGuardian,
    TradeCandidate,
    check_economic_calendar,
)
from src.notifications import CompositeNotifier, EmailNotifier, TelegramNotifier
from src.storage import DEFAULT_DB_PATH, TradingAuditStore
from src.universe import (
    FmpClient,
    MarketDataClient,
    UniverseConfig,
    UniverseScreener,
    UniverseScreeningResult,
    compute_return_correlation_matrix,
    select_top_candidates,
)

if TYPE_CHECKING:
    from src.execution import (
        AlpacaExecutionClient,
        ExecutionOrderResult,
        ExitEvaluationResult,
        ManagedPosition,
        PositionManager,
    )

logger = logging.getLogger(__name__)

NY_TIMEZONE = ZoneInfo("America/New_York")
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)


class ProcessLockError(RuntimeError):
    """Raised when another instance of the trading bot already holds the runtime lock."""


class ProcessLock:
    """Non-blocking advisory file lock preventing concurrent bot instances."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._handle: Any | None = None

    def __enter__(self) -> ProcessLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ProcessLockError(
                f"Another trading bot instance is already running (lock: {self.lock_path})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os_getpid()}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


def os_getpid() -> int:
    """Return current process ID (indirected cleanly for clarity)."""
    import os

    return os.getpid()


def normalize_ny_datetime(now: datetime | None = None) -> datetime:
    """Normalize a datetime to America/New_York.

    Naive datetimes are interpreted as America/New_York wall-clock time;
    timezone-aware datetimes are converted to America/New_York.
    """
    if now is None:
        return datetime.now(NY_TIMEZONE)
    if now.tzinfo is None:
        return now.replace(tzinfo=NY_TIMEZONE)
    return now.astimezone(NY_TIMEZONE)


def is_market_open(now: datetime | None = None) -> bool:
    """Return True during regular US equity market hours (Mon-Fri 09:30 <= t < 16:00 ET)."""
    ny_now = normalize_ny_datetime(now)
    if ny_now.weekday() >= 5:
        return False
    current_time = ny_now.time()
    return MARKET_OPEN_TIME <= current_time < MARKET_CLOSE_TIME


def estimate_intraday_run_rate_volume(
    latest_volume: float,
    now: datetime | None = None,
    min_session_fraction: float = 0.05,
) -> float:
    """Scale cumulative intraday volume to an estimated full-day run-rate during NY hours."""
    if latest_volume <= 0.0:
        return 0.0
    ny_now = normalize_ny_datetime(now)
    if not is_market_open(ny_now):
        return float(latest_volume)
    market_open_dt = datetime.combine(ny_now.date(), MARKET_OPEN_TIME, tzinfo=NY_TIMEZONE)
    elapsed_minutes = (ny_now - market_open_dt).total_seconds() / 60.0
    session_fraction = max(elapsed_minutes / 390.0, min_session_fraction)
    return float(latest_volume) / min(session_fraction, 1.0)


def check_news_confirmation_window(
    news_items: Sequence[NewsItem],
    now: datetime | None = None,
    min_delay_minutes: float = 2.0,
    max_delay_minutes: float = 5.0,
) -> tuple[bool, str]:
    """Enforce the 2-5 minute anti-disinformation delay after recent breaking news.

    If the latest news item is younger than ``min_delay_minutes``, execution is deferred
    to prevent reacting to unconfirmed spikes or disinformation.
    """
    if min_delay_minutes < 0.0:
        raise ValueError("min_delay_minutes must be non-negative")
    if max_delay_minutes < min_delay_minutes:
        raise ValueError("max_delay_minutes must be >= min_delay_minutes")

    ny_now = normalize_ny_datetime(now)
    if not news_items:
        return True, "No recent news items requiring confirmation delay"

    latest_published = max(normalize_ny_datetime(item.published_at) for item in news_items)
    age_minutes = (ny_now - latest_published).total_seconds() / 60.0

    if age_minutes < min_delay_minutes:
        return (
            False,
            f"News confirmation window active: latest news is {age_minutes:.2f} min old "
            f"(< {min_delay_minutes:.2f} min delay; {max_delay_minutes:.2f} min window)",
        )

    return (
        True,
        f"News confirmation window satisfied: latest news is {age_minutes:.2f} min old "
        f"(>= {min_delay_minutes:.2f} min)",
    )


def _parse_earnings_dates(raw_dates: Sequence[str], now_ny: datetime) -> list[datetime]:
    """Convert YYYY-MM-DD or ISO timestamp earnings strings to NY-aware datetimes."""
    parsed_dates: list[datetime] = []
    for raw in raw_dates:
        cleaned = str(raw).strip()
        if not cleaned:
            continue
        try:
            if "T" in cleaned or " " in cleaned:
                iso_str = f"{cleaned[:-1]}+00:00" if cleaned.endswith("Z") else cleaned
                parsed_dates.append(normalize_ny_datetime(datetime.fromisoformat(iso_str)))
            else:
                earnings_day = date.fromisoformat(cleaned[:10])
                if earnings_day == now_ny.date():
                    # Same-day earnings are always treated as active blackout.
                    parsed_dates.append(now_ny)
                else:
                    parsed_dates.append(
                        datetime.combine(earnings_day, time(0, 0), tzinfo=NY_TIMEZONE)
                    )
        except ValueError:
            logger.warning("Ignoring unparseable earnings date %r", raw)
    return parsed_dates


@dataclass(frozen=True, slots=True)
class TickerPipelineOutcome:
    """Structured outcome for a single ticker processed through the 5-level funnel."""

    symbol: str
    stage_reached: str
    status: str
    approved: bool
    reasons: list[str] = field(default_factory=list)
    screening_result: UniverseScreeningResult | None = None
    analyst_result: AnalystEvaluationResult | None = None
    guardian_decision: GuardianDecision | None = None
    order_result: ExecutionOrderResult | None = None


@dataclass(frozen=True, slots=True)
class PipelineCycleReport:
    """Structured report returned by TradingBotOrchestrator.run_cycle."""

    timestamp: datetime
    market_open: bool
    skipped: bool = False
    skip_reason: str | None = None
    dry_run: bool = False
    exit_results: list[ExitEvaluationResult] = field(default_factory=list)
    ticker_outcomes: list[TickerPipelineOutcome] = field(default_factory=list)


class TradingBotOrchestrator:
    """Coordinates the 5-stage decision funnel and open-position exit management."""

    def __init__(
        self,
        screener: UniverseScreener | None = None,
        news_aggregator: NewsAggregator | None = None,
        analyst: GeminiAnalyst | None = None,
        guardian: RiskGuardian | None = None,
        execution_client: AlpacaExecutionClient | None = None,
        position_manager: PositionManager | None = None,
        audit_store: TradingAuditStore | None = None,
        notifier: TelegramNotifier | None = None,
        confirmation_delay_minutes: float = 2.0,
        max_confirmation_delay_minutes: float = 5.0,
        default_sector: str = "Technology",
        sector_map: Mapping[str, str] | None = None,
        take_profit_atr_multiplier: float = 2.0,
        max_candidates_for_ai: int = 5,
        default_fed_dates: Sequence[datetime] = (),
        candle_store: Any | None = None,
        skip_ai_on_prefilter_fail: bool = False,
        max_stop_loss_pct: float = 0.0,
        max_ema_extension_atr: float = 0.0,
        allow_fractional_shares: bool = False,
    ) -> None:
        self.audit_store = audit_store or TradingAuditStore()
        self.guardian = guardian or RiskGuardian()
        self.notifier = notifier or TelegramNotifier()

        self._screener = screener
        self._news_aggregator = news_aggregator
        self._analyst = analyst
        self._execution_client = execution_client
        self._position_manager = position_manager
        self._candle_store = candle_store

        self.confirmation_delay_minutes = float(confirmation_delay_minutes)
        self.max_confirmation_delay_minutes = float(max_confirmation_delay_minutes)
        self.default_sector = default_sector
        self.sector_map: dict[str, str] = {
            k.strip().upper(): v for k, v in (sector_map or {}).items()
        }
        self.take_profit_atr_multiplier = float(take_profit_atr_multiplier)
        self.max_candidates_for_ai = int(max_candidates_for_ai)
        self.default_fed_dates: tuple[datetime, ...] = tuple(default_fed_dates)
        self.skip_ai_on_prefilter_fail = bool(skip_ai_on_prefilter_fail)
        self.max_stop_loss_pct = float(max_stop_loss_pct)
        self.max_ema_extension_atr = float(max_ema_extension_atr)
        self.allow_fractional_shares = bool(allow_fractional_shares)
        self.stop_reentry_cooldown_hours: float = 24.0
        self._stop_loss_cooldowns: dict[str, datetime] = {}
        self.ai_cache_ttl_minutes: float = 60.0
        self._ai_signal_cache: dict[str, tuple[datetime, AnalystEvaluationResult]] = {}

    @property
    def screener(self) -> UniverseScreener:
        """Lazily construct the Stage-1 UniverseScreener if not injected."""
        if self._screener is None:
            self._screener = UniverseScreener(
                market_data=MarketDataClient(),
                fmp_client=FmpClient(store=self.audit_store),
            )
        return self._screener

    @property
    def news_aggregator(self) -> NewsAggregator:
        """Lazily construct NewsAggregator if not injected."""
        if self._news_aggregator is None:
            self._news_aggregator = NewsAggregator()
        return self._news_aggregator

    @property
    def analyst(self) -> GeminiAnalyst:
        """Lazily construct GeminiAnalyst if not injected."""
        if self._analyst is None:
            self._analyst = GeminiAnalyst(
                audit_store=self.audit_store,
                news_aggregator=self.news_aggregator,
            )
        return self._analyst

    @property
    def execution_client(self) -> AlpacaExecutionClient:
        """Lazily construct AlpacaExecutionClient if not injected."""
        if self._execution_client is None:
            from src.execution import AlpacaExecutionClient

            self._execution_client = AlpacaExecutionClient(audit_store=self.audit_store)
        return self._execution_client

    @property
    def position_manager(self) -> PositionManager:
        """Lazily construct PositionManager if not injected."""
        if self._position_manager is None:
            from src.execution import PositionManager

            self._position_manager = PositionManager(
                execution_client=self.execution_client,
                audit_store=self.audit_store,
            )
        return self._position_manager

    def _manage_open_positions(
        self,
        open_positions: Sequence[ManagedPosition],
        now_ny: datetime,
        dry_run: bool,
        current_prices: Mapping[str, float] | None,
        bid_prices: Mapping[str, float] | None,
    ) -> list[ExitEvaluationResult]:
        """Evaluate trailing stops, time exits, and emergency exits for open positions."""
        exit_results: list[ExitEvaluationResult] = []
        price_lookup = {k.strip().upper(): float(v) for k, v in (current_prices or {}).items()}
        bid_lookup = {k.strip().upper(): float(v) for k, v in (bid_prices or {}).items()}

        batched_quotes: dict[str, Any] = {}
        missing_syms = [
            pos.symbol.strip().upper()
            for pos in open_positions
            if pos.symbol.strip().upper() not in price_lookup
        ]
        if len(missing_syms) > 1 and type(self.screener.market_data) is MarketDataClient:
            try:
                batched_quotes = self.screener.market_data.get_quotes_batch(missing_syms)
            except Exception as exc:
                logger.warning("Batched quote fetch for open positions failed (%s); falling back", exc)

        for position in open_positions:
            symbol = position.symbol.strip().upper()
            try:
                current_price = price_lookup.get(symbol)
                bid_price = bid_lookup.get(symbol)

                if current_price is None:
                    quote = batched_quotes.get(symbol) or self.screener.market_data.get_quote(symbol)
                    current_price = quote.price
                    if bid_price is None and quote.bid_price > 0.0:
                        bid_price = quote.bid_price

                if bid_price is None:
                    bid_price = current_price

                result = self.position_manager.evaluate_and_manage(
                    position,
                    current_price=current_price,
                    bid_price=bid_price,
                    now=now_ny,
                    dry_run=dry_run,
                )
                exit_results.append(result)

                if result.should_exit:
                    limit_price = (
                        result.order_result.limit_price
                        if result.order_result is not None
                        else bid_price
                    )
                    if (
                        result.action in {"TRAILING_STOP_EXIT", "EMERGENCY_EXIT"}
                        and limit_price is not None
                        and limit_price <= position.entry_price
                    ):
                        self._stop_loss_cooldowns[symbol] = now_ny + timedelta(
                            hours=self.stop_reentry_cooldown_hours
                        )
                    close_fn = self.notifier.notify_position_closed
                    try:
                        sig = inspect.signature(close_fn)
                        if "entry_price" in sig.parameters:
                            close_fn(
                                symbol=symbol,
                                action=result.action,
                                qty=position.qty,
                                limit_price=limit_price,
                                reason=result.exit_reason or result.action,
                                entry_price=position.entry_price,
                                dry_run=dry_run,
                            )
                        else:
                            close_fn(
                                symbol=symbol,
                                action=result.action,
                                qty=position.qty,
                                limit_price=limit_price,
                                reason=result.exit_reason or result.action,
                            )
                    except (TypeError, ValueError):
                        close_fn(
                            symbol=symbol,
                            action=result.action,
                            qty=position.qty,
                            limit_price=limit_price,
                            reason=result.exit_reason or result.action,
                        )
            except Exception as exc:
                logger.exception("Error managing open position %s: %s", symbol, exc)
                if hasattr(self.notifier, "send_message"):
                    self.notifier.send_message(
                        f"POSITION MANAGEMENT ERROR ({symbol}): {exc}"
                    )

        return exit_results

    def _notify_veto_and_circuit_breaker(
        self,
        symbol: str,
        decision: GuardianDecision,
        vix: float | None,
        starting_equity: float,
        daily_pnl: float,
    ) -> None:
        """Dispatch Telegram notifications for RiskGuardian vetoes and circuit breakers."""
        for rule in decision.rule_results:
            if rule.rule_name == "circuit_breaker" and not rule.passed:
                drawdown_pct = float(
                    rule.details.get(
                        "drawdown_pct",
                        max(0.0, -daily_pnl / starting_equity) if starting_equity > 0 else 0.0,
                    )
                )
                self.notifier.notify_circuit_breaker(
                    drawdown_pct=drawdown_pct,
                    starting_equity=starting_equity,
                    daily_pnl=daily_pnl,
                )
                break

        self.notifier.notify_guardian_veto(
            symbol=symbol,
            veto_reasons=decision.veto_reasons,
            vix=vix,
        )

    def _process_symbol(
        self,
        symbol: str,
        now_ny: datetime,
        shared_vix: float | None,
        dry_run: bool,
        portfolio_positions: Sequence[PortfolioPosition],
        correlation_matrix: Mapping[tuple[str, str], float] | Mapping[str, float],
        starting_equity: float,
        daily_pnl: float,
        fed_dates: Sequence[datetime],
        current_prices: Mapping[str, float],
        current_volumes: Mapping[str, float],
        avg_volumes_20d: Mapping[str, float],
        pre_screened: UniverseScreeningResult | None = None,
        available_capital: float | None = None,
    ) -> TickerPipelineOutcome:
        """Run a single symbol sequentially through stages 1-5 of the decision funnel."""
        # Stage 1: Universe screening
        if pre_screened is not None:
            screening = pre_screened
        elif shared_vix is not None:
            screening = self.screener.screen_symbol(symbol, now=now_ny, vix=shared_vix)
        else:
            screening = self.screener.screen_symbol(symbol, now=now_ny)

        if not screening.passed:
            return TickerPipelineOutcome(
                symbol=symbol,
                stage_reached="universe",
                status="rejected_universe",
                approved=False,
                reasons=list(screening.rejection_reasons),
                screening_result=screening,
            )

        quote = screening.quote
        indicators = screening.indicators
        fundamentals = screening.fundamentals

        if indicators is None:
            return TickerPipelineOutcome(
                symbol=symbol,
                stage_reached="universe",
                status="rejected_universe",
                approved=False,
                reasons=["Missing technical indicators from UniverseScreener"],
                screening_result=screening,
            )

        entry_price = current_prices.get(
            symbol,
            quote.price if quote is not None and quote.price > 0.0 else indicators.current_price,
        )
        ask_price = (
            quote.ask_price
            if quote is not None and quote.ask_price > 0.0
            else entry_price
        )
        ind_latest_vol = float(getattr(indicators, "latest_volume", 0.0) or 0.0)
        ind_avg_vol_20d = float(getattr(indicators, "avg_volume_20d", 0.0) or 0.0)

        if symbol in current_volumes:
            cur_vol = current_volumes[symbol]
        elif ind_latest_vol > 0.0 and ind_avg_vol_20d > 0.0:
            cur_vol = estimate_intraday_run_rate_volume(ind_latest_vol, now=now_ny)
        else:
            cur_vol = quote.volume if quote is not None else 0.0

        if symbol in avg_volumes_20d:
            avg_vol = avg_volumes_20d[symbol]
        elif ind_avg_vol_20d > 0.0:
            avg_vol = ind_avg_vol_20d
        else:
            avg_vol = quote.volume if quote is not None and quote.volume > 0.0 else cur_vol

        # Stage 2: News aggregation & 2-5 minute confirmation window
        analyst_config = getattr(self.analyst, "config", None)
        lookback_hours = int(getattr(analyst_config, "lookback_hours", 48))
        news_items = self.news_aggregator.fetch_news(
            symbol,
            lookback_hours=lookback_hours,
            as_of=now_ny,
        )
        window_ok, window_reason = check_news_confirmation_window(
            news_items=news_items,
            now=now_ny,
            min_delay_minutes=self.confirmation_delay_minutes,
            max_delay_minutes=self.max_confirmation_delay_minutes,
        )
        if not window_ok:
            return TickerPipelineOutcome(
                symbol=symbol,
                stage_reached="news_confirmation",
                status="deferred_news_window",
                approved=False,
                reasons=[window_reason],
                screening_result=screening,
            )

        # Deterministic pre-LLM quota guard: skip Gemini call if volume or source consensus
        # already fails hard anti-disinformation thresholds.
        if self.skip_ai_on_prefilter_fail and isinstance(analyst_config, AnalystConfig):
            prefilter_reasons: list[str] = []
            if available_capital is not None and ask_price > 0.0:
                if self.allow_fractional_shares and available_capital < 10.0:
                    return TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="cash_prefilter",
                        status="skipped_insufficient_cash",
                        approved=False,
                        reasons=[
                            f"Insufficient cash (${available_capital:,.2f} < $10.00 minimum fractional order; skipped LLM call)"
                        ],
                        screening_result=screening,
                    )
                if not self.allow_fractional_shares and available_capital < ask_price:
                    return TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="cash_prefilter",
                        status="skipped_insufficient_cash",
                        approved=False,
                        reasons=[
                            f"Insufficient cash (${available_capital:,.2f}) for 1 share of {symbol} at ${ask_price:,.2f} (skipped LLM call)"
                        ],
                        screening_result=screening,
                    )
            if (
                self.max_ema_extension_atr > 0.0
                and indicators.ema_20 > 0.0
                and indicators.atr_14 > 0.0
            ):
                ext_atr = (ask_price - indicators.ema_20) / indicators.atr_14
                if ext_atr > self.max_ema_extension_atr:
                    prefilter_reasons.append(
                        f"ema_extension_atr {ext_atr:.2f} > {self.max_ema_extension_atr:.2f} "
                        "(stretched near local peak; waiting for trough pullback)"
                    )
            rel_vol = compute_relative_volume(cur_vol, avg_vol)
            if rel_vol < analyst_config.min_relative_volume:
                prefilter_reasons.append(
                    f"relative_volume {rel_vol:.2f} < {analyst_config.min_relative_volume:.2f} "
                    "(price-volume divergence; skipped LLM call)"
                )
            distinct_sources = NewsAggregator.distinct_sources(news_items)
            if len(distinct_sources) < analyst_config.min_independent_sources:
                prefilter_reasons.append(
                    f"independent_sources_count {len(distinct_sources)} < "
                    f"{analyst_config.min_independent_sources} (skipped LLM call)"
                )
            if prefilter_reasons:
                return TickerPipelineOutcome(
                    symbol=symbol,
                    stage_reached="analyst_prefilter",
                    status="rejected_analyst_prefilter",
                    approved=False,
                    reasons=prefilter_reasons,
                    screening_result=screening,
                )

        # Stage 3: AI Analyst evaluation (with 60-minute intraday cache in live prefilter mode)
        cached_ai = (
            self._ai_signal_cache.get(symbol)
            if self.skip_ai_on_prefilter_fail and self.ai_cache_ttl_minutes > 0.0
            else None
        )
        if cached_ai is not None and (now_ny - cached_ai[0]) < timedelta(
            minutes=self.ai_cache_ttl_minutes
        ):
            analyst_result = cached_ai[1]
        else:
            analyst_result = self.analyst.evaluate(
                symbol=symbol,
                current_volume=cur_vol,
                avg_volume_20d=avg_vol,
                news_items=news_items,
                as_of=now_ny,
            )
            if self.skip_ai_on_prefilter_fail and self.ai_cache_ttl_minutes > 0.0:
                self._ai_signal_cache[symbol] = (now_ny, analyst_result)
        if not analyst_result.approved:
            return TickerPipelineOutcome(
                symbol=symbol,
                stage_reached="analyst",
                status="rejected_analyst",
                approved=False,
                reasons=list(analyst_result.rejection_reasons),
                screening_result=screening,
                analyst_result=analyst_result,
            )

        # Stage 4: Deterministic RiskGuardian & Capital Sizing + SQLite Audit
        earnings_dts = _parse_earnings_dates(
            fundamentals.earnings_dates if fundamentals is not None else (),
            now_ny=now_ny,
        )
        sector = self.sector_map.get(symbol, self.default_sector)
        candidate = TradeCandidate(
            symbol=symbol,
            sector=sector,
            entry_price=ask_price,
            atr=indicators.atr_14,
            earnings_dates=earnings_dts,
            side="BUY",
        )
        scale = 1000.0 if self.allow_fractional_shares else 1.0
        scaled_portfolio = (
            [
                PortfolioPosition(
                    symbol=pos.symbol,
                    sector=pos.sector,
                    market_value=pos.effective_market_value * scale,
                    shares=float(pos.shares) * scale,
                    avg_price=pos.avg_price,
                )
                for pos in portfolio_positions
            ]
            if self.allow_fractional_shares
            else portfolio_positions
        )
        context = MarketContext(
            now=now_ny,
            vix=indicators.vix,
            starting_equity=starting_equity * scale,
            daily_pnl=daily_pnl * scale,
            portfolio_positions=scaled_portfolio,
            correlation_matrix=correlation_matrix,
            fed_dates=fed_dates,
            earnings_dates=earnings_dts,
        )
        decision = self.guardian.evaluate(candidate, context)

        resolved_shares: float = (
            round(decision.recommended_shares / scale, 4)
            if self.allow_fractional_shares
            else float(decision.recommended_shares)
        )
        recommended_notional = (
            (decision.position_size.position_value / scale)
            if decision.position_size is not None
            else None
        )
        self.audit_store.log_guardian_decision(
            symbol=symbol,
            action="BUY",
            approved=decision.approved,
            veto_reasons=decision.veto_reasons,
            vix=indicators.vix,
            recommended_shares=float(resolved_shares),
            recommended_notional=recommended_notional,
            metadata={
                "dry_run": dry_run,
                "signal_id": analyst_result.signal_id,
                "confidence_score": analyst_result.report.confidence_score,
                "sentiment_score": analyst_result.report.sentiment_score,
            },
            decided_at=now_ny,
        )

        if not decision.approved:
            self._notify_veto_and_circuit_breaker(
                symbol=symbol,
                decision=decision,
                vix=indicators.vix,
                starting_equity=starting_equity,
                daily_pnl=daily_pnl,
            )
            return TickerPipelineOutcome(
                symbol=symbol,
                stage_reached="guardian",
                status="vetoed_guardian",
                approved=False,
                reasons=list(decision.veto_reasons),
                screening_result=screening,
                analyst_result=analyst_result,
                guardian_decision=decision,
            )

        # Stage 5: Limit order execution & Telegram trade alert
        stop_loss_price = (
            decision.position_size.stop_price
            if decision.position_size is not None
            else max(0.01, ask_price - self.guardian.config.atr_multiplier * indicators.atr_14)
        )
        if self.max_stop_loss_pct > 0.0:
            max_stop_floor = round(ask_price * (1.0 - self.max_stop_loss_pct), 2)
            stop_loss_price = max(stop_loss_price, max_stop_floor)
        take_profit_price: float | None = None
        if self.take_profit_atr_multiplier > 0.0 and indicators.atr_14 > 0.0:
            take_profit_price = round(
                ask_price + self.take_profit_atr_multiplier * indicators.atr_14,
                2,
            )
        if self.allow_fractional_shares:
            order_shares: float = resolved_shares
            if available_capital is not None:
                affordable = (
                    round(max(0.0, (available_capital * 0.995) / ask_price), 4)
                    if ask_price > 0.0
                    else 0.0
                )
                order_shares = min(order_shares, affordable)
            if order_shares <= 0.0 or order_shares * ask_price < 10.0:
                return TickerPipelineOutcome(
                    symbol=symbol,
                    stage_reached="cash_prefilter",
                    status="skipped_insufficient_cash",
                    approved=False,
                    reasons=[
                        f"Insufficient cash (${available_capital or 0.0:,.2f}) for fractional {symbol} at ${ask_price:,.2f} (< $10.00 min)"
                    ],
                    screening_result=screening,
                    analyst_result=analyst_result,
                    guardian_decision=decision,
                )
        else:
            order_shares = decision.recommended_shares
            if available_capital is not None:
                affordable = math.floor(max(0.0, available_capital) / ask_price) if ask_price > 0.0 else 0
                order_shares = min(order_shares, affordable)
                if order_shares <= 0:
                    return TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="cash_prefilter",
                        status="skipped_insufficient_cash",
                        approved=False,
                        reasons=[
                            f"Insufficient cash (${available_capital:,.2f}) for {symbol} at ${ask_price:,.2f} (margin disabled)"
                        ],
                        screening_result=screening,
                        analyst_result=analyst_result,
                        guardian_decision=decision,
                    )

        order_result = self.execution_client.submit_limit_buy(
            symbol=symbol,
            qty=order_shares,
            ask_price=ask_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            dry_run=dry_run,
            metadata={"atr_14": indicators.atr_14},
        )

        executed = bool(order_result.submitted) or (
            dry_run and str(getattr(order_result, "status", "")).lower() == "dry_run"
        )
        if executed:
            self.notifier.notify_trade_executed(
                symbol=symbol,
                side=order_result.side,
                qty=order_result.qty,
                limit_price=order_result.limit_price,
                stop_price=order_result.stop_price,
                dry_run=dry_run,
            )
            return TickerPipelineOutcome(
                symbol=symbol,
                stage_reached="execution",
                status="dry_run_executed" if dry_run else "executed",
                approved=True,
                reasons=[order_result.reason] if order_result.reason else [],
                screening_result=screening,
                analyst_result=analyst_result,
                guardian_decision=decision,
                order_result=order_result,
            )

        return TickerPipelineOutcome(
            symbol=symbol,
            stage_reached="execution",
            status="rejected_execution",
            approved=False,
            reasons=[order_result.reason] if order_result.reason else ["Order not submitted"],
            screening_result=screening,
            analyst_result=analyst_result,
            guardian_decision=decision,
            order_result=order_result,
        )

    def _sync_broker_account_state(
        self,
        starting_equity: float,
        daily_pnl: float,
        portfolio_positions: Sequence[PortfolioPosition],
    ) -> tuple[float, float, list[PortfolioPosition], float | None]:
        """Sync live/paper equity, daily PnL, open positions, and available cash from the broker."""
        snapshot = self.execution_client.get_account_snapshot()
        resolved_equity = starting_equity
        resolved_pnl = daily_pnl
        available_cash: float | None = None
        if snapshot is not None:
            raw_last_equity = getattr(snapshot, "last_equity", None)
            raw_equity = getattr(snapshot, "equity", None)
            raw_daily_pnl = getattr(snapshot, "daily_pnl", None)
            raw_cash = getattr(snapshot, "cash", None)
            if isinstance(raw_last_equity, (int, float)) and raw_last_equity > 0.0:
                resolved_equity = float(raw_last_equity)
            elif isinstance(raw_equity, (int, float)) and raw_equity > 0.0:
                resolved_equity = float(raw_equity)
            if isinstance(raw_daily_pnl, (int, float)):
                resolved_pnl = float(raw_daily_pnl)
            if isinstance(raw_cash, (int, float)) and math.isfinite(float(raw_cash)):
                available_cash = float(raw_cash)

        resolved_positions = list(portfolio_positions)
        synced_positions = self.execution_client.get_open_positions()
        if isinstance(synced_positions, Sequence):
            if synced_positions:
                resolved_positions = [
                    PortfolioPosition(
                        symbol=pos.symbol,
                        sector=pos.sector or self.sector_map.get(pos.symbol.strip().upper(), self.default_sector),
                        market_value=pos.market_value,
                        shares=pos.shares,
                        avg_price=pos.avg_price,
                    )
                    for pos in synced_positions
                ]
            elif snapshot is not None:
                resolved_positions = []

        return resolved_equity, resolved_pnl, resolved_positions, available_cash

    def _resolve_correlation_matrix(
        self,
        candidate_symbols: Sequence[str],
        portfolio_positions: Sequence[PortfolioPosition],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise return correlations from local cached candles if holdings exist."""
        if not portfolio_positions or not candidate_symbols:
            return {}

        store = self._candle_store
        if store is None:
            try:
                from src.backtest import ParquetCandleStore

                store = ParquetCandleStore()
            except Exception:
                return {}

        all_symbols = sorted(
            {s.strip().upper() for s in candidate_symbols if s.strip()}
            | {p.symbol.strip().upper() for p in portfolio_positions if p.symbol.strip()}
        )
        histories: dict[str, Any] = {}
        for sym in all_symbols:
            try:
                df = store.load_candles(sym)
                if df is not None and not df.empty:
                    histories[sym] = df
            except Exception as exc:
                logger.debug("Skipping correlation history for %s: %s", sym, exc)

        if len(histories) < 2:
            return {}
        return compute_return_correlation_matrix(histories)

    def run_cycle(
        self,
        symbols: Sequence[str],
        now: datetime | None = None,
        dry_run: bool = False,
        enforce_market_hours: bool = True,
        portfolio_positions: Sequence[PortfolioPosition] = (),
        correlation_matrix: Mapping[tuple[str, str], float] | Mapping[str, float] | None = None,
        starting_equity: float = 100_000.0,
        daily_pnl: float = 0.0,
        fed_dates: Sequence[datetime] = (),
        open_positions: Sequence[ManagedPosition] = (),
        current_prices: Mapping[str, float] | None = None,
        current_volumes: Mapping[str, float] | None = None,
        avg_volumes_20d: Mapping[str, float] | None = None,
        bid_prices: Mapping[str, float] | None = None,
        sync_account: bool = False,
    ) -> PipelineCycleReport:
        """Execute a full trading cycle respecting market hours, exits, and 5-stage funnel."""
        now_ny = normalize_ny_datetime(now)
        market_open = is_market_open(now_ny)

        if enforce_market_hours and not market_open:
            reason = f"Market closed at {now_ny.isoformat()} (America/New_York)"
            logger.info("Skipping cycle: %s", reason)
            return PipelineCycleReport(
                timestamp=now_ny,
                market_open=False,
                skipped=True,
                skip_reason=reason,
                dry_run=dry_run,
            )

        remaining_cash: float | None = None
        active_portfolio: list[PortfolioPosition] = list(portfolio_positions)
        if sync_account:
            (
                starting_equity,
                daily_pnl,
                active_portfolio,
                remaining_cash,
            ) = self._sync_broker_account_state(
                starting_equity=starting_equity,
                daily_pnl=daily_pnl,
                portfolio_positions=active_portfolio,
            )
            if not open_positions:
                reconcile_fn = getattr(self.execution_client, "reconcile_managed_positions", None)
                if callable(reconcile_fn):
                    try:
                        sig = inspect.signature(reconcile_fn)
                        if "ensure_protective_stops" in sig.parameters:
                            reconciled = reconcile_fn(
                                now=now_ny,
                                ensure_protective_stops=not dry_run,
                            )
                        else:
                            reconciled = reconcile_fn(now=now_ny)
                    except (TypeError, ValueError):
                        reconciled = reconcile_fn(now=now_ny)
                    if isinstance(reconciled, Sequence):
                        open_positions = reconciled
                    for closed_pos in getattr(
                        self.execution_client, "last_broker_closed_positions", ()
                    ):
                        closed_sym = str(closed_pos.get("symbol", "")).strip().upper()
                        closed_exit = float(closed_pos.get("exit_price", 0.0) or 0.0)
                        closed_entry = float(closed_pos.get("entry_price", 0.0) or 0.0)
                        if (
                            closed_sym
                            and closed_entry > 0.0
                            and closed_exit <= closed_entry
                        ):
                            self._stop_loss_cooldowns[closed_sym] = now_ny + timedelta(
                                hours=self.stop_reentry_cooldown_hours
                            )
                        close_fn = getattr(self.notifier, "notify_position_closed", None)
                        if callable(close_fn):
                            try:
                                close_fn(
                                    symbol=closed_pos["symbol"],
                                    action=closed_pos.get("action", "TRAILING_STOP_EXIT"),
                                    qty=closed_pos["qty"],
                                    limit_price=closed_pos["exit_price"],
                                    reason=closed_pos.get(
                                        "reason", "Broker GTC order filled"
                                    ),
                                    entry_price=closed_pos.get("entry_price"),
                                    dry_run=dry_run,
                                )
                            except TypeError:
                                close_fn(
                                    symbol=closed_pos["symbol"],
                                    action=closed_pos.get("action", "TRAILING_STOP_EXIT"),
                                    qty=closed_pos["qty"],
                                    limit_price=closed_pos["exit_price"],
                                    reason=closed_pos.get(
                                        "reason", "Broker GTC order filled"
                                    ),
                                )

        exit_results = (
            self._manage_open_positions(
                open_positions=open_positions,
                now_ny=now_ny,
                dry_run=dry_run,
                current_prices=current_prices,
                bid_prices=bid_prices,
            )
            if open_positions
            else []
        )

        normalized_prices = {
            k.strip().upper(): float(v) for k, v in (current_prices or {}).items()
        }
        normalized_cur_vols = {
            k.strip().upper(): float(v) for k, v in (current_volumes or {}).items()
        }
        normalized_avg_vols = {
            k.strip().upper(): float(v) for k, v in (avg_volumes_20d or {}).items()
        }
        cleaned_symbols = [raw.strip().upper() for raw in symbols if raw.strip()]

        # Cash-only guard: skip new entry scanning when broker cash is exhausted (no margin buying)
        min_cash = 15.0 if self.allow_fractional_shares else 100.0
        if remaining_cash is not None and remaining_cash < min_cash:
            cash_reason = (
                f"Insufficient available cash (${remaining_cash:,.2f} < ${min_cash:,.2f}; "
                "margin disabled — monitoring open positions only)"
            )
            logger.info("%s", cash_reason)
            return PipelineCycleReport(
                timestamp=now_ny,
                market_open=market_open,
                skipped=False,
                dry_run=dry_run,
                exit_results=exit_results,
                ticker_outcomes=[
                    TickerPipelineOutcome(
                        symbol=sym,
                        stage_reached="cash_prefilter",
                        status="skipped_insufficient_cash",
                        approved=False,
                        reasons=[cash_reason],
                    )
                    for sym in cleaned_symbols
                ],
            )

        resolved_corr: Mapping[tuple[str, str], float] | Mapping[str, float] = (
            correlation_matrix
            if correlation_matrix is not None
            else self._resolve_correlation_matrix(cleaned_symbols, active_portfolio)
        )
        resolved_fed_dates = fed_dates if fed_dates else self.default_fed_dates

        shared_vix: float | None = None
        if cleaned_symbols and type(self.screener) is UniverseScreener:
            shared_vix = self.screener.market_data.get_vix()

        pre_screened_map: dict[str, UniverseScreeningResult] = {}
        pre_screen_errors: dict[str, str] = {}
        prefiltered_volume_reasons: dict[str, str] = {}
        prefiltered_skip_outcomes: dict[str, TickerPipelineOutcome] = {}
        selected_symbols: set[str] | None = None

        held_symbols = {
            pos.symbol.strip().upper()
            for pos in active_portfolio
            if getattr(pos, "symbol", None)
        }
        vix_blocked_reason: str | None = None
        if (
            self.skip_ai_on_prefilter_fail
            and shared_vix is not None
            and shared_vix > self.guardian.config.max_vix
        ):
            vix_blocked_reason = (
                f"VIX {shared_vix:.2f} > {self.guardian.config.max_vix:.2f} (skipped LLM call)"
            )

        fed_blackout_reason: str | None = None
        if self.skip_ai_on_prefilter_fail and resolved_fed_dates:
            fed_check = check_economic_calendar(
                now=now_ny,
                earnings_dates=(),
                fed_dates=resolved_fed_dates,
                blackout_hours=self.guardian.config.blackout_hours,
            )
            if not fed_check.passed:
                fed_blackout_reason = f"{fed_check.reason} (skipped LLM call)"

        if len(cleaned_symbols) > 1 and self.max_candidates_for_ai > 0:
            symbols_to_screen: list[str] = []
            for symbol in cleaned_symbols:
                if self.skip_ai_on_prefilter_fail and symbol in held_symbols:
                    prefiltered_skip_outcomes[symbol] = TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="portfolio_prefilter",
                        status="skipped_already_held",
                        approved=False,
                        reasons=[f"Symbol {symbol} already held in portfolio (skipped LLM call)"],
                    )
                    continue
                cooldown_until = self._stop_loss_cooldowns.get(symbol)
                if (
                    self.skip_ai_on_prefilter_fail
                    and cooldown_until is not None
                    and now_ny < cooldown_until
                ):
                    prefiltered_skip_outcomes[symbol] = TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="cooldown_prefilter",
                        status="skipped_stop_cooldown",
                        approved=False,
                        reasons=[
                            f"Post-stop-loss cooldown active for {symbol} until {cooldown_until.isoformat()} (skipped LLM call)"
                        ],
                    )
                    continue
                if vix_blocked_reason is not None:
                    prefiltered_skip_outcomes[symbol] = TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="guardian_prefilter",
                        status="skipped_vix_blackout",
                        approved=False,
                        reasons=[vix_blocked_reason],
                    )
                    continue
                if fed_blackout_reason is not None:
                    prefiltered_skip_outcomes[symbol] = TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="calendar_prefilter",
                        status="skipped_calendar_blackout",
                        approved=False,
                        reasons=[fed_blackout_reason],
                    )
                    continue
                symbols_to_screen.append(symbol)

            # Batch-fetch Alpaca IEX snapshots and warm yfinance histories concurrently
            batched_screen_quotes: dict[str, Any] = {}
            if (
                symbols_to_screen
                and type(self.screener) is UniverseScreener
                and type(self.screener.market_data) is MarketDataClient
            ):
                try:
                    batched_screen_quotes = self.screener.market_data.get_quotes_batch(
                        symbols_to_screen
                    )
                except Exception as exc:
                    logger.warning("Batched universe quote fetch failed (%s); falling back", exc)

                def _warm_history(sym: str) -> None:
                    try:
                        self.screener.market_data.get_historical_ohlcv(sym, period="1y")
                    except Exception:
                        pass

                with ThreadPoolExecutor(max_workers=min(6, len(symbols_to_screen))) as pool:
                    list(pool.map(_warm_history, symbols_to_screen))

            for symbol in symbols_to_screen:
                try:
                    prefetched_quote = batched_screen_quotes.get(symbol)
                    if shared_vix is not None:
                        if prefetched_quote is not None:
                            pre_screened_map[symbol] = self.screener.screen_symbol(
                                symbol,
                                now=now_ny,
                                vix=shared_vix,
                                quote=prefetched_quote,
                            )
                        else:
                            pre_screened_map[symbol] = self.screener.screen_symbol(
                                symbol,
                                now=now_ny,
                                vix=shared_vix,
                            )
                    else:
                        if prefetched_quote is not None:
                            pre_screened_map[symbol] = self.screener.screen_symbol(
                                symbol,
                                now=now_ny,
                                quote=prefetched_quote,
                            )
                        else:
                            pre_screened_map[symbol] = self.screener.screen_symbol(
                                symbol,
                                now=now_ny,
                            )
                except Exception as exc:
                    logger.exception("Error pre-screening symbol %s: %s", symbol, exc)
                    pre_screen_errors[symbol] = str(exc)

            ranking_pool = list(pre_screened_map.values())
            analyst_cfg = getattr(self.analyst, "config", None)
            if self.skip_ai_on_prefilter_fail and isinstance(analyst_cfg, AnalystConfig):
                filtered_pool: list[UniverseScreeningResult] = []
                for res in ranking_pool:
                    if not res.passed:
                        filtered_pool.append(res)
                        continue
                    # Skip symbols inside an active 24h earnings blackout before wasting AI calls
                    if res.fundamentals is not None and res.fundamentals.earnings_dates:
                        earnings_dts = _parse_earnings_dates(
                            res.fundamentals.earnings_dates,
                            now_ny=now_ny,
                        )
                        cal_check = check_economic_calendar(
                            now=now_ny,
                            earnings_dates=earnings_dts,
                            fed_dates=(),
                            blackout_hours=self.guardian.config.blackout_hours,
                        )
                        if not cal_check.passed:
                            prefiltered_skip_outcomes[res.symbol] = TickerPipelineOutcome(
                                symbol=res.symbol,
                                stage_reached="calendar_prefilter",
                                status="skipped_calendar_blackout",
                                approved=False,
                                reasons=[f"{cal_check.reason} (skipped LLM call)"],
                                screening_result=res,
                            )
                            continue
                    ind = res.indicators
                    quote = res.quote
                    candidate_ask = (
                        quote.ask_price
                        if quote is not None and quote.ask_price > 0.0
                        else (ind.current_price if ind is not None else 0.0)
                    )
                    if (
                        not self.allow_fractional_shares
                        and remaining_cash is not None
                        and candidate_ask > 0.0
                        and remaining_cash < candidate_ask
                    ):
                        prefiltered_skip_outcomes[res.symbol] = TickerPipelineOutcome(
                            symbol=res.symbol,
                            stage_reached="cash_prefilter",
                            status="skipped_insufficient_cash",
                            approved=False,
                            reasons=[
                                f"Insufficient cash (${remaining_cash:,.2f}) for 1 share of {res.symbol} at ${candidate_ask:,.2f} (skipped LLM call)"
                            ],
                            screening_result=res,
                        )
                        continue
                    ind_latest = float(getattr(ind, "latest_volume", 0.0) or 0.0) if ind else 0.0
                    ind_avg = float(getattr(ind, "avg_volume_20d", 0.0) or 0.0) if ind else 0.0
                    if res.symbol in normalized_cur_vols:
                        cv = normalized_cur_vols[res.symbol]
                    elif ind_latest > 0.0 and ind_avg > 0.0:
                        cv = estimate_intraday_run_rate_volume(ind_latest, now=now_ny)
                    else:
                        cv = quote.volume if quote is not None else 0.0
                    if res.symbol in normalized_avg_vols:
                        av = normalized_avg_vols[res.symbol]
                    elif ind_avg > 0.0:
                        av = ind_avg
                    else:
                        av = quote.volume if quote is not None and quote.volume > 0.0 else cv
                    rel_vol = compute_relative_volume(cv, av)
                    if rel_vol < analyst_cfg.min_relative_volume:
                        prefiltered_volume_reasons[res.symbol] = (
                            f"relative_volume {rel_vol:.2f} < {analyst_cfg.min_relative_volume:.2f} "
                            "(price-volume divergence; skipped LLM call)"
                        )
                    else:
                        filtered_pool.append(res)
                ranking_pool = filtered_pool

            top_candidates = select_top_candidates(
                ranking_pool,
                max_candidates=self.max_candidates_for_ai,
            )
            selected_symbols = {candidate.symbol for candidate in top_candidates}

        outcomes: list[TickerPipelineOutcome] = []
        for symbol in cleaned_symbols:
            if symbol in prefiltered_skip_outcomes:
                outcomes.append(prefiltered_skip_outcomes[symbol])
                continue
            if self.skip_ai_on_prefilter_fail and symbol in held_symbols:
                outcomes.append(
                    TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="portfolio_prefilter",
                        status="skipped_already_held",
                        approved=False,
                        reasons=[f"Symbol {symbol} already held in portfolio (skipped LLM call)"],
                    )
                )
                continue
            if symbol in pre_screen_errors:
                outcomes.append(
                    TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="error",
                        status="error",
                        approved=False,
                        reasons=[pre_screen_errors[symbol]],
                    )
                )
                continue

            pre_screened = pre_screened_map.get(symbol)
            if selected_symbols is not None and pre_screened is not None:
                if not pre_screened.passed:
                    outcomes.append(
                        TickerPipelineOutcome(
                            symbol=symbol,
                            stage_reached="universe",
                            status="rejected_universe",
                            approved=False,
                            reasons=list(pre_screened.rejection_reasons),
                            screening_result=pre_screened,
                        )
                    )
                    continue
                if symbol in prefiltered_volume_reasons:
                    outcomes.append(
                        TickerPipelineOutcome(
                            symbol=symbol,
                            stage_reached="analyst_prefilter",
                            status="rejected_analyst_prefilter",
                            approved=False,
                            reasons=[prefiltered_volume_reasons[symbol]],
                            screening_result=pre_screened,
                        )
                    )
                    continue
                if symbol not in selected_symbols:
                    outcomes.append(
                        TickerPipelineOutcome(
                            symbol=symbol,
                            stage_reached="universe_ranking",
                            status="filtered_max_candidates",
                            approved=False,
                            reasons=[
                                f"Outside top {self.max_candidates_for_ai} candidates for AI analysis"
                            ],
                            screening_result=pre_screened,
                        )
                    )
                    continue

            min_cash = 15.0 if self.allow_fractional_shares else 100.0
            if remaining_cash is not None and remaining_cash < min_cash:
                outcomes.append(
                    TickerPipelineOutcome(
                        symbol=symbol,
                        stage_reached="cash_prefilter",
                        status="skipped_insufficient_cash",
                        approved=False,
                        reasons=[
                            f"Insufficient remaining cash (${remaining_cash:,.2f} < ${min_cash:,.2f}; skipped LLM call)"
                        ],
                        screening_result=pre_screened,
                    )
                )
                continue

            try:
                outcome = self._process_symbol(
                    symbol=symbol,
                    now_ny=now_ny,
                    shared_vix=shared_vix,
                    dry_run=dry_run,
                    portfolio_positions=active_portfolio,
                    correlation_matrix=resolved_corr,
                    starting_equity=starting_equity,
                    daily_pnl=daily_pnl,
                    fed_dates=resolved_fed_dates,
                    current_prices=normalized_prices,
                    current_volumes=normalized_cur_vols,
                    avg_volumes_20d=normalized_avg_vols,
                    pre_screened=pre_screened,
                    available_capital=remaining_cash,
                )
            except Exception as exc:
                logger.exception("Error processing symbol %s: %s", symbol, exc)
                outcome = TickerPipelineOutcome(
                    symbol=symbol,
                    stage_reached="error",
                    status="error",
                    approved=False,
                    reasons=[str(exc)],
                )
            if outcome.approved and outcome.order_result is not None:
                spent = float(outcome.order_result.qty) * float(outcome.order_result.limit_price)
                if remaining_cash is not None:
                    remaining_cash = max(0.0, remaining_cash - spent)
                active_portfolio.append(
                    PortfolioPosition(
                        symbol=symbol,
                        sector=self.sector_map.get(symbol, self.default_sector),
                        market_value=spent,
                        shares=float(outcome.order_result.qty),
                        avg_price=float(outcome.order_result.limit_price),
                    )
                )
            outcomes.append(outcome)

        return PipelineCycleReport(
            timestamp=now_ny,
            market_open=market_open,
            skipped=False,
            skip_reason=None,
            dry_run=dry_run,
            exit_results=exit_results,
            ticker_outcomes=outcomes,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for single-cycle or continuous trading bot execution."""
    parser = argparse.ArgumentParser(
        description="Algorithmic Volatility & Equity Trading Bot (5-stage decision funnel)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="List of equity ticker symbols to evaluate (defaults to strategy config or AAPL MSFT NVDA)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to strategy JSON config (defaults to config/strategy.local.json if present)",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=None,
        help="Advisory runtime lockfile path preventing duplicate bot instances",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the full pipeline without submitting live orders to Alpaca",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in a polling loop until interrupted",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Polling interval in seconds when --loop is enabled (default: 60.0)",
    )
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Bypass the America/New_York 09:30-16:00 market-hours gate (useful for off-hours dry-runs)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite audit database path (default: data/trading_audit.db)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run historical walk-forward backtest over cached/fetched OHLCV data and exit",
    )
    parser.add_argument(
        "--backtest-period",
        default="1y",
        help="Historical lookback period for --backtest mode (default: 1y)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital for --backtest mode (default: 10000.0)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum top-ranked Stage-1 candidates passed to AI news/sentiment analysis (defaults to strategy config or 5)",
    )
    parser.add_argument(
        "--sync-account",
        action="store_true",
        help="Sync live/paper Alpaca account equity, daily PnL, and open positions for compounding",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Send a sample HTML 'Good News' trade notification email via Gmail SMTP and exit",
    )
    parser.add_argument(
        "--send-daily-summary",
        action="store_true",
        help="Sync live/paper account state, send the daily summary HTML email, and exit",
    )
    parser.add_argument(
        "--close-symbols",
        nargs="+",
        default=None,
        help="Close specific open broker positions via protective Limit SELL orders and exit",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run walk-forward universe and wave-parameter self-calibration and exit",
    )
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Persist improved self-calibration parameters and active universe to strategy.local.json",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be strictly positive")
    if args.initial_capital <= 0:
        parser.error("--initial-capital must be strictly positive")
    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates must be strictly positive")
    return args


def _run_backtest_cli(
    args: argparse.Namespace,
    settings: StrategySettings,
    guardian: RiskGuardian | None = None,
) -> int:
    """Execute a historical backtest lazily importing src.backtest and log/print metrics."""
    from src.backtest import BacktestConfig, BacktestEngine, ParquetCandleStore

    raw_tickers = args.tickers if args.tickers is not None else settings.tickers
    symbols = [ticker.strip().upper() for ticker in raw_tickers if ticker.strip()]
    store = ParquetCandleStore()
    market_data = {
        sym: store.get_or_fetch_candles(sym, period=args.backtest_period)
        for sym in symbols
    }
    vix_series = None
    try:
        vix_df = store.get_or_fetch_candles("^VIX", period=args.backtest_period)
        if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
            vix_series = vix_df["Close"]
    except Exception as exc:
        logger.warning("Could not load ^VIX history for backtest (%s); using default VIX", exc)

    bt_kwargs: dict[str, Any] = {
        "initial_capital": args.initial_capital,
        "atr_stop_multiplier": settings.atr_stop_multiplier,
        "atr_take_profit_multiplier": settings.atr_take_profit_multiplier,
        "max_hold_bars": settings.max_hold_bars,
        "rsi_entry_max": settings.rsi_entry_max,
        "min_relative_volume": settings.min_relative_volume,
    }
    if settings.max_stop_loss_pct > 0.0:
        bt_kwargs["max_stop_loss_pct"] = settings.max_stop_loss_pct
    if settings.max_ema_extension_atr > 0.0:
        bt_kwargs["max_ema_extension_atr"] = settings.max_ema_extension_atr
    if settings.breakeven_trigger_pct > 0.0:
        bt_kwargs["breakeven_trigger_pct"] = settings.breakeven_trigger_pct
    if settings.peak_activation_pct > 0.0:
        bt_kwargs["peak_activation_pct"] = settings.peak_activation_pct
    if settings.peak_rsi_threshold != 66.0:
        bt_kwargs["peak_rsi_threshold"] = settings.peak_rsi_threshold
    if settings.allow_fractional_shares:
        bt_kwargs["allow_fractional_shares"] = True
    config = BacktestConfig(**bt_kwargs)
    resolved_guardian = guardian
    if resolved_guardian is None:
        if (
            settings.risk_per_trade_pct != 0.01
            or settings.max_asset_exposure_pct != 0.20
            or settings.max_sector_exposure_pct != 0.35
            or settings.kelly_fraction != 0.25
        ):
            from src.guardian import GuardianConfig

            resolved_guardian = RiskGuardian(
                GuardianConfig(
                    risk_per_trade_pct=settings.risk_per_trade_pct,
                    max_asset_exposure_pct=settings.max_asset_exposure_pct,
                    max_sector_exposure_pct=settings.max_sector_exposure_pct,
                    kelly_fraction=settings.kelly_fraction,
                )
            )
        else:
            resolved_guardian = RiskGuardian()
    engine = BacktestEngine(guardian=resolved_guardian, config=config)
    corr_matrix = compute_return_correlation_matrix(market_data)
    report = engine.run(
        market_data=market_data,
        vix_series=vix_series,
        sector_map=settings.sector_map,
        correlation_matrix=corr_matrix,
    )
    metrics = report.metrics

    summary = (
        "\n=== Backtest Summary ===\n"
        f"Tickers       : {', '.join(symbols)}\n"
        f"Period        : {args.backtest_period}\n"
        f"Total Trades  : {metrics.total_trades}\n"
        f"Win Rate      : {metrics.win_rate:.2%}\n"
        f"Profit Factor : {metrics.profit_factor:.2f}\n"
        f"Max Drawdown  : {metrics.max_drawdown_pct:.2%}\n"
        f"Total Return  : {metrics.total_return_pct:.2%}\n"
        f"Initial Capital: ${metrics.initial_capital:,.2f}\n"
        f"Final Equity  : ${metrics.final_equity:,.2f}\n"
        f"Compounded PnL: ${metrics.compounded_pnl:,.2f}\n"
        f"Guardian Vetoes: {report.vetoed_entries_count}\n"
        "========================"
    )
    logger.info("%s", summary)
    print(summary)
    return 0


def _build_orchestrator_from_settings(
    db_path: Path,
    settings: StrategySettings,
    max_candidates: int,
) -> TradingBotOrchestrator:
    """Construct a crash-safe TradingBotOrchestrator configured from StrategySettings."""
    from src.execution import AlpacaExecutionClient, ExecutionConfig, PositionManager

    audit_store = TradingAuditStore(db_path=db_path)
    exec_config = ExecutionConfig(
        trailing_atr_multiplier=settings.trailing_atr_multiplier,
        take_profit_atr_multiplier=settings.atr_take_profit_multiplier,
        enable_take_profit_exit=settings.atr_take_profit_multiplier > 0.0,
        max_stop_loss_pct=settings.max_stop_loss_pct,
        breakeven_trigger_pct=settings.breakeven_trigger_pct,
        peak_activation_pct=settings.peak_activation_pct,
        peak_pullback_pct=settings.peak_pullback_pct,
        max_hold_hours=settings.max_hold_hours,
        min_target_return_pct=settings.min_target_return_pct,
    )
    execution_client = AlpacaExecutionClient(audit_store=audit_store, config=exec_config)
    position_manager = PositionManager(
        execution_client=execution_client,
        audit_store=audit_store,
        config=exec_config,
    )
    screener = UniverseScreener(
        market_data=MarketDataClient(),
        fmp_client=FmpClient(store=audit_store),
        config=UniverseConfig(max_spread_pct=settings.max_spread_pct),
    )
    analyst = GeminiAnalyst(
        audit_store=audit_store,
        config=AnalystConfig(
            min_relative_volume=settings.min_relative_volume,
            min_sentiment_score=0.10,
        ),
    )
    from src.guardian import GuardianConfig

    guardian = RiskGuardian(
        GuardianConfig(
            risk_per_trade_pct=settings.risk_per_trade_pct,
            max_asset_exposure_pct=settings.max_asset_exposure_pct,
            max_sector_exposure_pct=settings.max_sector_exposure_pct,
            kelly_fraction=settings.kelly_fraction,
        )
    )
    return TradingBotOrchestrator(
        screener=screener,
        analyst=analyst,
        guardian=guardian,
        execution_client=execution_client,
        position_manager=position_manager,
        audit_store=audit_store,
        notifier=CompositeNotifier(),
        confirmation_delay_minutes=settings.confirmation_delay_minutes,
        max_confirmation_delay_minutes=settings.max_confirmation_delay_minutes,
        default_sector=settings.default_sector,
        sector_map=settings.sector_map,
        take_profit_atr_multiplier=settings.atr_take_profit_multiplier,
        max_candidates_for_ai=max_candidates,
        default_fed_dates=load_fed_dates(),
        skip_ai_on_prefilter_fail=True,
        max_stop_loss_pct=settings.max_stop_loss_pct,
        max_ema_extension_atr=settings.max_ema_extension_atr,
        allow_fractional_shares=settings.allow_fractional_shares,
    )


def _log_cycle_summary(report: PipelineCycleReport) -> None:
    """Log a concise summary of the completed pipeline cycle."""
    if report.skipped:
        logger.info("Cycle skipped: %s", report.skip_reason)
        return

    logger.info(
        "Cycle complete at %s | dry_run=%s | exits=%d | tickers=%d",
        report.timestamp.isoformat(),
        report.dry_run,
        len(report.exit_results),
        len(report.ticker_outcomes),
    )
    for outcome in report.ticker_outcomes:
        logger.info(
            "  %s -> stage=%s status=%s approved=%s reasons=%s",
            outcome.symbol,
            outcome.stage_reached,
            outcome.status,
            outcome.approved,
            outcome.reasons,
        )


def _run_loop(
    bot: TradingBotOrchestrator,
    symbols: list[str],
    args: argparse.Namespace,
    sleep_fn: Callable[[float], None],
) -> int:
    """Run one or continuous cycles with signal handling and exponential error backoff."""
    enforce_market_hours = not args.ignore_market_hours
    stop_event = threading.Event()

    def _request_stop(signum: int, _frame: object) -> None:
        logger.info("Received signal %s; shutting down gracefully after current step.", signum)
        stop_event.set()

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    consecutive_errors = 0
    had_open_market_cycle = False
    try:
        while not stop_event.is_set():
            run_kwargs: dict[str, Any] = {
                "symbols": symbols,
                "dry_run": args.dry_run,
                "enforce_market_hours": enforce_market_hours,
            }
            if args.sync_account:
                run_kwargs["sync_account"] = True

            try:
                report = bot.run_cycle(**run_kwargs)
                _log_cycle_summary(report)
                consecutive_errors = 0
                wait_seconds = args.interval_seconds
                if report.market_open and not report.skipped:
                    had_open_market_cycle = True
                elif report.skipped and not report.market_open and had_open_market_cycle:
                    try:
                        eq, pnl, positions, cash = bot._sync_broker_account_state(
                            starting_equity=100_000.0,
                            daily_pnl=0.0,
                            portfolio_positions=(),
                        )
                        summary_fn = getattr(bot.notifier, "notify_daily_summary", None)
                        if callable(summary_fn):
                            summary_fn(
                                equity=eq + pnl,
                                daily_pnl=pnl,
                                cash=cash,
                                open_positions=positions,
                            )
                    except Exception as exc:
                        logger.warning("Could not send end-of-day summary email: %s", exc)
                    if type(bot) is TradingBotOrchestrator:
                        try:
                            _run_eod_auto_calibration(bot, args)
                        except Exception as exc:
                            logger.warning("End-of-day self-calibration skipped: %s", exc)
                    logger.info(
                        "Market closed for the day after active session; daily summary sent. Exiting loop cleanly."
                    )
                    break
            except KeyboardInterrupt:
                logger.info("Trading bot stopped cleanly by user interrupt.")
                return 0
            except Exception as exc:
                consecutive_errors += 1
                logger.exception("Cycle error (#%d): %s", consecutive_errors, exc)
                notifier = getattr(bot, "notifier", None)
                if notifier is not None and hasattr(notifier, "send_message"):
                    notifier.send_message(
                        f"BOT CYCLE ERROR (#{consecutive_errors}): {exc}"
                    )
                if not args.loop:
                    raise
                wait_seconds = min(
                    args.interval_seconds * (2 ** min(consecutive_errors - 1, 4)),
                    300.0,
                )

            if not args.loop or stop_event.is_set():
                break
            if sleep_fn is time_module.sleep:
                stop_event.wait(timeout=wait_seconds)
            else:
                sleep_fn(wait_seconds)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped cleanly by user interrupt.")
        return 0
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

    return 0


def _run_eod_auto_calibration(
    bot: TradingBotOrchestrator,
    args: argparse.Namespace,
) -> None:
    """Automatically recalibrate active universe and wave parameters using today's candles and live SQLite outcomes."""
    import json as json_module
    from src.backtest import ParquetCandleStore
    from src.calibration import StrategySelfCalibrator
    from src.config import DEFAULT_STRATEGY_CONFIG_PATH

    cfg_path = getattr(args, "config", None) or DEFAULT_STRATEGY_CONFIG_PATH
    settings = load_strategy_settings(cfg_path)
    master_tickers: list[str] = list(settings.tickers)
    if Path(cfg_path).exists():
        try:
            raw_cfg = json_module.loads(Path(cfg_path).read_text(encoding="utf-8"))
            if isinstance(raw_cfg, dict) and isinstance(raw_cfg.get("master_tickers"), list):
                master_tickers = [
                    str(t).strip().upper()
                    for t in raw_cfg["master_tickers"]
                    if str(t).strip()
                ]
        except Exception:
            pass

    store = ParquetCandleStore()
    all_syms = sorted(set(master_tickers) | set(settings.tickers))
    market_data = {
        sym: store.get_or_fetch_candles(sym, period="3y", refresh=True)
        for sym in all_syms
    }
    vix_series = None
    try:
        vix_df = store.get_or_fetch_candles("^VIX", period="3y", refresh=True)
        if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
            vix_series = vix_df["Close"]
    except Exception:
        pass

    calibrator = StrategySelfCalibrator()
    report = calibrator.calibrate(
        market_data=market_data,
        vix_series=vix_series,
        settings=settings,
        master_tickers=master_tickers,
        audit_store=bot.audit_store,
    )
    applied = calibrator.apply_to_config_file(report, config_path=cfg_path)
    logger.info(
        "EOD Auto-Calibration complete: improved=%s applied=%s active=%d pruned=%d score=%.4f->%.4f",
        report.improved,
        applied,
        len(report.active_tickers),
        len(report.pruned_tickers),
        report.baseline_score,
        report.calibrated_score,
    )


def main(
    argv: Sequence[str] | None = None,
    orchestrator: TradingBotOrchestrator | None = None,
    sleep_fn: Callable[[float], None] = time_module.sleep,
) -> int:
    """CLI entrypoint loading .env, private strategy config, and crash-safe runtime loop."""
    load_dotenv()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    args = parse_args(argv)
    settings = load_strategy_settings(args.config)
    resolved_tickers = [
        t.strip().upper()
        for t in (args.tickers if args.tickers is not None else settings.tickers)
        if t.strip()
    ]
    resolved_max_candidates = (
        args.max_candidates
        if args.max_candidates is not None
        else settings.max_candidates_for_ai
    )

    if args.test_email:
        email_notifier = EmailNotifier()
        ok = email_notifier.notify_position_closed(
            symbol="JNJ",
            action="TAKE_PROFIT_EXIT",
            qty=73,
            limit_price=272.25,
            reason="Prueba de plantilla HTML — Objetivo de ganancia alcanzado",
            entry_price=266.74,
            dry_run=True,
        )
        if ok:
            logger.info("Sample HTML email sent successfully to %s", email_notifier.recipient_email)
            return 0
        logger.error(
            "Could not send sample email. Verify SMTP_USER, SMTP_APP_PASSWORD, and EMAIL_RECIPIENT in .env"
        )
        return 1

    if args.backtest:
        return _run_backtest_cli(
            args,
            settings=settings,
            guardian=orchestrator.guardian if orchestrator is not None else None,
        )

    if args.calibrate:
        import json as json_module
        from src.backtest import ParquetCandleStore
        from src.calibration import StrategySelfCalibrator
        from src.config import DEFAULT_STRATEGY_CONFIG_PATH

        cfg_path = args.config if args.config is not None else DEFAULT_STRATEGY_CONFIG_PATH
        master_tickers: list[str] = list(resolved_tickers)
        if Path(cfg_path).exists():
            try:
                raw_cfg = json_module.loads(Path(cfg_path).read_text(encoding="utf-8"))
                if isinstance(raw_cfg, dict) and isinstance(raw_cfg.get("master_tickers"), list):
                    master_tickers = [
                        str(t).strip().upper()
                        for t in raw_cfg["master_tickers"]
                        if str(t).strip()
                    ]
            except Exception:
                pass

        store = ParquetCandleStore()
        all_syms = sorted(set(master_tickers) | set(resolved_tickers))
        market_data = {
            sym: store.get_or_fetch_candles(sym, period=args.backtest_period)
            for sym in all_syms
        }
        vix_series = None
        try:
            vix_df = store.get_or_fetch_candles("^VIX", period=args.backtest_period)
            if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
                vix_series = vix_df["Close"]
        except Exception:
            pass

        calibrator = StrategySelfCalibrator()
        cal_audit_store = (
            orchestrator.audit_store
            if orchestrator is not None
            else TradingAuditStore(db_path=args.db_path)
        )
        cal_report = calibrator.calibrate(
            market_data=market_data,
            vix_series=vix_series,
            settings=settings,
            master_tickers=master_tickers,
            audit_store=cal_audit_store,
        )
        applied = False
        if args.apply_calibration and cal_report.improved:
            applied = calibrator.apply_to_config_file(cal_report, config_path=cfg_path)

        bm = cal_report.baseline_metrics
        cm = cal_report.calibrated_metrics
        cal_summary = (
            "\n=== Self-Calibration Walk-Forward Report ===\n"
            f"Baseline      : Trades={bm.total_trades} | WR={bm.win_rate:.2%} | PF={bm.profit_factor:.2f} | DD={bm.max_drawdown_pct:.2%} | Return={bm.total_return_pct:.2%} (Score={cal_report.baseline_score:.4f})\n"
            f"Calibrated    : Trades={cm.total_trades} | WR={cm.win_rate:.2%} | PF={cm.profit_factor:.2f} | DD={cm.max_drawdown_pct:.2%} | Return={cm.total_return_pct:.2%} (Score={cal_report.calibrated_score:.4f})\n"
            f"Active Tickers ({len(cal_report.active_tickers)}): {', '.join(cal_report.active_tickers)}\n"
            f"Pruned Tickers ({len(cal_report.pruned_tickers)}): {', '.join(cal_report.pruned_tickers) or 'None'}\n"
            f"Wave Params   : {cal_report.recommended_params}\n"
            f"Improved      : {cal_report.improved} | Applied to Config: {applied}\n"
            "============================================"
        )
        logger.info("%s", cal_summary)
        print(cal_summary)
        return 0

    if args.send_daily_summary:
        bot = orchestrator or _build_orchestrator_from_settings(
            db_path=args.db_path,
            settings=settings,
            max_candidates=resolved_max_candidates,
        )
        eq, pnl, positions, cash = bot._sync_broker_account_state(
            starting_equity=100_000.0,
            daily_pnl=0.0,
            portfolio_positions=(),
        )
        summary_fn = getattr(bot.notifier, "notify_daily_summary", None)
        if callable(summary_fn):
            ok = summary_fn(
                equity=eq + pnl,
                daily_pnl=pnl,
                cash=cash,
                open_positions=positions,
            )
            return 0 if ok else 1
        return 1

    if args.close_symbols:
        bot = orchestrator or _build_orchestrator_from_settings(
            db_path=args.db_path,
            settings=settings,
            max_candidates=resolved_max_candidates,
        )
        targets = {s.strip().upper() for s in args.close_symbols if s.strip()}
        open_positions = bot.execution_client.get_open_positions() or []
        pos_by_sym = {p.symbol.strip().upper(): p for p in open_positions}
        for sym in sorted(targets):
            pos = pos_by_sym.get(sym)
            if pos is None or pos.shares <= 0:
                logger.warning("No active broker position found for %s; skipping", sym)
                continue
            ref_price = pos.market_value / pos.shares if pos.market_value > 0 else pos.avg_price
            try:
                q = bot.screener.market_data.get_quote(sym)
                if q.bid_price > 0:
                    ref_price = q.bid_price
                elif q.price > 0:
                    ref_price = q.price
            except Exception:
                pass
            order_res = bot.execution_client.submit_limit_sell(
                symbol=sym,
                qty=pos.shares,
                reference_price=ref_price,
                dry_run=args.dry_run,
                reason="Manual CLI position close (--close-symbols)",
            )
            if not args.dry_run:
                bot.audit_store.delete_position_state(sym)
            close_fn = getattr(bot.notifier, "notify_position_closed", None)
            if callable(close_fn):
                close_fn(
                    symbol=sym,
                    action="MANUAL_LIMIT_CLOSE",
                    qty=pos.shares,
                    limit_price=order_res.limit_price,
                    reason="Cierre manual programado (--close-symbols) mediante orden Limit",
                    entry_price=pos.avg_price,
                    dry_run=args.dry_run,
                )
            logger.info(
                "Submitted limit close for %s: qty=%s limit_price=%.2f (dry_run=%s)",
                sym,
                pos.shares,
                order_res.limit_price,
                args.dry_run,
            )
        return 0

    if orchestrator is not None:
        return _run_loop(orchestrator, resolved_tickers, args, sleep_fn)

    lock_path = args.lock_file or args.db_path.with_suffix(".lock")
    with ProcessLock(lock_path):
        bot = _build_orchestrator_from_settings(
            db_path=args.db_path,
            settings=settings,
            max_candidates=resolved_max_candidates,
        )
        return _run_loop(bot, resolved_tickers, args, sleep_fn)


if __name__ == "__main__":
    raise SystemExit(main())
