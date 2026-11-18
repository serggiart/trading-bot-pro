"""Main orchestration pipeline and CLI entrypoint for the algorithmic trading bot."""

from __future__ import annotations

import argparse
import logging
import time as time_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.analyst import AnalystEvaluationResult, GeminiAnalyst, NewsAggregator, NewsItem
from src.guardian import (
    GuardianDecision,
    MarketContext,
    PortfolioPosition,
    RiskGuardian,
    TradeCandidate,
)
from src.notifications import TelegramNotifier
from src.storage import DEFAULT_DB_PATH, TradingAuditStore
from src.universe import (
    FmpClient,
    MarketDataClient,
    UniverseScreener,
    UniverseScreeningResult,
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
    ) -> None:
        self.audit_store = audit_store or TradingAuditStore()
        self.guardian = guardian or RiskGuardian()
        self.notifier = notifier or TelegramNotifier()

        self._screener = screener
        self._news_aggregator = news_aggregator
        self._analyst = analyst
        self._execution_client = execution_client
        self._position_manager = position_manager

        self.confirmation_delay_minutes = float(confirmation_delay_minutes)
        self.max_confirmation_delay_minutes = float(max_confirmation_delay_minutes)
        self.default_sector = default_sector
        self.sector_map: dict[str, str] = {
            k.strip().upper(): v for k, v in (sector_map or {}).items()
        }

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

        for position in open_positions:
            symbol = position.symbol.strip().upper()
            current_price = price_lookup.get(symbol)
            bid_price = bid_lookup.get(symbol)

            if current_price is None:
                quote = self.screener.market_data.get_quote(symbol)
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
                self.notifier.notify_position_closed(
                    symbol=symbol,
                    action=result.action,
                    qty=position.qty,
                    limit_price=limit_price,
                    reason=result.exit_reason or result.action,
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
    ) -> TickerPipelineOutcome:
        """Run a single symbol sequentially through stages 1-5 of the decision funnel."""
        # Stage 1: Universe screening
        if shared_vix is not None:
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

        # Stage 3: AI Analyst evaluation
        analyst_result = self.analyst.evaluate(
            symbol=symbol,
            current_volume=cur_vol,
            avg_volume_20d=avg_vol,
            news_items=news_items,
            as_of=now_ny,
        )
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
        context = MarketContext(
            now=now_ny,
            vix=indicators.vix,
            starting_equity=starting_equity,
            daily_pnl=daily_pnl,
            portfolio_positions=portfolio_positions,
            correlation_matrix=correlation_matrix,
            fed_dates=fed_dates,
            earnings_dates=earnings_dts,
        )
        decision = self.guardian.evaluate(candidate, context)

        recommended_notional = (
            decision.position_size.position_value
            if decision.position_size is not None
            else None
        )
        self.audit_store.log_guardian_decision(
            symbol=symbol,
            action="BUY",
            approved=decision.approved,
            veto_reasons=decision.veto_reasons,
            vix=indicators.vix,
            recommended_shares=float(decision.recommended_shares),
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
        order_result = self.execution_client.submit_limit_buy(
            symbol=symbol,
            qty=decision.recommended_shares,
            ask_price=ask_price,
            stop_loss_price=stop_loss_price,
            dry_run=dry_run,
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
        resolved_corr: Mapping[tuple[str, str], float] | Mapping[str, float] = (
            correlation_matrix if correlation_matrix is not None else {}
        )

        shared_vix: float | None = None
        if symbols and type(self.screener) is UniverseScreener:
            shared_vix = self.screener.market_data.get_vix()

        outcomes: list[TickerPipelineOutcome] = []
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper()
            if not symbol:
                continue
            try:
                outcome = self._process_symbol(
                    symbol=symbol,
                    now_ny=now_ny,
                    shared_vix=shared_vix,
                    dry_run=dry_run,
                    portfolio_positions=portfolio_positions,
                    correlation_matrix=resolved_corr,
                    starting_equity=starting_equity,
                    daily_pnl=daily_pnl,
                    fed_dates=fed_dates,
                    current_prices=normalized_prices,
                    current_volumes=normalized_cur_vols,
                    avg_volumes_20d=normalized_avg_vols,
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
        default=["AAPL", "MSFT", "NVDA"],
        help="List of equity ticker symbols to evaluate (default: AAPL MSFT NVDA)",
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

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be strictly positive")
    return args


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


def main(
    argv: Sequence[str] | None = None,
    orchestrator: TradingBotOrchestrator | None = None,
    sleep_fn: Callable[[float], None] = time_module.sleep,
) -> int:
    """CLI entrypoint loading .env and executing one or continuous trading cycles."""
    load_dotenv()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    args = parse_args(argv)
    bot = orchestrator or TradingBotOrchestrator(
        audit_store=TradingAuditStore(db_path=args.db_path),
    )
    enforce_market_hours = not args.ignore_market_hours

    try:
        while True:
            report = bot.run_cycle(
                symbols=args.tickers,
                dry_run=args.dry_run,
                enforce_market_hours=enforce_market_hours,
            )
            _log_cycle_summary(report)
            if not args.loop:
                break
            sleep_fn(args.interval_seconds)
    except KeyboardInterrupt:
        logger.info("Trading bot stopped cleanly by user interrupt.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
