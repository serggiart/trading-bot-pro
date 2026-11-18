"""Deterministic RiskGuardian orchestration engine."""

from __future__ import annotations

from src.guardian.models import (
    GuardianConfig,
    GuardianDecision,
    MarketContext,
    RuleResult,
    TradeCandidate,
    normalize_to_ny,
)
from src.guardian.rules import (
    check_circuit_breaker,
    check_economic_calendar,
    check_sector_correlation,
    check_vix_regime,
)
from src.guardian.sizing import calculate_atr_position_size, calculate_fractional_kelly


class RiskGuardian:
    """Pure deterministic risk engine with absolute veto authority.

    Executes all risk rules without short-circuiting so every veto reason is surfaced,
    then computes ATR + Fractional Kelly position sizing only when all rules pass.
    """

    def __init__(self, config: GuardianConfig | None = None) -> None:
        self.config = config or GuardianConfig()

    def evaluate(
        self,
        candidate: TradeCandidate,
        context: MarketContext,
    ) -> GuardianDecision:
        """Evaluate a trade candidate against all deterministic rules and sizing limits."""
        now_ny = normalize_to_ny(context.now)
        symbol = str(candidate.symbol).strip().upper()
        held_symbols = [pos.symbol for pos in context.portfolio_positions]
        combined_earnings = [*context.earnings_dates, *candidate.earnings_dates]

        rule_results: list[RuleResult] = [
            check_vix_regime(
                vix=context.vix,
                max_vix=self.config.max_vix,
            ),
            check_sector_correlation(
                candidate_symbol=symbol,
                current_portfolio_symbols=held_symbols,
                correlation_matrix=context.correlation_matrix,
                max_correlation=self.config.max_correlation,
            ),
            check_economic_calendar(
                now=now_ny,
                earnings_dates=combined_earnings,
                fed_dates=context.fed_dates,
                blackout_hours=self.config.blackout_hours,
            ),
            check_circuit_breaker(
                daily_pnl=context.daily_pnl,
                starting_equity=context.starting_equity,
                max_daily_drawdown_pct=self.config.max_daily_drawdown_pct,
            ),
        ]

        veto_reasons = [result.reason for result in rule_results if not result.passed]
        if veto_reasons:
            return GuardianDecision(
                approved=False,
                symbol=symbol,
                timestamp=now_ny,
                rule_results=rule_results,
                veto_reasons=veto_reasons,
                recommended_shares=0,
                position_size=None,
            )

        capital = (
            context.available_capital
            if context.available_capital is not None
            else (context.starting_equity + context.daily_pnl)
        )

        candidate_sector = candidate.sector.strip().lower()
        current_sector_value = 0.0
        if candidate_sector:
            current_sector_value = sum(
                pos.effective_market_value
                for pos in context.portfolio_positions
                if pos.sector.strip().lower() == candidate_sector
            )
        current_sector_exposure_pct = (
            current_sector_value / capital if capital > 0.0 else 0.0
        )

        kelly_cap_pct: float | None = None
        if (
            self.config.use_kelly_cap
            and candidate.win_rate is not None
            and candidate.payoff_ratio is not None
        ):
            kelly_cap_pct = calculate_fractional_kelly(
                win_rate=candidate.win_rate,
                payoff_ratio=candidate.payoff_ratio,
                kelly_fraction=self.config.kelly_fraction,
                max_allocation_pct=self.config.max_asset_exposure_pct,
            )

        position_size = calculate_atr_position_size(
            capital=capital,
            risk_per_trade=self.config.risk_per_trade_pct,
            entry_price=candidate.entry_price,
            atr=candidate.atr,
            atr_multiplier=self.config.atr_multiplier,
            max_asset_exposure_pct=self.config.max_asset_exposure_pct,
            max_sector_exposure_pct=self.config.max_sector_exposure_pct,
            current_sector_exposure_pct=current_sector_exposure_pct,
            kelly_cap_pct=kelly_cap_pct,
        )

        if position_size.shares <= 0:
            return GuardianDecision(
                approved=False,
                symbol=symbol,
                timestamp=now_ny,
                rule_results=rule_results,
                veto_reasons=[position_size.reason],
                recommended_shares=0,
                position_size=position_size,
            )

        return GuardianDecision(
            approved=True,
            symbol=symbol,
            timestamp=now_ny,
            rule_results=rule_results,
            veto_reasons=[],
            recommended_shares=position_size.shares,
            position_size=position_size,
        )
