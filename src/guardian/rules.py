"""Pure deterministic risk rules for the Risk Guardian."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from src.guardian.models import RuleResult, normalize_to_ny


def _is_finite(value: Any) -> bool:
    """Return True if value is a finite int/float (excluding bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_vix_regime(vix: float, max_vix: float = 30.0) -> RuleResult:
    """Block trades when VIX exceeds max_vix or is invalid/negative/NaN.

    Boundary behavior: vix == max_vix (e.g., 30.0) is allowed.
    """
    rule_name = "vix_regime"
    if not _is_finite(max_vix) or max_vix <= 0.0:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Invalid max_vix threshold: {max_vix!r}",
            details={"vix": vix, "max_vix": max_vix},
        )

    if not _is_finite(vix) or vix < 0.0:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Invalid or negative VIX reading: {vix!r}",
            details={"vix": vix, "max_vix": max_vix},
        )

    if vix > max_vix:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"VIX {vix:.2f} exceeds maximum allowed threshold {max_vix:.2f}",
            details={"vix": vix, "max_vix": max_vix},
        )

    return RuleResult(
        rule_name=rule_name,
        passed=True,
        reason=f"VIX {vix:.2f} is within allowed regime (<= {max_vix:.2f})",
        details={"vix": vix, "max_vix": max_vix},
    )


def _build_correlation_lookup(
    correlation_matrix: Mapping[tuple[str, str], float] | Mapping[str, float],
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Normalize pair-keyed and symbol-keyed correlation mappings to uppercase."""
    pair_map: dict[tuple[str, str], float] = {}
    symbol_map: dict[str, float] = {}

    for key, value in correlation_matrix.items():
        if isinstance(key, tuple) and len(key) == 2:
            a = str(key[0]).strip().upper()
            b = str(key[1]).strip().upper()
            pair_map[(a, b)] = value
        elif isinstance(key, str):
            symbol_map[key.strip().upper()] = value

    return pair_map, symbol_map


def check_sector_correlation(
    candidate_symbol: str,
    current_portfolio_symbols: Iterable[str],
    correlation_matrix: Mapping[tuple[str, str], float] | Mapping[str, float],
    max_correlation: float = 0.75,
) -> RuleResult:
    """Block trades if candidate is already held or correlates > max_correlation with holdings.

    Supports both tuple-keyed matrices {(a, b): corr} (checked symmetrically) and
    single-symbol mappings {held_symbol: corr}. Boundary correlation == max_correlation passes.
    """
    rule_name = "sector_correlation"
    candidate = str(candidate_symbol).strip().upper()
    if not candidate:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason="Candidate symbol must not be empty",
            details={"candidate": candidate_symbol},
        )

    if not _is_finite(max_correlation) or max_correlation < 0.0:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Invalid max_correlation threshold: {max_correlation!r}",
            details={"candidate": candidate, "max_correlation": max_correlation},
        )

    held_symbols = [
        str(sym).strip().upper()
        for sym in current_portfolio_symbols
        if sym is not None and str(sym).strip()
    ]

    if candidate in held_symbols:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Candidate {candidate} is already present in the portfolio",
            details={"candidate": candidate, "duplicate": True},
        )

    if not held_symbols:
        return RuleResult(
            rule_name=rule_name,
            passed=True,
            reason=f"No existing portfolio holdings to correlate against {candidate}",
            details={"candidate": candidate, "max_observed_correlation": 0.0},
        )

    pair_map, symbol_map = _build_correlation_lookup(correlation_matrix)
    breaches: list[dict[str, Any]] = []
    invalid_entries: list[dict[str, Any]] = []
    max_observed: float | None = None

    for held in held_symbols:
        corr: float | None = None
        if (candidate, held) in pair_map:
            corr = pair_map[(candidate, held)]
        elif (held, candidate) in pair_map:
            corr = pair_map[(held, candidate)]
        elif held in symbol_map:
            corr = symbol_map[held]

        if corr is None:
            continue

        if not _is_finite(corr):
            invalid_entries.append({"symbol": held, "correlation": corr})
            continue

        if max_observed is None or corr > max_observed:
            max_observed = corr

        if corr > max_correlation:
            breaches.append({"symbol": held, "correlation": corr})

    if invalid_entries:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Invalid non-finite correlation value for {candidate}: {invalid_entries}",
            details={"candidate": candidate, "invalid_entries": invalid_entries},
        )

    if breaches:
        breach_desc = ", ".join(f"{b['symbol']} ({b['correlation']:.2f})" for b in breaches)
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=(
                f"Correlation of {candidate} exceeds {max_correlation:.2f} with: {breach_desc}"
            ),
            details={
                "candidate": candidate,
                "max_correlation": max_correlation,
                "breaches": breaches,
                "max_observed_correlation": max_observed,
            },
        )

    return RuleResult(
        rule_name=rule_name,
        passed=True,
        reason=(
            f"Correlation check passed for {candidate} "
            f"(max observed: {max_observed:.2f} <= {max_correlation:.2f})"
            if max_observed is not None
            else f"Correlation check passed for {candidate} (no conflicting pairs)"
        ),
        details={
            "candidate": candidate,
            "max_correlation": max_correlation,
            "max_observed_correlation": max_observed,
        },
    )


def check_economic_calendar(
    now: datetime,
    earnings_dates: Sequence[datetime] | None = None,
    fed_dates: Sequence[datetime] | None = None,
    blackout_hours: float = 24.0,
) -> RuleResult:
    """Block trades when an Earnings or Fed decision occurs within [0, blackout_hours] ahead.

    All timestamps are strictly normalized to America/New_York before comparison.
    """
    rule_name = "economic_calendar"
    if not _is_finite(blackout_hours) or blackout_hours < 0.0:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Invalid blackout_hours threshold: {blackout_hours!r}",
            details={"blackout_hours": blackout_hours},
        )

    now_ny = normalize_to_ny(now)
    events: list[tuple[str, datetime]] = []
    if earnings_dates:
        events.extend(("Earnings", dt) for dt in earnings_dates)
    if fed_dates:
        events.extend(("Fed", dt) for dt in fed_dates)

    blocking_events: list[dict[str, Any]] = []
    for event_type, raw_dt in events:
        event_ny = normalize_to_ny(raw_dt)
        delta_hours = (event_ny - now_ny).total_seconds() / 3600.0
        if 0.0 <= delta_hours <= blackout_hours:
            blocking_events.append(
                {
                    "type": event_type,
                    "event_time_ny": event_ny.isoformat(),
                    "hours_until_event": delta_hours,
                }
            )

    if blocking_events:
        summary = "; ".join(
            f"{e['type']} in {e['hours_until_event']:.2f}h ({e['event_time_ny']})"
            for e in blocking_events
        )
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Economic calendar blackout ({blackout_hours:.1f}h): {summary}",
            details={
                "now_ny": now_ny.isoformat(),
                "blackout_hours": blackout_hours,
                "blocking_events": blocking_events,
            },
        )

    return RuleResult(
        rule_name=rule_name,
        passed=True,
        reason=f"No Earnings or Fed events within the next {blackout_hours:.1f}h (America/New_York)",
        details={"now_ny": now_ny.isoformat(), "blackout_hours": blackout_hours},
    )


def check_circuit_breaker(
    daily_pnl: float,
    starting_equity: float,
    max_daily_drawdown_pct: float = 0.03,
) -> RuleResult:
    """Block trades when daily drawdown reaches or exceeds max_daily_drawdown_pct."""
    rule_name = "circuit_breaker"
    if not (
        _is_finite(daily_pnl)
        and _is_finite(starting_equity)
        and _is_finite(max_daily_drawdown_pct)
    ):
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason="Invalid non-finite input for daily circuit breaker check",
            details={
                "daily_pnl": daily_pnl,
                "starting_equity": starting_equity,
                "max_daily_drawdown_pct": max_daily_drawdown_pct,
            },
        )

    if starting_equity <= 0.0:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Starting equity must be strictly positive (got {starting_equity:.2f})",
            details={"starting_equity": starting_equity},
        )

    if max_daily_drawdown_pct <= 0.0:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=f"Invalid max_daily_drawdown_pct threshold: {max_daily_drawdown_pct!r}",
            details={"max_daily_drawdown_pct": max_daily_drawdown_pct},
        )

    drawdown_pct = max(0.0, -daily_pnl / starting_equity)
    if drawdown_pct >= max_daily_drawdown_pct - 1e-12:
        return RuleResult(
            rule_name=rule_name,
            passed=False,
            reason=(
                f"Daily circuit breaker triggered: drawdown {drawdown_pct:.2%} "
                f">= limit {max_daily_drawdown_pct:.2%}"
            ),
            details={
                "daily_pnl": daily_pnl,
                "starting_equity": starting_equity,
                "drawdown_pct": drawdown_pct,
                "max_daily_drawdown_pct": max_daily_drawdown_pct,
            },
        )

    return RuleResult(
        rule_name=rule_name,
        passed=True,
        reason=(
            f"Daily drawdown {drawdown_pct:.2%} is below circuit breaker limit "
            f"{max_daily_drawdown_pct:.2%}"
        ),
        details={
            "daily_pnl": daily_pnl,
            "starting_equity": starting_equity,
            "drawdown_pct": drawdown_pct,
            "max_daily_drawdown_pct": max_daily_drawdown_pct,
        },
    )
