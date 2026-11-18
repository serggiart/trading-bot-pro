"""Deterministic capital allocation using Fractional Kelly and ATR sizing."""

from __future__ import annotations

import math

from src.guardian.models import PositionSizeResult


def _is_finite(value: float) -> bool:
    """Return True if value is a finite real number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _zero_size_result(
    reason: str,
    *,
    stop_distance: float = 0.0,
    stop_price: float = 0.0,
    capped_by: str | None = None,
) -> PositionSizeResult:
    """Construct a zero-share PositionSizeResult with an explicit reason."""
    return PositionSizeResult(
        shares=0,
        allocation_pct=0.0,
        position_value=0.0,
        stop_distance=stop_distance,
        stop_price=stop_price,
        risk_amount=0.0,
        capped_by=capped_by,
        reason=reason,
    )


def calculate_fractional_kelly(
    win_rate: float,
    payoff_ratio: float,
    kelly_fraction: float = 0.25,
    max_allocation_pct: float = 0.20,
) -> float:
    """Calculate fractional Kelly portfolio allocation fraction in [0.0, max_allocation_pct].

    Full Kelly: f* = win_rate - (1 - win_rate) / payoff_ratio.
    Returns 0.0 for non-positive, invalid, non-finite, or negative-expectancy inputs.
    """
    if not (
        _is_finite(win_rate)
        and _is_finite(payoff_ratio)
        and _is_finite(kelly_fraction)
        and _is_finite(max_allocation_pct)
    ):
        return 0.0

    if (
        win_rate <= 0.0
        or win_rate > 1.0
        or payoff_ratio <= 0.0
        or kelly_fraction <= 0.0
        or max_allocation_pct <= 0.0
    ):
        return 0.0

    full_kelly = win_rate - ((1.0 - win_rate) / payoff_ratio)
    if full_kelly <= 0.0:
        return 0.0

    fractional = full_kelly * kelly_fraction
    return min(fractional, max_allocation_pct)


def calculate_atr_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0,
    max_asset_exposure_pct: float = 0.20,
    max_sector_exposure_pct: float = 0.35,
    current_sector_exposure_pct: float = 0.0,
    kelly_cap_pct: float | None = None,
) -> PositionSizeResult:
    """Calculate deterministic position size via ATR risk budget and exposure caps.

    Core formula:
        stop_distance = atr * atr_multiplier
        raw_shares = (capital * risk_per_trade) / stop_distance
    """
    required_numeric = (
        capital,
        risk_per_trade,
        entry_price,
        atr,
        atr_multiplier,
        max_asset_exposure_pct,
        max_sector_exposure_pct,
        current_sector_exposure_pct,
    )
    if not all(_is_finite(v) for v in required_numeric):
        return _zero_size_result("Invalid non-finite numeric input for position sizing")

    if kelly_cap_pct is not None and not _is_finite(kelly_cap_pct):
        return _zero_size_result("Invalid non-finite kelly_cap_pct")

    if capital <= 0.0:
        return _zero_size_result("Capital must be strictly positive")
    if risk_per_trade <= 0.0:
        return _zero_size_result("risk_per_trade must be strictly positive")
    if entry_price <= 0.0:
        return _zero_size_result("entry_price must be strictly positive")
    if atr <= 0.0:
        return _zero_size_result("ATR must be strictly positive to compute stop distance")
    if atr_multiplier <= 0.0:
        return _zero_size_result("atr_multiplier must be strictly positive")
    if max_asset_exposure_pct <= 0.0:
        return _zero_size_result("max_asset_exposure_pct must be strictly positive")
    if max_sector_exposure_pct <= 0.0:
        return _zero_size_result("max_sector_exposure_pct must be strictly positive")
    if current_sector_exposure_pct < 0.0:
        return _zero_size_result("current_sector_exposure_pct cannot be negative")

    stop_distance = atr * atr_multiplier
    stop_price = max(0.0, entry_price - stop_distance)

    if kelly_cap_pct is not None and kelly_cap_pct <= 0.0:
        return _zero_size_result(
            "Kelly cap is zero or negative due to non-positive expectancy",
            stop_distance=stop_distance,
            stop_price=stop_price,
            capped_by="kelly_cap",
        )

    remaining_sector_pct = max_sector_exposure_pct - current_sector_exposure_pct
    if remaining_sector_pct <= 1e-12:
        return _zero_size_result(
            f"Sector exposure limit reached ({current_sector_exposure_pct:.2%} >= {max_sector_exposure_pct:.2%})",
            stop_distance=stop_distance,
            stop_price=stop_price,
            capped_by="sector_exposure",
        )

    # Tamaño = (Capital * Riesgo_Operacion) / Distancia_Stop_Loss
    risk_budget = capital * risk_per_trade
    raw_shares = risk_budget / stop_distance
    raw_position_value = raw_shares * entry_price

    caps: list[tuple[float, str]] = [
        (capital * max_asset_exposure_pct, "asset_exposure"),
        (capital * remaining_sector_pct, "sector_exposure"),
    ]
    if kelly_cap_pct is not None:
        caps.append((capital * kelly_cap_pct, "kelly_cap"))

    tightest_cap_value, tightest_cap_name = min(caps, key=lambda item: item[0])

    capped_by: str | None = None
    target_value = raw_position_value
    if raw_position_value > tightest_cap_value + 1e-9:
        target_value = tightest_cap_value
        capped_by = tightest_cap_name

    # Floor shares with tiny epsilon to avoid IEEE-754 underflow on exact multiples.
    shares = math.floor((target_value / entry_price) + 1e-9)
    if shares <= 0:
        return _zero_size_result(
            "Calculated position size rounds to 0 shares under capital/exposure constraints",
            stop_distance=stop_distance,
            stop_price=stop_price,
            capped_by=capped_by,
        )

    position_value = shares * entry_price
    allocation_pct = position_value / capital
    actual_risk_amount = shares * stop_distance
    reason = (
        f"Sized {shares} shares ({allocation_pct:.2%} of capital, capped by {capped_by})"
        if capped_by
        else f"Sized {shares} shares ({allocation_pct:.2%} of capital) by ATR risk budget"
    )

    return PositionSizeResult(
        shares=shares,
        allocation_pct=allocation_pct,
        position_value=position_value,
        stop_distance=stop_distance,
        stop_price=stop_price,
        risk_amount=actual_risk_amount,
        capped_by=capped_by,
        reason=reason,
    )
