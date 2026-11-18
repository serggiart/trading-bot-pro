"""Comprehensive unit tests for the deterministic Risk Guardian module."""

from __future__ import annotations

import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.guardian import (
    NY_TZ,
    GuardianConfig,
    MarketContext,
    PortfolioPosition,
    RiskGuardian,
    TradeCandidate,
    calculate_atr_position_size,
    calculate_fractional_kelly,
    check_circuit_breaker,
    check_economic_calendar,
    check_sector_correlation,
    check_vix_regime,
)


# ---------------------------------------------------------------------------
# 1. Dependency Isolation (ZERO AI/LLM imports)
# ---------------------------------------------------------------------------


def test_guardian_has_zero_llm_dependencies() -> None:
    """Verify src/guardian never imports google-genai or any LLM SDK."""
    guardian_dir = Path(__file__).resolve().parents[1] / "src" / "guardian"
    forbidden_modules = ("google.genai", "google-genai", "openai", "anthropic", "vertexai")

    for py_file in guardian_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        assert not alias.name.startswith(forbidden.replace("-", ".")), (
                            f"Forbidden LLM import {alias.name} in {py_file.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in forbidden_modules:
                    assert not node.module.startswith(forbidden.replace("-", ".")), (
                        f"Forbidden LLM import {node.module} in {py_file.name}"
                    )

    assert "google.genai" not in sys.modules


# ---------------------------------------------------------------------------
# 2. Fractional Kelly Sizing Tests
# ---------------------------------------------------------------------------


def test_fractional_kelly_positive_expectancy_and_cap() -> None:
    # f* = 0.60 - (0.40 / 2.0) = 0.40; 0.25 * 0.40 = 0.10
    alloc = calculate_fractional_kelly(
        win_rate=0.60,
        payoff_ratio=2.0,
        kelly_fraction=0.25,
        max_allocation_pct=0.20,
    )
    assert alloc == pytest.approx(0.10)

    # Very high edge should be capped at max_allocation_pct=0.20
    capped = calculate_fractional_kelly(
        win_rate=0.90,
        payoff_ratio=5.0,
        kelly_fraction=0.50,
        max_allocation_pct=0.20,
    )
    assert capped == pytest.approx(0.20)


def test_fractional_kelly_negative_or_zero_expectancy() -> None:
    # Breakeven: 0.50 - 0.50 / 1.0 = 0.0
    assert calculate_fractional_kelly(win_rate=0.50, payoff_ratio=1.0) == 0.0
    # Negative expectancy: 0.30 - 0.70 / 1.5 < 0
    assert calculate_fractional_kelly(win_rate=0.30, payoff_ratio=1.5) == 0.0


@pytest.mark.parametrize(
    ("win_rate", "payoff_ratio", "kelly_fraction", "max_alloc"),
    [
        (0.55, 0.0, 0.25, 0.20),
        (0.55, -1.5, 0.25, 0.20),
        (0.0, 2.0, 0.25, 0.20),
        (-0.1, 2.0, 0.25, 0.20),
        (1.05, 2.0, 0.25, 0.20),
        (0.55, 2.0, 0.0, 0.20),
        (0.55, 2.0, 0.25, 0.0),
        (float("nan"), 2.0, 0.25, 0.20),
        (0.55, float("inf"), 0.25, 0.20),
    ],
)
def test_fractional_kelly_invalid_and_zero_division_inputs(
    win_rate: float,
    payoff_ratio: float,
    kelly_fraction: float,
    max_alloc: float,
) -> None:
    assert calculate_fractional_kelly(win_rate, payoff_ratio, kelly_fraction, max_alloc) == 0.0


# ---------------------------------------------------------------------------
# 3. ATR Position Sizing Tests
# ---------------------------------------------------------------------------


def test_atr_position_size_exact_formula() -> None:
    # Capital=100,000, risk=1% -> $1,000 risk budget
    # ATR=5.0, mult=2.0 -> stop_distance=10.0
    # Raw shares = 1000 / 10 = 100 shares @ $100 = $10,000 (10% <= 20% asset cap)
    res = calculate_atr_position_size(
        capital=100_000.0,
        risk_per_trade=0.01,
        entry_price=100.0,
        atr=5.0,
        atr_multiplier=2.0,
        max_asset_exposure_pct=0.20,
        max_sector_exposure_pct=0.35,
    )
    assert res.shares == 100
    assert res.stop_distance == pytest.approx(10.0)
    assert res.stop_price == pytest.approx(90.0)
    assert res.position_value == pytest.approx(10_000.0)
    assert res.allocation_pct == pytest.approx(0.10)
    assert res.risk_amount == pytest.approx(1_000.0)
    assert res.capped_by is None


def test_atr_position_size_enforces_asset_and_sector_and_kelly_caps() -> None:
    # Raw risk budget = 100,000 * 0.05 = 5,000; stop_distance = 1.0 -> 5,000 shares ($500k)
    # Asset cap 20% = $20,000 (200 shares)
    res_asset = calculate_atr_position_size(
        capital=100_000.0,
        risk_per_trade=0.05,
        entry_price=100.0,
        atr=0.5,
        atr_multiplier=2.0,
        max_asset_exposure_pct=0.20,
        max_sector_exposure_pct=0.35,
    )
    assert res_asset.shares == 200
    assert res_asset.capped_by == "asset_exposure"

    # Sector cap 35% with 25% already used -> only 10% remaining ($10,000 -> 100 shares)
    res_sector = calculate_atr_position_size(
        capital=100_000.0,
        risk_per_trade=0.05,
        entry_price=100.0,
        atr=0.5,
        atr_multiplier=2.0,
        max_asset_exposure_pct=0.20,
        max_sector_exposure_pct=0.35,
        current_sector_exposure_pct=0.25,
    )
    assert res_sector.shares == 100
    assert res_sector.capped_by == "sector_exposure"

    # Sector already at cap -> 0 shares
    res_sector_full = calculate_atr_position_size(
        capital=100_000.0,
        risk_per_trade=0.01,
        entry_price=100.0,
        atr=2.0,
        max_sector_exposure_pct=0.35,
        current_sector_exposure_pct=0.35,
    )
    assert res_sector_full.shares == 0
    assert res_sector_full.capped_by == "sector_exposure"

    # Kelly cap 5% tighter than asset (20%) and sector (35%) -> $5,000 -> 50 shares
    res_kelly = calculate_atr_position_size(
        capital=100_000.0,
        risk_per_trade=0.05,
        entry_price=100.0,
        atr=0.5,
        atr_multiplier=2.0,
        max_asset_exposure_pct=0.20,
        max_sector_exposure_pct=0.35,
        kelly_cap_pct=0.05,
    )
    assert res_kelly.shares == 50
    assert res_kelly.capped_by == "kelly_cap"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capital": 0.0, "risk_per_trade": 0.01, "entry_price": 100.0, "atr": 2.0},
        {"capital": -1000.0, "risk_per_trade": 0.01, "entry_price": 100.0, "atr": 2.0},
        {"capital": 100_000.0, "risk_per_trade": 0.0, "entry_price": 100.0, "atr": 2.0},
        {"capital": 100_000.0, "risk_per_trade": 0.01, "entry_price": 0.0, "atr": 2.0},
        {"capital": 100_000.0, "risk_per_trade": 0.01, "entry_price": 100.0, "atr": 0.0},
        {"capital": 100_000.0, "risk_per_trade": 0.01, "entry_price": 100.0, "atr": -1.0},
        {
            "capital": 100_000.0,
            "risk_per_trade": 0.01,
            "entry_price": 100.0,
            "atr": 2.0,
            "atr_multiplier": 0.0,
        },
        {"capital": float("nan"), "risk_per_trade": 0.01, "entry_price": 100.0, "atr": 2.0},
        {"capital": 100_000.0, "risk_per_trade": 0.01, "entry_price": 100.0, "atr": float("inf")},
        {
            "capital": 100_000.0,
            "risk_per_trade": 0.01,
            "entry_price": 100.0,
            "atr": 2.0,
            "kelly_cap_pct": 0.0,
        },
    ],
)
def test_atr_position_size_safe_zero_division_and_invalid_inputs(kwargs: dict[str, float]) -> None:
    res = calculate_atr_position_size(**kwargs)
    assert res.shares == 0
    assert res.allocation_pct == 0.0
    assert res.position_value == 0.0
    assert res.reason != ""


# ---------------------------------------------------------------------------
# 4. Individual Deterministic Rules Tests
# ---------------------------------------------------------------------------


def test_vix_regime_boundaries_and_invalid_inputs() -> None:
    assert check_vix_regime(18.5).passed is True
    assert check_vix_regime(30.0, max_vix=30.0).passed is True
    assert check_vix_regime(30.01, max_vix=30.0).passed is False
    assert check_vix_regime(-0.1).passed is False
    assert check_vix_regime(float("nan")).passed is False
    assert check_vix_regime(float("inf")).passed is False


def test_sector_correlation_boundaries_symmetric_dict_and_duplicates() -> None:
    # Boundary 0.75 allowed, 0.76 blocked (checking both (a, b) and (b, a))
    assert check_sector_correlation("NVDA", ["AMD"], {("NVDA", "AMD"): 0.75}).passed is True
    assert check_sector_correlation("NVDA", ["AMD"], {("AMD", "NVDA"): 0.75}).passed is True
    assert check_sector_correlation("NVDA", ["AMD"], {("AMD", "NVDA"): 0.76}).passed is False

    # Support dict[str, float] mapping held symbol -> correlation
    assert check_sector_correlation("NVDA", ["MSFT", "JPM"], {"MSFT": 0.75, "JPM": 0.20}).passed is True
    assert check_sector_correlation("NVDA", ["MSFT", "JPM"], {"MSFT": 0.76, "JPM": 0.20}).passed is False

    # Negative correlation (hedging) should pass
    assert check_sector_correlation("TLT", ["SPY"], {"SPY": -0.85}).passed is True

    # Duplicate symbol in portfolio (case-insensitive) must block
    dup = check_sector_correlation("nvda", ["AAPL", "NVDA"], {})
    assert dup.passed is False
    assert "already" in dup.reason.lower()

    # NaN correlation must block
    assert check_sector_correlation("NVDA", ["AMD"], {"AMD": float("nan")}).passed is False


def test_economic_calendar_blackout_boundaries_and_timezones() -> None:
    now_ny = datetime(2026, 5, 1, 10, 0, tzinfo=NY_TZ)

    # Exact 24.0h ahead -> blocked
    earnings_exact_24h = [now_ny + timedelta(hours=24)]
    assert check_economic_calendar(now_ny, earnings_dates=earnings_exact_24h).passed is False

    # < 24h ahead (Fed decision) -> blocked
    fed_12h = [now_ny + timedelta(hours=12)]
    res_fed = check_economic_calendar(now_ny, fed_dates=fed_12h)
    assert res_fed.passed is False
    assert "Fed" in res_fed.reason

    # 0h ahead (happening now) -> blocked
    assert check_economic_calendar(now_ny, earnings_dates=[now_ny]).passed is False

    # > 24h ahead -> allowed
    earnings_25h = [now_ny + timedelta(hours=24, minutes=1)]
    assert check_economic_calendar(now_ny, earnings_dates=earnings_25h).passed is True

    # Past event -> allowed
    past_event = [now_ny - timedelta(hours=2)]
    assert check_economic_calendar(now_ny, earnings_dates=past_event).passed is True

    # Timezone conversion: 2026-05-01 10:00 EDT == 14:00 UTC.
    # Event at 2026-05-02 10:00 UTC == 06:00 EDT (20h ahead) -> blocked!
    now_utc = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
    event_london = datetime(2026, 5, 2, 11, 0, tzinfo=ZoneInfo("Europe/London"))  # 10:00 UTC = 06:00 EDT
    res_tz = check_economic_calendar(now_utc, earnings_dates=[event_london], blackout_hours=24.0)
    assert res_tz.passed is False


def test_circuit_breaker_thresholds_and_edge_cases() -> None:
    # Below 3% drawdown -> allowed
    assert check_circuit_breaker(daily_pnl=-2999.0, starting_equity=100_000.0).passed is True
    # Positive PnL -> allowed
    assert check_circuit_breaker(daily_pnl=1500.0, starting_equity=100_000.0).passed is True
    # Exact 3.0% drawdown -> blocked
    assert check_circuit_breaker(daily_pnl=-3000.0, starting_equity=100_000.0).passed is False
    # Above 3.0% drawdown -> blocked
    assert check_circuit_breaker(daily_pnl=-4500.0, starting_equity=100_000.0).passed is False
    # Invalid equity or NaN -> blocked
    assert check_circuit_breaker(daily_pnl=0.0, starting_equity=0.0).passed is False
    assert check_circuit_breaker(daily_pnl=-100.0, starting_equity=-5000.0).passed is False
    assert check_circuit_breaker(daily_pnl=float("nan"), starting_equity=100_000.0).passed is False


# ---------------------------------------------------------------------------
# 5. Full RiskGuardian Integration Tests
# ---------------------------------------------------------------------------


def test_risk_guardian_approves_and_sizes_valid_candidate() -> None:
    guardian = RiskGuardian(
        GuardianConfig(
            max_vix=30.0,
            max_correlation=0.75,
            blackout_hours=24.0,
            max_daily_drawdown_pct=0.03,
            risk_per_trade_pct=0.01,
            atr_multiplier=2.0,
            kelly_fraction=0.25,
            max_asset_exposure_pct=0.20,
            max_sector_exposure_pct=0.35,
        )
    )
    now_utc = datetime(2026, 6, 15, 14, 30, tzinfo=timezone.utc)
    context = MarketContext(
        now=now_utc,
        vix=19.5,
        starting_equity=100_000.0,
        daily_pnl=500.0,
        portfolio_positions=[
            PortfolioPosition(symbol="MSFT", sector="Technology", market_value=25_000.0),
        ],
        correlation_matrix={("AAPL", "MSFT"): 0.65},
        fed_dates=[now_utc + timedelta(hours=72)],
    )
    candidate = TradeCandidate(
        symbol="aapl",
        sector="Technology",
        entry_price=200.0,
        atr=2.5,
        win_rate=0.60,
        payoff_ratio=2.0,
        earnings_dates=[now_utc + timedelta(days=10)],
    )

    decision = guardian.evaluate(candidate, context)

    assert decision.approved is True
    assert decision.symbol == "AAPL"
    assert decision.timestamp.tzinfo == NY_TZ
    assert decision.veto_reasons == []
    assert len(decision.rule_results) == 4
    assert all(r.passed for r in decision.rule_results)
    assert decision.position_size is not None
    # Capital = 100,500; risk=1% -> $1,005; stop_dist=5.0 -> raw=201 shares ($40,200)
    # Sector remaining = 35% - 25,000/100,500 (~24.8756%) -> remaining ~$10,175 -> 50 shares
    # Kelly cap = 10% -> $10,050 -> 50 shares
    assert decision.recommended_shares == 50
    assert decision.position_size.position_value <= 100_500.0 * 0.10 + 1e-6


def test_risk_guardian_collects_all_veto_reasons_without_short_circuiting() -> None:
    guardian = RiskGuardian()
    now_ny = datetime(2026, 6, 15, 10, 0, tzinfo=NY_TZ)
    context = MarketContext(
        now=now_ny,
        vix=35.0,  # Veto 1: VIX > 30
        starting_equity=100_000.0,
        daily_pnl=-3500.0,  # Veto 2: drawdown 3.5% >= 3%
        portfolio_positions=[
            PortfolioPosition(symbol="MSFT", sector="Technology", market_value=10_000.0),
        ],
        correlation_matrix={("MSFT", "AAPL"): 0.85},  # Veto 3: corr 0.85 > 0.75
        fed_dates=[now_ny + timedelta(hours=6)],  # Veto 4: Fed within 24h
    )
    candidate = TradeCandidate(
        symbol="AAPL",
        sector="Technology",
        entry_price=150.0,
        atr=3.0,
    )

    decision = guardian.evaluate(candidate, context)

    assert decision.approved is False
    assert decision.recommended_shares == 0
    assert decision.position_size is None
    assert len(decision.rule_results) == 4
    assert len(decision.veto_reasons) == 4
    assert all(not r.passed for r in decision.rule_results)


def test_risk_guardian_vetoes_when_kelly_expectancy_is_negative() -> None:
    guardian = RiskGuardian()
    now_ny = datetime(2026, 6, 15, 10, 0, tzinfo=NY_TZ)
    context = MarketContext(
        now=now_ny,
        vix=15.0,
        starting_equity=100_000.0,
        daily_pnl=0.0,
    )
    candidate = TradeCandidate(
        symbol="XYZ",
        sector="Industrials",
        entry_price=50.0,
        atr=1.5,
        win_rate=0.30,
        payoff_ratio=1.2,
    )

    decision = guardian.evaluate(candidate, context)

    assert decision.approved is False
    assert decision.recommended_shares == 0
    assert decision.position_size is not None
    assert decision.position_size.capped_by == "kelly_cap"
    assert len(decision.veto_reasons) == 1
