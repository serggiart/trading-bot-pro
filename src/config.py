"""Strategy configuration loader separating public defaults from private parameters."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

NY_TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_STRATEGY_CONFIG_PATH = Path("config/strategy.local.json")
DEFAULT_FED_CALENDAR_PATH = Path("config/fed_calendar.json")


@dataclass(frozen=True, slots=True)
class StrategySettings:
    """Generic baseline strategy defaults overridable via private local JSON config."""

    tickers: tuple[str, ...] = ("AAPL", "MSFT", "NVDA")
    max_candidates_for_ai: int = 5
    atr_stop_multiplier: float = 2.0
    atr_take_profit_multiplier: float = 2.0
    trailing_atr_multiplier: float = 2.0
    max_hold_bars: int = 10
    max_hold_hours: float = 72.0
    min_target_return_pct: float = 0.01
    rsi_entry_max: float = 60.0
    min_relative_volume: float = 1.2
    max_spread_pct: float = 0.015
    max_stop_loss_pct: float = 0.0
    max_ema_extension_atr: float = 0.0
    breakeven_trigger_pct: float = 0.0
    peak_activation_pct: float = 0.0
    peak_pullback_pct: float = 0.0
    peak_rsi_threshold: float = 66.0
    confirmation_delay_minutes: float = 2.0
    max_confirmation_delay_minutes: float = 5.0
    default_sector: str = "Technology"
    sector_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_candidates_for_ai <= 0:
            raise ValueError("max_candidates_for_ai must be positive")
        if self.atr_stop_multiplier <= 0.0:
            raise ValueError("atr_stop_multiplier must be positive")
        if self.atr_take_profit_multiplier < 0.0:
            raise ValueError("atr_take_profit_multiplier must be non-negative")
        if self.trailing_atr_multiplier <= 0.0:
            raise ValueError("trailing_atr_multiplier must be positive")
        if self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive")
        if self.max_hold_hours <= 0.0:
            raise ValueError("max_hold_hours must be positive")
        if not 0.0 < self.rsi_entry_max <= 100.0:
            raise ValueError("rsi_entry_max must be in (0, 100]")
        if self.min_relative_volume < 0.0:
            raise ValueError("min_relative_volume must be non-negative")
        if not 0.0 < self.max_spread_pct <= 1.0:
            raise ValueError("max_spread_pct must be in (0, 1]")
        if not 0.0 <= self.max_stop_loss_pct < 1.0:
            raise ValueError("max_stop_loss_pct must be in [0, 1)")
        if self.max_ema_extension_atr < 0.0:
            raise ValueError("max_ema_extension_atr must be non-negative")
        if not 0.0 <= self.breakeven_trigger_pct < 1.0:
            raise ValueError("breakeven_trigger_pct must be in [0, 1)")
        if not 0.0 <= self.peak_activation_pct < 1.0:
            raise ValueError("peak_activation_pct must be in [0, 1)")
        if not 0.0 <= self.peak_pullback_pct < 1.0:
            raise ValueError("peak_pullback_pct must be in [0, 1)")
        if not 0.0 < self.peak_rsi_threshold <= 100.0:
            raise ValueError("peak_rsi_threshold must be in (0, 100]")
        if self.confirmation_delay_minutes < 0.0:
            raise ValueError("confirmation_delay_minutes must be non-negative")
        if self.max_confirmation_delay_minutes < self.confirmation_delay_minutes:
            raise ValueError("max_confirmation_delay_minutes must be >= confirmation_delay_minutes")


def _parse_tickers(raw: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return fallback
    if not isinstance(raw, list):
        raise ValueError("tickers must be a list of ticker symbols")
    cleaned = tuple(str(sym).strip().upper() for sym in raw if str(sym).strip())
    if not cleaned:
        raise ValueError("tickers list cannot be empty")
    return cleaned


def _parse_sector_map(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("sector_map must be a JSON object")
    return {
        str(k).strip().upper(): str(v).strip()
        for k, v in raw.items()
        if str(k).strip() and str(v).strip()
    }


def load_strategy_settings(
    config_path: str | Path | None = None,
) -> StrategySettings:
    """Load private strategy settings if the JSON file exists; otherwise return public defaults."""
    defaults = StrategySettings()
    if config_path is not None:
        resolved = Path(config_path)
        explicit = True
    else:
        env_path = os.getenv("STRATEGY_CONFIG_PATH", "").strip()
        resolved = Path(env_path) if env_path else DEFAULT_STRATEGY_CONFIG_PATH
        explicit = bool(env_path)

    if not resolved.exists():
        if explicit:
            raise FileNotFoundError(f"Strategy configuration file not found: {resolved}")
        return defaults

    raw_data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError(f"Strategy config must be a JSON object: {resolved}")

    return StrategySettings(
        tickers=_parse_tickers(raw_data.get("tickers"), defaults.tickers),
        max_candidates_for_ai=int(
            raw_data.get("max_candidates_for_ai", defaults.max_candidates_for_ai)
        ),
        atr_stop_multiplier=float(
            raw_data.get("atr_stop_multiplier", defaults.atr_stop_multiplier)
        ),
        atr_take_profit_multiplier=float(
            raw_data.get("atr_take_profit_multiplier", defaults.atr_take_profit_multiplier)
        ),
        trailing_atr_multiplier=float(
            raw_data.get("trailing_atr_multiplier", defaults.trailing_atr_multiplier)
        ),
        max_hold_bars=int(raw_data.get("max_hold_bars", defaults.max_hold_bars)),
        max_hold_hours=float(raw_data.get("max_hold_hours", defaults.max_hold_hours)),
        min_target_return_pct=float(
            raw_data.get("min_target_return_pct", defaults.min_target_return_pct)
        ),
        rsi_entry_max=float(raw_data.get("rsi_entry_max", defaults.rsi_entry_max)),
        min_relative_volume=float(
            raw_data.get("min_relative_volume", defaults.min_relative_volume)
        ),
        max_spread_pct=float(raw_data.get("max_spread_pct", defaults.max_spread_pct)),
        max_stop_loss_pct=float(
            raw_data.get("max_stop_loss_pct", defaults.max_stop_loss_pct)
        ),
        max_ema_extension_atr=float(
            raw_data.get("max_ema_extension_atr", defaults.max_ema_extension_atr)
        ),
        breakeven_trigger_pct=float(
            raw_data.get("breakeven_trigger_pct", defaults.breakeven_trigger_pct)
        ),
        peak_activation_pct=float(
            raw_data.get("peak_activation_pct", defaults.peak_activation_pct)
        ),
        peak_pullback_pct=float(
            raw_data.get("peak_pullback_pct", defaults.peak_pullback_pct)
        ),
        peak_rsi_threshold=float(
            raw_data.get("peak_rsi_threshold", defaults.peak_rsi_threshold)
        ),
        confirmation_delay_minutes=float(
            raw_data.get("confirmation_delay_minutes", defaults.confirmation_delay_minutes)
        ),
        max_confirmation_delay_minutes=float(
            raw_data.get(
                "max_confirmation_delay_minutes", defaults.max_confirmation_delay_minutes
            )
        ),
        default_sector=str(raw_data.get("default_sector", defaults.default_sector)).strip()
        or defaults.default_sector,
        sector_map=_parse_sector_map(raw_data.get("sector_map")),
    )


def load_fed_dates(calendar_path: str | Path | None = None) -> list[datetime]:
    """Load scheduled FOMC decision timestamps normalized to America/New_York."""
    if calendar_path is not None:
        resolved = Path(calendar_path)
        explicit = True
    else:
        env_path = os.getenv("FED_CALENDAR_PATH", "").strip()
        resolved = Path(env_path) if env_path else DEFAULT_FED_CALENDAR_PATH
        explicit = bool(env_path)

    if not resolved.exists():
        if explicit:
            raise FileNotFoundError(f"Fed calendar file not found: {resolved}")
        return []

    raw_data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError(f"Fed calendar must be a JSON object: {resolved}")

    raw_dates = raw_data.get("fomc_decisions", [])
    if not isinstance(raw_dates, list):
        raise ValueError("fomc_decisions must be a list of ISO timestamps")

    parsed: list[datetime] = []
    for item in raw_dates:
        text = str(item).strip()
        if not text:
            continue
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NY_TIMEZONE)
        else:
            dt = dt.astimezone(NY_TIMEZONE)
        parsed.append(dt)
    return parsed
