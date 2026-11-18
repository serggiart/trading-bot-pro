"""Unit tests for private/public strategy configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import NY_TIMEZONE, StrategySettings, load_fed_dates, load_strategy_settings


def test_load_strategy_settings_defaults_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STRATEGY_CONFIG_PATH", raising=False)

    settings = load_strategy_settings()
    assert settings == StrategySettings()
    assert settings.tickers == ("AAPL", "MSFT", "NVDA")
    assert settings.atr_stop_multiplier == pytest.approx(2.0)
    assert settings.atr_take_profit_multiplier == pytest.approx(2.0)


def test_load_strategy_settings_from_explicit_json_and_example_file(
    tmp_path: Path,
) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "tickers": [" spy ", "qqq"],
                "max_candidates_for_ai": 3,
                "atr_stop_multiplier": 2.5,
                "atr_take_profit_multiplier": 1.5,
                "trailing_atr_multiplier": 2.5,
                "max_hold_bars": 20,
                "rsi_entry_max": 55.0,
                "min_relative_volume": 1.1,
                "sector_map": {"spy": "Broad Market"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_strategy_settings(custom)
    assert loaded.tickers == ("SPY", "QQQ")
    assert loaded.max_candidates_for_ai == 3
    assert loaded.atr_stop_multiplier == pytest.approx(2.5)
    assert loaded.atr_take_profit_multiplier == pytest.approx(1.5)
    assert loaded.max_hold_bars == 20
    assert loaded.rsi_entry_max == pytest.approx(55.0)
    assert loaded.min_relative_volume == pytest.approx(1.1)
    assert loaded.sector_map == {"SPY": "Broad Market"}

    example_path = Path(__file__).resolve().parents[1] / "config" / "strategy.example.json"
    example_settings = load_strategy_settings(example_path)
    assert example_settings.tickers == ("AAPL", "MSFT", "NVDA")


def test_load_strategy_settings_validation_and_missing_explicit_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_strategy_settings(tmp_path / "missing.json")

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"rsi_entry_max": 120.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="rsi_entry_max"):
        load_strategy_settings(bad_file)

    empty_tickers = tmp_path / "empty_tickers.json"
    empty_tickers.write_text(json.dumps({"tickers": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="tickers"):
        load_strategy_settings(empty_tickers)


def test_load_fed_dates_from_repo_calendar_and_missing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_fed = Path(__file__).resolve().parents[1] / "config" / "fed_calendar.json"
    fed_dates = load_fed_dates(repo_fed)
    assert len(fed_dates) >= 8
    assert all(dt.tzinfo == NY_TIMEZONE for dt in fed_dates)
    assert any(dt.year == 2026 and dt.month == 12 and dt.day == 9 for dt in fed_dates)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FED_CALENDAR_PATH", raising=False)
    assert load_fed_dates() == []

    with pytest.raises(FileNotFoundError):
        load_fed_dates(tmp_path / "missing_fed.json")
