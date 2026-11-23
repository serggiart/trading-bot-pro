"""Walk-forward self-calibration engine for universe pruning and wave parameter tuning."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import BacktestConfig, BacktestEngine, BacktestMetrics
from src.config import DEFAULT_STRATEGY_CONFIG_PATH, StrategySettings
from src.guardian import RiskGuardian
from src.universe import compute_return_correlation_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TickerCalibrationScore:
    """Standalone walk-forward wave compatibility score for a single symbol."""

    symbol: str
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return_pct: float
    keep: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Outcome of a walk-forward universe and parameter self-calibration run."""

    baseline_metrics: BacktestMetrics
    calibrated_metrics: BacktestMetrics
    active_tickers: tuple[str, ...]
    pruned_tickers: tuple[str, ...]
    ticker_scores: tuple[TickerCalibrationScore, ...]
    recommended_params: dict[str, float]
    baseline_score: float
    calibrated_score: float
    improved: bool


def compute_objective_score(metrics: BacktestMetrics) -> float:
    """Compute a risk-adjusted objective score rewarding return * PF and penalizing drawdown."""
    if metrics.total_trades <= 0:
        return -1.0
    capped_pf = min(float(metrics.profit_factor), 3.0)
    return (
        float(metrics.total_return_pct) * capped_pf
        - 2.0 * float(metrics.max_drawdown_pct)
    )


class StrategySelfCalibrator:
    """Deterministic walk-forward calibrator operating strictly outside RiskGuardian rules."""

    def __init__(
        self,
        min_trades_for_pruning: int = 4,
        min_ticker_profit_factor: float = 0.95,
        min_active_tickers: int = 8,
        max_allowed_drawdown_pct: float = 0.06,
    ) -> None:
        self.min_trades_for_pruning = int(min_trades_for_pruning)
        self.min_ticker_profit_factor = float(min_ticker_profit_factor)
        self.min_active_tickers = int(min_active_tickers)
        self.max_allowed_drawdown_pct = float(max_allowed_drawdown_pct)

    def _compute_live_pf_multiplier(
        self,
        symbol: str,
        audit_store: Any | None,
    ) -> tuple[float, int, float]:
        """Inspect live SQLite BUY/SELL order history to reward live winners and penalize live losers."""
        if audit_store is None or not hasattr(audit_store, "get_orders"):
            return 1.0, 0, 0.0
        try:
            records = list(reversed(audit_store.get_orders(symbol=symbol, limit=40)))
        except Exception:
            return 1.0, 0, 0.0

        last_buy_price: float | None = None
        realized_pnls: list[float] = []
        for rec in records:
            status = str(getattr(rec, "status", "")).lower()
            if status in {"dry_run", "rejected", "canceled", "cancelled"}:
                continue
            side = str(getattr(rec, "side", "")).upper()
            price = float(getattr(rec, "limit_price", 0.0) or 0.0)
            qty = float(getattr(rec, "qty", 0.0) or 0.0)
            if price <= 0.0 or qty <= 0.0:
                continue
            if side == "BUY" and status == "filled":
                last_buy_price = price
            elif side == "SELL" and last_buy_price is not None:
                # Only count actual exits (where sell limit is near or above entry, or filled stop)
                if status in {"filled", "pending_new"}:
                    realized_pnls.append((price - last_buy_price) * qty)
                    last_buy_price = None

        if not realized_pnls:
            return 1.0, 0, 0.0
        net_live_pnl = sum(realized_pnls)
        live_losses = sum(1 for p in realized_pnls if p < 0.0)
        if net_live_pnl < 0.0 and live_losses >= 2:
            return 0.70, len(realized_pnls), net_live_pnl
        if net_live_pnl < 0.0:
            return 0.85, len(realized_pnls), net_live_pnl
        if net_live_pnl > 0.0:
            return 1.10, len(realized_pnls), net_live_pnl
        return 1.0, len(realized_pnls), net_live_pnl

    def evaluate_tickers(
        self,
        market_data: Mapping[str, pd.DataFrame],
        vix_series: pd.Series | None,
        settings: StrategySettings,
        candidate_tickers: Sequence[str] | None = None,
        audit_store: Any | None = None,
    ) -> tuple[list[str], list[str], list[TickerCalibrationScore]]:
        """Evaluate each ticker's standalone wave expectancy and prune chronic chop tickers."""
        symbols = [
            s.strip().upper()
            for s in (candidate_tickers if candidate_tickers is not None else settings.tickers)
            if s.strip() and s.strip().upper() in market_data
        ]
        base_cfg = BacktestConfig(
            initial_capital=10_000.0,
            atr_stop_multiplier=settings.atr_stop_multiplier,
            atr_take_profit_multiplier=settings.atr_take_profit_multiplier,
            max_hold_bars=settings.max_hold_bars,
            rsi_entry_max=settings.rsi_entry_max,
            min_relative_volume=settings.min_relative_volume,
            max_stop_loss_pct=settings.max_stop_loss_pct or 0.06,
            max_ema_extension_atr=settings.max_ema_extension_atr or 1.2,
            breakeven_trigger_pct=settings.breakeven_trigger_pct or 0.06,
            peak_activation_pct=settings.peak_activation_pct or 0.06,
            peak_rsi_threshold=settings.peak_rsi_threshold or 68.0,
        )

        scores: list[TickerCalibrationScore] = []
        for sym in symbols:
            df = market_data.get(sym)
            if df is None or df.empty:
                continue
            m = (
                BacktestEngine(guardian=RiskGuardian(), config=base_cfg)
                .run({sym: df}, vix_series=vix_series, sector_map=settings.sector_map)
                .metrics
            )
            live_mult, live_trades, live_pnl = self._compute_live_pf_multiplier(
                sym, audit_store
            )
            effective_pf = m.profit_factor * live_mult
            total_combined_trades = m.total_trades + live_trades
            should_prune = (
                total_combined_trades >= self.min_trades_for_pruning
                and effective_pf < self.min_ticker_profit_factor
            )
            live_tag = (
                f", live={live_trades} trades ({live_pnl:+.2f} USD)"
                if live_trades > 0
                else ""
            )
            reason = (
                f"Pruned: effective PF {effective_pf:.2f} < {self.min_ticker_profit_factor:.2f} across {total_combined_trades} trades{live_tag}"
                if should_prune
                else f"Active: effective PF {effective_pf:.2f} ({m.total_trades} bt trades, {m.total_return_pct:+.2%}{live_tag})"
            )
            scores.append(
                TickerCalibrationScore(
                    symbol=sym,
                    total_trades=total_combined_trades,
                    win_rate=m.win_rate,
                    profit_factor=effective_pf,
                    total_return_pct=m.total_return_pct,
                    keep=not should_prune,
                    reason=reason,
                )
            )

        active = [s.symbol for s in scores if s.keep]
        pruned = [s.symbol for s in scores if not s.keep]

        # Safety floor: never prune below min_active_tickers (if universe is small in tests)
        if len(active) < min(self.min_active_tickers, len(symbols)):
            ranked = sorted(
                scores,
                key=lambda item: (item.total_return_pct, item.profit_factor),
                reverse=True,
            )
            needed = min(self.min_active_tickers, len(symbols))
            active_set = {s.symbol for s in ranked[:needed]}
            active = [s.symbol for s in scores if s.symbol in active_set]
            pruned = [s.symbol for s in scores if s.symbol not in active_set]

        return active, pruned, scores

    def calibrate(
        self,
        market_data: Mapping[str, pd.DataFrame],
        vix_series: pd.Series | None,
        settings: StrategySettings,
        master_tickers: Sequence[str] | None = None,
        parameter_grid: Sequence[dict[str, float]] | None = None,
        audit_store: Any | None = None,
    ) -> CalibrationReport:
        """Run walk-forward universe pruning + bounded wave parameter optimization."""
        current_symbols = [
            s.strip().upper()
            for s in settings.tickers
            if s.strip().upper() in market_data
        ]
        current_md = {sym: market_data[sym] for sym in current_symbols}
        current_corr = compute_return_correlation_matrix(current_md)

        baseline_cfg = BacktestConfig(
            initial_capital=10_000.0,
            atr_stop_multiplier=settings.atr_stop_multiplier,
            atr_take_profit_multiplier=settings.atr_take_profit_multiplier,
            max_hold_bars=settings.max_hold_bars,
            rsi_entry_max=settings.rsi_entry_max,
            min_relative_volume=settings.min_relative_volume,
            max_stop_loss_pct=settings.max_stop_loss_pct,
            max_ema_extension_atr=settings.max_ema_extension_atr,
            breakeven_trigger_pct=settings.breakeven_trigger_pct,
            peak_activation_pct=settings.peak_activation_pct,
            peak_rsi_threshold=settings.peak_rsi_threshold,
        )
        baseline_metrics = (
            BacktestEngine(guardian=RiskGuardian(), config=baseline_cfg)
            .run(
                current_md,
                vix_series=vix_series,
                sector_map=settings.sector_map,
                correlation_matrix=current_corr,
            )
            .metrics
        )
        baseline_score = compute_objective_score(baseline_metrics)

        pool_tickers = master_tickers if master_tickers is not None else settings.tickers
        active_tickers, pruned_tickers, ticker_scores = self.evaluate_tickers(
            market_data=market_data,
            vix_series=vix_series,
            settings=settings,
            candidate_tickers=pool_tickers,
            audit_store=audit_store,
        )
        active_md = {sym: market_data[sym] for sym in active_tickers if sym in market_data}
        active_corr = compute_return_correlation_matrix(active_md)

        grid = list(parameter_grid) if parameter_grid is not None else [
            {
                "max_stop_loss_pct": 0.06,
                "max_ema_extension_atr": 1.2,
                "breakeven_trigger_pct": 0.06,
                "peak_activation_pct": 0.06,
                "peak_rsi_threshold": 68.0,
            },
            {
                "max_stop_loss_pct": 0.055,
                "max_ema_extension_atr": 1.2,
                "breakeven_trigger_pct": 0.055,
                "peak_activation_pct": 0.055,
                "peak_rsi_threshold": 67.0,
            },
            {
                "max_stop_loss_pct": 0.06,
                "max_ema_extension_atr": 1.2,
                "breakeven_trigger_pct": 0.05,
                "peak_activation_pct": 0.05,
                "peak_rsi_threshold": 66.0,
            },
            {
                "max_stop_loss_pct": 0.06,
                "max_ema_extension_atr": 1.4,
                "breakeven_trigger_pct": 0.06,
                "peak_activation_pct": 0.06,
                "peak_rsi_threshold": 68.0,
            },
        ]

        best_metrics = baseline_metrics
        best_score = baseline_score
        best_params: dict[str, float] = {
            "max_stop_loss_pct": settings.max_stop_loss_pct,
            "max_ema_extension_atr": settings.max_ema_extension_atr,
            "breakeven_trigger_pct": settings.breakeven_trigger_pct,
            "peak_activation_pct": settings.peak_activation_pct,
            "peak_rsi_threshold": settings.peak_rsi_threshold,
        }
        best_active = tuple(current_symbols)
        best_pruned: tuple[str, ...] = ()

        for candidate in grid:
            cfg = BacktestConfig(
                initial_capital=10_000.0,
                atr_stop_multiplier=settings.atr_stop_multiplier,
                atr_take_profit_multiplier=settings.atr_take_profit_multiplier,
                max_hold_bars=settings.max_hold_bars,
                rsi_entry_max=settings.rsi_entry_max,
                min_relative_volume=settings.min_relative_volume,
                max_stop_loss_pct=float(candidate["max_stop_loss_pct"]),
                max_ema_extension_atr=float(candidate["max_ema_extension_atr"]),
                breakeven_trigger_pct=float(candidate["breakeven_trigger_pct"]),
                peak_activation_pct=float(candidate["peak_activation_pct"]),
                peak_rsi_threshold=float(candidate["peak_rsi_threshold"]),
            )
            metrics = (
                BacktestEngine(guardian=RiskGuardian(), config=cfg)
                .run(
                    active_md,
                    vix_series=vix_series,
                    sector_map=settings.sector_map,
                    correlation_matrix=active_corr,
                )
                .metrics
            )
            score = compute_objective_score(metrics)
            if (
                metrics.max_drawdown_pct <= self.max_allowed_drawdown_pct
                and score > best_score
            ):
                best_score = score
                best_metrics = metrics
                best_params = dict(candidate)
                best_active = tuple(active_tickers)
                best_pruned = tuple(pruned_tickers)

        return CalibrationReport(
            baseline_metrics=baseline_metrics,
            calibrated_metrics=best_metrics,
            active_tickers=best_active,
            pruned_tickers=best_pruned,
            ticker_scores=tuple(ticker_scores),
            recommended_params=best_params,
            baseline_score=baseline_score,
            calibrated_score=best_score,
            improved=best_score > baseline_score + 1e-6,
        )

    @staticmethod
    def apply_to_config_file(
        report: CalibrationReport,
        config_path: str | Path | None = None,
    ) -> bool:
        """Persist calibrated active universe and wave parameters to strategy.local.json."""
        if not report.improved:
            return False
        resolved = Path(config_path) if config_path is not None else DEFAULT_STRATEGY_CONFIG_PATH
        raw_data: dict[str, Any] = {}
        if resolved.exists():
            loaded = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw_data = dict(loaded)

        existing_tickers = raw_data.get("tickers", [])
        if "master_tickers" not in raw_data and isinstance(existing_tickers, list) and existing_tickers:
            raw_data["master_tickers"] = list(existing_tickers)

        raw_data["tickers"] = list(report.active_tickers)
        raw_data["pruned_tickers"] = list(report.pruned_tickers)
        for key, val in report.recommended_params.items():
            raw_data[key] = val

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(raw_data, indent=2) + "\n", encoding="utf-8")
        return True
