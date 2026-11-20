"""Deterministic position exit rules: trailing ATR ratchet, flash crash guard, and time exit."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.execution.client import AlpacaExecutionClient
from src.execution.models import (
    ExecutionConfig,
    ExecutionOrderResult,
    ExitEvaluationResult,
    ManagedPosition,
    ensure_ny_datetime,
)
from src.storage import TradingAuditStore

logger = logging.getLogger(__name__)
_STALE_STOP_ERROR_TOKENS = ("404", "not found", "canceled", "cancelled", "expired", "filled", "replaced")


def compute_trailing_stop(
    entry_price: float,
    current_price: float,
    current_stop: float,
    atr_14: float,
    atr_multiplier: float = 2.0,
    highest_price: float | None = None,
) -> float:
    """Compute a monotonic trailing stop that can only ratchet upward."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if current_stop < 0:
        raise ValueError("current_stop must be non-negative")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive")

    if atr_14 <= 0:
        return current_stop

    peak = (
        max(entry_price, current_price)
        if highest_price is None
        else max(entry_price, current_price, highest_price)
    )
    candidate_stop = round(peak - atr_14 * atr_multiplier, 2)
    return max(current_stop, candidate_stop)


def is_flash_crash_triggered(
    entry_price: float,
    current_price: float,
    opened_at: datetime,
    now: datetime,
    window_minutes: float = 5.0,
    drop_pct: float = 0.03,
) -> bool:
    """Return True if price dropped >= drop_pct within window_minutes of entry."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    if not 0 < drop_pct < 1.0:
        raise ValueError("drop_pct must be in (0, 1.0)")

    opened_ny = ensure_ny_datetime(opened_at)
    now_ny = ensure_ny_datetime(now)
    elapsed_seconds = (now_ny - opened_ny).total_seconds()
    if elapsed_seconds < 0 or elapsed_seconds > window_minutes * 60.0:
        return False

    observed_drop = (entry_price - current_price) / entry_price
    return round(observed_drop, 12) >= drop_pct


def is_time_window_expired(
    entry_price: float,
    current_price: float,
    opened_at: datetime,
    now: datetime,
    max_hold_hours: float = 72.0,
    min_target_return_pct: float = 0.01,
) -> bool:
    """Return True when a position exceeded max_hold_hours without hitting min_target_return_pct."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if max_hold_hours <= 0:
        raise ValueError("max_hold_hours must be positive")

    opened_ny = ensure_ny_datetime(opened_at)
    now_ny = ensure_ny_datetime(now)
    held_hours = (now_ny - opened_ny).total_seconds() / 3600.0
    if held_hours < max_hold_hours:
        return False

    current_return = (current_price - entry_price) / entry_price
    return round(current_return, 12) < min_target_return_pct


class PositionManager:
    """Evaluates open positions against deterministic exit and trailing-stop rules."""

    def __init__(
        self,
        execution_client: AlpacaExecutionClient,
        audit_store: TradingAuditStore | None = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        self.execution_client = execution_client
        self.audit_store = (
            audit_store if audit_store is not None else execution_client.audit_store
        )
        self.config = config if config is not None else execution_client.config

    def _mirror_order_to_local_store(
        self,
        order_result: ExecutionOrderResult,
        metadata: dict[str, Any],
        created_at: datetime,
    ) -> None:
        """Persist order in PositionManager's audit_store when distinct from client's store."""
        if (
            self.audit_store is None
            or self.audit_store is self.execution_client.audit_store
        ):
            return
        self.audit_store.log_order(
            client_order_id=order_result.client_order_id,
            symbol=order_result.symbol,
            side=order_result.side,
            qty=order_result.qty,
            order_type=order_result.order_type,
            limit_price=order_result.limit_price,
            stop_price=order_result.stop_price,
            status=order_result.status,
            broker_order_id=order_result.broker_order_id,
            metadata=metadata,
            created_at=created_at,
        )

    def evaluate_and_manage(
        self,
        position: ManagedPosition,
        current_price: float,
        bid_price: float | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> ExitEvaluationResult:
        """Evaluate an active position and execute protective exits or stop ratchets."""
        if current_price <= 0:
            raise ValueError("current_price must be positive")
        if bid_price is not None and bid_price <= 0:
            raise ValueError("bid_price must be positive when provided")

        eval_now = ensure_ny_datetime(now)
        opened_ny = ensure_ny_datetime(position.opened_at)
        reference_price = bid_price if bid_price is not None else current_price
        peak = (
            max(position.entry_price, current_price)
            if position.highest_price is None
            else max(position.entry_price, current_price, position.highest_price)
        )

        # 1. 5-Minute Flash Crash Guard
        if is_flash_crash_triggered(
            entry_price=position.entry_price,
            current_price=current_price,
            opened_at=opened_ny,
            now=eval_now,
            window_minutes=self.config.flash_crash_window_minutes,
            drop_pct=self.config.flash_crash_drop_pct,
        ):
            elapsed_minutes = (eval_now - opened_ny).total_seconds() / 60.0
            drop_pct = (position.entry_price - current_price) / position.entry_price
            reason = (
                f"Flash crash guard triggered: {drop_pct:.2%} drop in "
                f"{elapsed_minutes:.2f} min (limit <= {self.config.flash_crash_window_minutes:.2f} min)"
            )
            anomaly_metadata: dict[str, Any] = {
                "anomaly": "flash_crash",
                "exit_type": "EMERGENCY_EXIT",
                "entry_price": position.entry_price,
                "current_price": current_price,
                "drop_pct": round(drop_pct, 6),
                "elapsed_minutes": round(elapsed_minutes, 4),
                "dry_run": dry_run,
            }
            order_result = self.execution_client.submit_limit_sell(
                symbol=position.symbol,
                qty=position.qty,
                reference_price=reference_price,
                slippage_pct=self.config.emergency_slippage_pct,
                dry_run=dry_run,
                reason=reason,
                metadata=anomaly_metadata,
            )
            self._mirror_order_to_local_store(order_result, anomaly_metadata, eval_now)

            if self.audit_store is not None:
                self.audit_store.log_guardian_decision(
                    symbol=position.symbol,
                    action="EMERGENCY_EXIT",
                    approved=False,
                    veto_reasons=[reason],
                    metadata={
                        **anomaly_metadata,
                        "client_order_id": order_result.client_order_id,
                    },
                    decided_at=eval_now,
                )

            return ExitEvaluationResult(
                symbol=position.symbol,
                action="EMERGENCY_EXIT",
                should_exit=True,
                new_stop_price=position.current_stop_price,
                exit_reason=reason,
                order_result=order_result,
            )

        # 2. Trailing Stop Hit
        if current_price <= position.current_stop_price:
            reason = (
                f"Trailing stop triggered: current_price {current_price:.2f} "
                f"<= stop {position.current_stop_price:.2f}"
            )
            stop_metadata: dict[str, Any] = {
                "exit_type": "TRAILING_STOP_EXIT",
                "stop_price": position.current_stop_price,
                "current_price": current_price,
                "dry_run": dry_run,
            }
            order_result = self.execution_client.submit_limit_sell(
                symbol=position.symbol,
                qty=position.qty,
                reference_price=reference_price,
                slippage_pct=self.config.exit_slippage_pct,
                dry_run=dry_run,
                reason=reason,
                metadata=stop_metadata,
            )
            self._mirror_order_to_local_store(order_result, stop_metadata, eval_now)
            return ExitEvaluationResult(
                symbol=position.symbol,
                action="TRAILING_STOP_EXIT",
                should_exit=True,
                new_stop_price=position.current_stop_price,
                exit_reason=reason,
                order_result=order_result,
            )

        # 2b. Take-Profit Target Exit (protects standalone STOP_LIMIT positions without bracket TP leg)
        if (
            self.config.enable_take_profit_exit
            and self.config.take_profit_atr_multiplier > 0.0
            and position.atr_14 > 0.0
        ):
            target_price = round(
                position.entry_price
                + self.config.take_profit_atr_multiplier * position.atr_14,
                2,
            )
            if current_price >= target_price:
                gain_pct = (current_price - position.entry_price) / position.entry_price
                reason = (
                    f"Take-profit target reached: current_price {current_price:.2f} "
                    f">= target {target_price:.2f} (+{gain_pct:.2%})"
                )
                tp_metadata: dict[str, Any] = {
                    "exit_type": "TAKE_PROFIT_EXIT",
                    "entry_price": position.entry_price,
                    "target_price": target_price,
                    "current_price": current_price,
                    "return_pct": round(gain_pct, 6),
                    "dry_run": dry_run,
                }
                order_result = self.execution_client.submit_limit_sell(
                    symbol=position.symbol,
                    qty=position.qty,
                    reference_price=reference_price,
                    slippage_pct=self.config.exit_slippage_pct,
                    dry_run=dry_run,
                    reason=reason,
                    metadata=tp_metadata,
                )
                self._mirror_order_to_local_store(order_result, tp_metadata, eval_now)
                return ExitEvaluationResult(
                    symbol=position.symbol,
                    action="TAKE_PROFIT_EXIT",
                    should_exit=True,
                    new_stop_price=position.current_stop_price,
                    exit_reason=reason,
                    order_result=order_result,
                )

        # 3. Time-Window Exit (Opportunity Cost Guard)
        if is_time_window_expired(
            entry_price=position.entry_price,
            current_price=current_price,
            opened_at=opened_ny,
            now=eval_now,
            max_hold_hours=self.config.max_hold_hours,
            min_target_return_pct=self.config.min_target_return_pct,
        ):
            held_hours = (eval_now - opened_ny).total_seconds() / 3600.0
            current_return = (current_price - position.entry_price) / position.entry_price
            reason = (
                f"Time window expired after {held_hours:.2f}h with return "
                f"{current_return:.2%} < {self.config.min_target_return_pct:.2%}"
            )
            time_metadata: dict[str, Any] = {
                "exit_type": "TIME_EXIT",
                "entry_price": position.entry_price,
                "current_price": current_price,
                "held_hours": round(held_hours, 4),
                "return_pct": round(current_return, 6),
                "dry_run": dry_run,
            }
            order_result = self.execution_client.submit_limit_sell(
                symbol=position.symbol,
                qty=position.qty,
                reference_price=reference_price,
                slippage_pct=self.config.exit_slippage_pct,
                dry_run=dry_run,
                reason=reason,
                metadata=time_metadata,
            )
            self._mirror_order_to_local_store(order_result, time_metadata, eval_now)
            return ExitEvaluationResult(
                symbol=position.symbol,
                action="TIME_EXIT",
                should_exit=True,
                new_stop_price=position.current_stop_price,
                exit_reason=reason,
                order_result=order_result,
            )

        # 4. Trailing Stop Upward Ratchet
        previous_highest = (
            position.highest_price
            if position.highest_price is not None
            else position.entry_price
        )
        new_stop = compute_trailing_stop(
            entry_price=position.entry_price,
            current_price=current_price,
            current_stop=position.current_stop_price,
            atr_14=position.atr_14,
            atr_multiplier=self.config.trailing_atr_multiplier,
            highest_price=position.highest_price,
        )
        position.highest_price = peak

        if new_stop > position.current_stop_price:
            if position.stop_order_id:
                try:
                    self.execution_client.update_stop_order(
                        stop_order_id=position.stop_order_id,
                        new_stop_price=new_stop,
                        dry_run=dry_run,
                    )
                except Exception as exc:
                    err_text = str(exc).lower()
                    logger.warning(
                        "Failed to replace broker stop %s for %s (%s); keeping local trailing stop %.2f active",
                        position.stop_order_id,
                        position.symbol,
                        exc,
                        new_stop,
                    )
                    if any(token in err_text for token in _STALE_STOP_ERROR_TOKENS):
                        position.stop_order_id = None
            position.current_stop_price = new_stop
            if not dry_run and self.audit_store is not None:
                self.audit_store.upsert_position_state(
                    symbol=position.symbol,
                    qty=position.qty,
                    entry_price=position.entry_price,
                    current_stop_price=new_stop,
                    atr_14=position.atr_14,
                    opened_at=opened_ny,
                    highest_price=peak,
                    stop_order_id=position.stop_order_id,
                    client_order_id=position.client_order_id,
                    updated_at=eval_now,
                )
            return ExitEvaluationResult(
                symbol=position.symbol,
                action="TRAIL_UPDATE",
                should_exit=False,
                new_stop_price=new_stop,
            )

        # 5. Hold (still persist new intraday highs so trailing memory survives restarts)
        if not dry_run and self.audit_store is not None and peak > previous_highest:
            self.audit_store.upsert_position_state(
                symbol=position.symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                current_stop_price=position.current_stop_price,
                atr_14=position.atr_14,
                opened_at=opened_ny,
                highest_price=peak,
                stop_order_id=position.stop_order_id,
                client_order_id=position.client_order_id,
                updated_at=eval_now,
            )
        return ExitEvaluationResult(
            symbol=position.symbol,
            action="HOLD",
            should_exit=False,
            new_stop_price=position.current_stop_price,
        )
