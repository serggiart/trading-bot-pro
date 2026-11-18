"""Alpaca limit-only execution client with SQLite audit logging."""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.execution.models import ExecutionConfig, ExecutionOrderResult, ensure_ny_datetime
from src.storage import TradingAuditStore


def _resolve_paper_mode(
    paper: bool | None,
    config: ExecutionConfig | None,
) -> bool:
    """Resolve paper-trading mode defaulting to True unless explicitly disabled."""
    if paper is not None:
        return bool(paper)
    env_value = os.getenv("ALPACA_PAPER")
    if env_value is not None and env_value.strip() != "":
        return env_value.strip().lower() not in {"false", "0"}
    if config is not None:
        return bool(config.paper)
    return True


def _extract_broker_status(broker_order: Any, default: str = "submitted") -> str:
    """Normalize an Alpaca order status enum or string to lowercase."""
    raw_status = getattr(broker_order, "status", None)
    if raw_status is None:
        return default
    value = getattr(raw_status, "value", raw_status)
    text = str(value).strip().lower()
    return text or default


def _generate_client_order_id(prefix: str, symbol: str) -> str:
    """Build a unique client_order_id stamped in America/New_York."""
    timestamp = ensure_ny_datetime().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{symbol}-{timestamp}-{uuid.uuid4().hex[:8]}"


def calculate_limit_buy_price(
    ask_price: float,
    slippage_tolerance_pct: float = 0.001,
) -> float:
    """Compute a capped limit buy price above the Ask to control entry slippage."""
    if ask_price <= 0:
        raise ValueError("ask_price must be positive")
    if slippage_tolerance_pct < 0:
        raise ValueError("slippage_tolerance_pct must be non-negative")
    return round(ask_price * (1.0 + slippage_tolerance_pct), 2)


def calculate_limit_sell_price(
    reference_price: float,
    slippage_pct: float = 0.003,
) -> float:
    """Compute a protective limit sell price below the reference/bid price."""
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if not 0 <= slippage_pct < 1.0:
        raise ValueError("slippage_pct must be in [0, 1.0)")
    return max(0.01, round(reference_price * (1.0 - slippage_pct), 2))


class AlpacaExecutionClient:
    """Limit-order execution wrapper around Alpaca's TradingClient."""

    def __init__(
        self,
        audit_store: TradingAuditStore | None = None,
        trading_client: Any | None = None,
        config: ExecutionConfig | None = None,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
    ) -> None:
        base_config = config or ExecutionConfig()
        resolved_paper = _resolve_paper_mode(paper, config)
        self.config = replace(base_config, paper=resolved_paper)
        self.paper = resolved_paper
        self.audit_store = audit_store

        self._api_key = api_key if api_key is not None else os.getenv("ALPACA_API_KEY")
        self._secret_key = (
            secret_key if secret_key is not None else os.getenv("ALPACA_SECRET_KEY")
        )

        if trading_client is not None:
            self._trading_client: Any | None = trading_client
        elif self._api_key and self._secret_key:
            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self.paper,
            )
        else:
            self._trading_client = None

    @property
    def trading_client(self) -> Any | None:
        """Return the underlying Alpaca TradingClient instance if initialized."""
        return self._trading_client

    def _require_trading_client(self) -> Any:
        """Return or initialize the broker TradingClient for live/paper submission."""
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self.paper,
            )
        return self._trading_client

    def _log_audit_order(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        stop_price: float | None,
        status: str,
        broker_order_id: str | None,
        metadata: dict[str, Any],
    ) -> None:
        """Record an order entry in SQLite audit store when configured."""
        if self.audit_store is None:
            return
        self.audit_store.log_order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="limit",
            limit_price=limit_price,
            stop_price=stop_price,
            status=status,
            broker_order_id=broker_order_id,
            metadata=metadata,
            created_at=ensure_ny_datetime(),
        )

    def submit_limit_buy(
        self,
        symbol: str,
        qty: float,
        ask_price: float,
        stop_loss_price: float,
        take_profit_price: float | None = None,
        client_order_id: str | None = None,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionOrderResult:
        """Submit an OTO or Bracket limit buy order with a mandatory attached stop-loss."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be a non-empty string")
        if qty <= 0:
            raise ValueError("qty must be positive")
        if ask_price <= 0:
            raise ValueError("ask_price must be positive")

        limit_price = calculate_limit_buy_price(
            ask_price=ask_price,
            slippage_tolerance_pct=self.config.slippage_tolerance_pct,
        )
        rounded_stop = round(stop_loss_price, 2)
        if not (0 < stop_loss_price < limit_price and 0 < rounded_stop < limit_price):
            raise ValueError(
                f"stop_loss_price ({stop_loss_price}) must satisfy 0 < stop_loss_price < limit_price ({limit_price})"
            )

        stop_loss = StopLossRequest(stop_price=rounded_stop)
        resolved_client_id = (
            client_order_id.strip()
            if client_order_id is not None and client_order_id.strip()
            else _generate_client_order_id("buy", normalized_symbol)
        )

        if take_profit_price is not None:
            rounded_tp = round(take_profit_price, 2)
            if rounded_tp <= limit_price:
                raise ValueError(
                    f"take_profit_price ({take_profit_price}) must be greater than limit_price ({limit_price})"
                )
            order_class = OrderClass.BRACKET
            order_request = LimitOrderRequest(
                symbol=normalized_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                order_class=order_class,
                stop_loss=stop_loss,
                take_profit=TakeProfitRequest(limit_price=rounded_tp),
                client_order_id=resolved_client_id,
            )
        else:
            rounded_tp = None
            order_class = OrderClass.OTO
            order_request = LimitOrderRequest(
                symbol=normalized_symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                order_class=order_class,
                stop_loss=stop_loss,
                client_order_id=resolved_client_id,
            )

        audit_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        audit_metadata.update(
            {
                "paper": self.paper,
                "dry_run": dry_run,
                "order_class": order_class.value,
                "ask_price": ask_price,
            }
        )
        if rounded_tp is not None:
            audit_metadata["take_profit_price"] = rounded_tp

        if dry_run:
            self._log_audit_order(
                client_order_id=resolved_client_id,
                symbol=normalized_symbol,
                side="BUY",
                qty=qty,
                limit_price=limit_price,
                stop_price=rounded_stop,
                status="dry_run",
                broker_order_id=None,
                metadata=audit_metadata,
            )
            return ExecutionOrderResult(
                submitted=False,
                client_order_id=resolved_client_id,
                broker_order_id=None,
                symbol=normalized_symbol,
                side="BUY",
                qty=qty,
                limit_price=limit_price,
                stop_price=rounded_stop,
                order_type="limit",
                status="dry_run",
                reason="Dry run enabled; order not sent to broker",
            )

        broker_order = self._require_trading_client().submit_order(
            order_data=order_request
        )
        raw_broker_id = getattr(broker_order, "id", None)
        broker_order_id = str(raw_broker_id) if raw_broker_id is not None else None
        status = _extract_broker_status(broker_order, default="submitted")

        self._log_audit_order(
            client_order_id=resolved_client_id,
            symbol=normalized_symbol,
            side="BUY",
            qty=qty,
            limit_price=limit_price,
            stop_price=rounded_stop,
            status=status,
            broker_order_id=broker_order_id,
            metadata=audit_metadata,
        )
        return ExecutionOrderResult(
            submitted=True,
            client_order_id=resolved_client_id,
            broker_order_id=broker_order_id,
            symbol=normalized_symbol,
            side="BUY",
            qty=qty,
            limit_price=limit_price,
            stop_price=rounded_stop,
            order_type="limit",
            status=status,
            reason="",
        )

    def submit_limit_sell(
        self,
        symbol: str,
        qty: float,
        reference_price: float,
        slippage_pct: float | None = None,
        client_order_id: str | None = None,
        dry_run: bool = False,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionOrderResult:
        """Submit a LimitOrderRequest sell order with protective downward slippage collar."""
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be a non-empty string")
        if qty <= 0:
            raise ValueError("qty must be positive")
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")

        effective_slippage = (
            self.config.exit_slippage_pct if slippage_pct is None else slippage_pct
        )
        limit_price = calculate_limit_sell_price(
            reference_price=reference_price,
            slippage_pct=effective_slippage,
        )
        resolved_client_id = (
            client_order_id.strip()
            if client_order_id is not None and client_order_id.strip()
            else _generate_client_order_id("sell", normalized_symbol)
        )

        order_request = LimitOrderRequest(
            symbol=normalized_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=resolved_client_id,
        )

        audit_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        audit_metadata.update(
            {
                "paper": self.paper,
                "dry_run": dry_run,
                "reference_price": reference_price,
                "slippage_pct": effective_slippage,
            }
        )
        if reason:
            audit_metadata["reason"] = reason

        if dry_run:
            self._log_audit_order(
                client_order_id=resolved_client_id,
                symbol=normalized_symbol,
                side="SELL",
                qty=qty,
                limit_price=limit_price,
                stop_price=None,
                status="dry_run",
                broker_order_id=None,
                metadata=audit_metadata,
            )
            return ExecutionOrderResult(
                submitted=False,
                client_order_id=resolved_client_id,
                broker_order_id=None,
                symbol=normalized_symbol,
                side="SELL",
                qty=qty,
                limit_price=limit_price,
                stop_price=None,
                order_type="limit",
                status="dry_run",
                reason=reason or "Dry run enabled; sell order not sent to broker",
            )

        broker_order = self._require_trading_client().submit_order(
            order_data=order_request
        )
        raw_broker_id = getattr(broker_order, "id", None)
        broker_order_id = str(raw_broker_id) if raw_broker_id is not None else None
        status = _extract_broker_status(broker_order, default="submitted")

        self._log_audit_order(
            client_order_id=resolved_client_id,
            symbol=normalized_symbol,
            side="SELL",
            qty=qty,
            limit_price=limit_price,
            stop_price=None,
            status=status,
            broker_order_id=broker_order_id,
            metadata=audit_metadata,
        )
        return ExecutionOrderResult(
            submitted=True,
            client_order_id=resolved_client_id,
            broker_order_id=broker_order_id,
            symbol=normalized_symbol,
            side="SELL",
            qty=qty,
            limit_price=limit_price,
            stop_price=None,
            order_type="limit",
            status=status,
            reason=reason,
        )

    def update_stop_order(
        self,
        stop_order_id: str,
        new_stop_price: float,
        dry_run: bool = False,
    ) -> bool:
        """Replace an active stop order price on the broker when not in dry_run mode."""
        cleaned_order_id = stop_order_id.strip()
        if not cleaned_order_id:
            raise ValueError("stop_order_id must be a non-empty string")
        if new_stop_price <= 0:
            raise ValueError("new_stop_price must be positive")

        if dry_run:
            return False

        replace_request = ReplaceOrderRequest(stop_price=round(new_stop_price, 2))
        self._require_trading_client().replace_order_by_id(
            order_id=cleaned_order_id,
            order_data=replace_request,
        )
        return True
