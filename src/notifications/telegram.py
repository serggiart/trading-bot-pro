"""Fail-safe asynchronous Telegram Bot API notifier."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Fail-safe Telegram notifier using httpx.AsyncClient.

    Never raises exceptions or halts trading execution if credentials are
    missing or Telegram API calls fail.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 10.0,
        base_url: str = "https://api.telegram.org",
        http_post: Callable[..., Any] | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        raw_token = os.getenv("TELEGRAM_BOT_TOKEN", "") if bot_token is None else bot_token
        raw_chat_id = os.getenv("TELEGRAM_CHAT_ID", "") if chat_id is None else chat_id
        self.bot_token: str = raw_token.strip()
        self.chat_id: str = raw_chat_id.strip()
        self.timeout: float = float(timeout)
        self.base_url: str = base_url.rstrip("/")
        self._http_post = http_post
        self._client_factory = client_factory

    @property
    def is_configured(self) -> bool:
        """Return True when both bot_token and chat_id are non-empty."""
        return bool(self.bot_token and self.chat_id)

    async def send_message_async(self, text: str) -> bool:
        """Send a Telegram message asynchronously, returning False on any failure."""
        if not self.is_configured:
            logger.warning(
                "Telegram notification skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty"
            )
            return False

        cleaned_text = text.strip()
        if not cleaned_text:
            logger.warning("Telegram notification skipped: message text is empty")
            return False

        url = f"{self.base_url}/bot{self.bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": cleaned_text,
            "disable_web_page_preview": True,
        }

        try:
            if self._http_post is not None:
                response = self._http_post(url, json=payload, timeout=self.timeout)
                if inspect.isawaitable(response):
                    response = await response
            else:
                factory = self._client_factory or httpx.AsyncClient
                async with factory(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)

            if hasattr(response, "raise_for_status"):
                response.raise_for_status()

            status_code = getattr(response, "status_code", 200)
            if isinstance(status_code, int) and status_code >= 400:
                logger.warning("Telegram API returned HTTP status %s", status_code)
                return False

            if hasattr(response, "json"):
                try:
                    body = response.json()
                except Exception:
                    body = None
                if isinstance(body, dict) and body.get("ok") is False:
                    logger.warning("Telegram API returned ok=False: %s", body)
                    return False

            return True
        except Exception as exc:
            logger.warning("Telegram notification failed (fail-safe): %s", exc)
            return False

    def send_message(self, text: str) -> bool:
        """Synchronous fail-safe wrapper around send_message_async."""
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.send_message_async(text))

            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, self.send_message_async(text)).result()
        except Exception as exc:
            logger.warning("Telegram sync notification failed (fail-safe): %s", exc)
            return False

    def notify_trade_executed(
        self,
        symbol: str,
        side: str,
        qty: float | int,
        limit_price: float,
        stop_price: float | None = None,
        dry_run: bool = False,
    ) -> bool:
        """Send an alert when a Limit order is submitted or simulated."""
        prefix = "[DRY-RUN] " if dry_run else ""
        stop_line = f"\nStop Loss: ${stop_price:,.2f}" if stop_price is not None else ""
        message = (
            f"{prefix}TRADE EXECUTED (LIMIT)\n"
            f"Symbol: {symbol.strip().upper()}\n"
            f"Side: {side.strip().upper()}\n"
            f"Qty: {qty}\n"
            f"Limit Price: ${limit_price:,.2f}"
            f"{stop_line}"
        )
        return self.send_message(message)

    def notify_position_closed(
        self,
        symbol: str,
        action: str,
        qty: float | int,
        limit_price: float | None,
        reason: str,
    ) -> bool:
        """Send an alert when an open position is exited by PositionManager."""
        price_str = f"${limit_price:,.2f}" if limit_price is not None else "N/A"
        message = (
            f"POSITION CLOSED ({action.strip().upper()})\n"
            f"Symbol: {symbol.strip().upper()}\n"
            f"Qty: {qty}\n"
            f"Limit Price: {price_str}\n"
            f"Reason: {reason}"
        )
        return self.send_message(message)

    def notify_guardian_veto(
        self,
        symbol: str,
        veto_reasons: Sequence[str],
        vix: float | None = None,
    ) -> bool:
        """Send an alert when the deterministic RiskGuardian vetoes a trade."""
        reasons_text = "\n".join(f"- {reason}" for reason in veto_reasons) or "-Unspecified veto"
        vix_line = f"\nVIX: {vix:.2f}" if vix is not None else ""
        message = (
            f"GUARDIAN VETO\n"
            f"Symbol: {symbol.strip().upper()}"
            f"{vix_line}\n"
            f"Reasons:\n{reasons_text}"
        )
        return self.send_message(message)

    def notify_circuit_breaker(
        self,
        drawdown_pct: float,
        starting_equity: float,
        daily_pnl: float,
    ) -> bool:
        """Send a high-priority alert when the daily drawdown circuit breaker triggers."""
        message = (
            f"CIRCUIT BREAKER TRIGGERED\n"
            f"Daily Drawdown: {drawdown_pct:.2%}\n"
            f"Starting Equity: ${starting_equity:,.2f}\n"
            f"Daily PnL: ${daily_pnl:,.2f}\n"
            f"Trading halted for the remainder of the session."
        )
        return self.send_message(message)
