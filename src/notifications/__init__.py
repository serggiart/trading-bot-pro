"""Fail-safe notification adapters (Gmail HTML Email primary + optional Telegram)."""

from src.notifications.email_notifier import CompositeNotifier, EmailNotifier
from src.notifications.telegram import TelegramNotifier

__all__ = ["CompositeNotifier", "EmailNotifier", "TelegramNotifier"]
