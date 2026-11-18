"""Fail-safe notification clients for trade alerts and Guardian vetoes."""

from src.notifications.telegram import TelegramNotifier

__all__ = ["TelegramNotifier"]
