"""Local Parquet candle cache backed by pyarrow with lazy yfinance fallback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf


def _normalize_symbol(symbol: str) -> str:
    """Normalize and validate a ticker symbol for file naming."""
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("symbol must be a non-empty string")
    return cleaned


class ParquetCandleStore:
    """Caches historical OHLCV DataFrames locally as `{SYMBOL}.parquet` files."""

    def __init__(
        self,
        cache_dir: Path = Path("data/candles"),
        ticker_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._ticker_factory = ticker_factory or yf.Ticker

    def _path_for(self, symbol: str) -> Path:
        """Return the local Parquet file path for a normalized ticker symbol."""
        norm_symbol = _normalize_symbol(symbol)
        return self.cache_dir / f"{norm_symbol}.parquet"

    def save_candles(self, symbol: str, df: pd.DataFrame) -> Path:
        """Persist an OHLCV DataFrame to `{SYMBOL}.parquet` using pyarrow."""
        if df is None or df.empty:
            raise ValueError("Cannot save empty OHLCV DataFrame to Parquet")
        path = self._path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, path)
        return path

    def load_candles(self, symbol: str) -> pd.DataFrame | None:
        """Load cached OHLCV candles from `{SYMBOL}.parquet` if present."""
        path = self._path_for(symbol)
        if not path.exists():
            return None
        table = pq.read_table(path)
        return table.to_pandas()

    def get_or_fetch_candles(
        self,
        symbol: str,
        period: str = "1y",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return cached candles when available unless refresh=True, else fetch and cache."""
        norm_symbol = _normalize_symbol(symbol)
        if not refresh:
            cached = self.load_candles(norm_symbol)
            if cached is not None and not cached.empty:
                return cached

        ticker = self._ticker_factory(norm_symbol)
        history = ticker.history(period=period)
        if history is None or history.empty:
            raise ValueError(f"No historical candle data returned for {norm_symbol}")

        self.save_candles(norm_symbol, history)
        return history
