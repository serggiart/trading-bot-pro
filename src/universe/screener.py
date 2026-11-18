"""Stage-1 Universe Screener combining liquidity, cross-price, trend, and fundamental rules."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.universe.fmp import FmpClient
from src.universe.market_data import MarketDataClient
from src.universe.models import UniverseConfig, UniverseScreeningResult


def _compute_setup_score(result: UniverseScreeningResult) -> float:
    """Score a passed screening result by EMA(20) trend, relative volume, and RSI sweet spot."""
    ind = result.indicators
    if ind is None:
        return 0.0

    score = 0.0
    if ind.ema_20 > 0.0 and ind.current_price > ind.ema_20:
        score += 2.0

    if ind.avg_volume_20d > 0.0 and ind.latest_volume > 0.0:
        rel_vol = ind.latest_volume / ind.avg_volume_20d
        score += min(rel_vol, 3.0)

    if 30.0 <= ind.rsi_14 <= 65.0:
        score += 1.5
    elif ind.rsi_14 < 30.0:
        score += 0.75

    return score


def select_top_candidates(
    results: Sequence[UniverseScreeningResult],
    max_candidates: int = 5,
) -> list[UniverseScreeningResult]:
    """Filter passing screening results and rank deterministically by technical setup score."""
    if max_candidates <= 0:
        return []
    passing = [(idx, r) for idx, r in enumerate(results) if r.passed]
    passing.sort(key=lambda item: (-_compute_setup_score(item[1]), item[0]))
    return [r for _, r in passing[:max_candidates]]


class UniverseScreener:
    """Deterministic Stage-1 screener for equity universe eligibility."""

    def __init__(
        self,
        market_data: MarketDataClient,
        fmp_client: FmpClient,
        config: UniverseConfig | None = None,
    ) -> None:
        self.market_data = market_data
        self.fmp_client = fmp_client
        self.config = config or UniverseConfig()

    def screen_symbol(
        self,
        symbol: str,
        now: datetime | None = None,
        vix: float | None = None,
    ) -> UniverseScreeningResult:
        """Evaluate a single ticker against all Stage-1 universe screening rules."""
        norm_symbol = symbol.strip().upper()
        if not norm_symbol:
            raise ValueError("symbol must be a non-empty string")

        quote = self.market_data.get_quote(norm_symbol)

        if type(self.market_data) is MarketDataClient:
            history = self.market_data.get_historical_ohlcv(norm_symbol, period="1y")
            validation = self.market_data.validate_price(
                norm_symbol,
                alpaca_price=quote.price,
                sigma_threshold=self.config.outlier_sigma_threshold,
                fallback_pct_threshold=self.config.fallback_pct_threshold,
                history=history,
            )
            indicators = self.market_data.get_technical_indicators(
                norm_symbol,
                vix=vix,
                sma_period=self.config.sma_period,
                atr_period=self.config.atr_period,
                history=history,
            )
        else:
            validation = self.market_data.validate_price(
                norm_symbol,
                alpaca_price=quote.price,
                sigma_threshold=self.config.outlier_sigma_threshold,
                fallback_pct_threshold=self.config.fallback_pct_threshold,
            )
            indicators = self.market_data.get_technical_indicators(
                norm_symbol,
                vix=vix,
                sma_period=self.config.sma_period,
                atr_period=self.config.atr_period,
            )

        fundamentals = self.fmp_client.get_fundamentals(norm_symbol, now=now)

        rejection_reasons: list[str] = []

        if not validation.is_valid:
            rejection_reasons.append(
                validation.alert_reason
                or f"Cross-price discrepancy z={validation.z_score:.2f} exceeds {self.config.outlier_sigma_threshold:.2f}σ"
            )

        if quote.bid_price <= 0.0 or quote.ask_price <= 0.0:
            rejection_reasons.append("Invalid or missing bid/ask quote")
        elif quote.spread_pct > self.config.max_spread_pct:
            rejection_reasons.append(
                f"Excessive bid/ask spread: {quote.spread_pct:.4%} > {self.config.max_spread_pct:.4%}"
            )

        avg_vol_20d = float(getattr(indicators, "avg_volume_20d", 0.0) or 0.0)
        if quote.volume < self.config.min_iex_volume:
            rejection_reasons.append(
                f"Insufficient liquidity volume (IEX): {quote.volume:,.0f} < {self.config.min_iex_volume:,.0f}"
            )
        elif avg_vol_20d > 0.0 and avg_vol_20d < self.config.min_avg_volume:
            rejection_reasons.append(
                f"Insufficient liquidity volume (20d avg): {avg_vol_20d:,.0f} < {self.config.min_avg_volume:,.0f}"
            )

        if fundamentals.free_cash_flow <= self.config.min_free_cash_flow:
            rejection_reasons.append(
                f"Negative or insufficient free cash flow: {fundamentals.free_cash_flow:,.2f} <= {self.config.min_free_cash_flow:,.2f}"
            )

        if fundamentals.debt_to_equity > self.config.max_debt_to_equity:
            rejection_reasons.append(
                f"Excessive debt-to-equity ratio: {fundamentals.debt_to_equity:.2f} > {self.config.max_debt_to_equity:.2f}"
            )

        if indicators.current_price < indicators.sma_200:
            rejection_reasons.append(
                f"Downtrend: price {indicators.current_price:.2f} < SMA({self.config.sma_period}) {indicators.sma_200:.2f}"
            )

        return UniverseScreeningResult(
            symbol=norm_symbol,
            passed=len(rejection_reasons) == 0,
            rejection_reasons=rejection_reasons,
            quote=quote,
            validation=validation,
            indicators=indicators,
            fundamentals=fundamentals,
        )

    def screen_universe(
        self,
        symbols: Sequence[str],
        now: datetime | None = None,
    ) -> list[UniverseScreeningResult]:
        """Screen a sequence of tickers using a single shared ^VIX fetch."""
        if not symbols:
            return []
        vix = self.market_data.get_vix()
        return [self.screen_symbol(symbol, now=now, vix=vix) for symbol in symbols]
