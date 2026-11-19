"""Unit tests for the AI Analyst module (models, news aggregation, Gemini gating, SQLite audit)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from alpaca.data.models.news import NewsSet
from alpaca.data.requests import NewsRequest
from google.genai import errors as genai_errors, types
from pydantic import ValidationError

from src.analyst import (
    NY_TIMEZONE,
    AnalystConfig,
    AnalystReport,
    GeminiAnalyst,
    NewsAggregator,
    NewsItem,
    compute_relative_volume,
    map_sentiment_label,
)
from src.storage import TradingAuditStore


def _make_report(**overrides: object) -> AnalystReport:
    payload: dict[str, object] = {
        "ticker": "AAPL",
        "sentiment_score": 0.65,
        "confidence_score": 0.85,
        "independent_sources_count": 4,
        "volume_confirmed": True,
        "manipulation_risk": False,
        "rationale": "Multi-source earnings beat corroborated by institutional volume.",
    }
    payload.update(overrides)
    return AnalystReport(**payload)


def _make_sample_news() -> list[NewsItem]:
    return [
        NewsItem(
            headline="Apple beats quarterly revenue estimates",
            summary="Services and iPhone demand topped consensus.",
            source="Reuters",
            url="https://example.com/aapl-1",
            published_at=datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc),
            provider="finnhub",
        ),
        NewsItem(
            headline="Apple raises guidance on AI hardware cycle",
            summary="Management guided above Wall Street forecasts.",
            source="Bloomberg",
            url="https://example.com/aapl-2",
            published_at=datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc),
            provider="alpaca",
        ),
        NewsItem(
            headline="Institutional inflows surge into Apple after bell",
            summary="Block trade activity doubled 20-day baseline.",
            source="Benzinga",
            url="https://example.com/aapl-3",
            published_at=datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc),
            provider="alpaca",
        ),
    ]


# ---------------------------------------------------------------------------
# AnalystReport Pydantic validation
# ---------------------------------------------------------------------------


def test_analyst_report_valid_boundaries_and_ticker_normalization() -> None:
    report_low = _make_report(
        ticker=" msft ",
        sentiment_score=-1.0,
        confidence_score=0.0,
        independent_sources_count=0,
    )
    report_high = _make_report(
        ticker="nvda",
        sentiment_score=1.0,
        confidence_score=1.0,
        independent_sources_count=5,
    )
    assert report_low.ticker == "MSFT"
    assert report_low.sentiment_score == -1.0
    assert report_low.confidence_score == 0.0
    assert report_high.ticker == "NVDA"
    assert report_high.sentiment_score == 1.0
    assert report_high.confidence_score == 1.0
    # Gemini v1 rejects additionalProperties/additional_properties in response_schema
    assert "additionalProperties" not in AnalystReport.model_json_schema()


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"sentiment_score": -1.01},
        {"sentiment_score": 1.01},
        {"sentiment_score": math.nan},
        {"confidence_score": -0.01},
        {"confidence_score": 1.01},
        {"confidence_score": math.inf},
        {"independent_sources_count": -1},
        {"ticker": ""},
        {"ticker": "   "},
        {"rationale": ""},
        {"rationale": "   "},
        {"unexpected_field": "disallowed"},
    ],
)
def test_analyst_report_rejects_invalid_bounds_and_fields(
    bad_fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _make_report(**bad_fields)


# ---------------------------------------------------------------------------
# NewsAggregator tests
# ---------------------------------------------------------------------------


def test_news_aggregator_combines_finnhub_and_alpaca_dedups_and_normalizes_ny() -> None:
    finnhub_mock = MagicMock()
    # 2026-05-01T14:30:00Z = 10:30:00 America/New_York (EDT, UTC-4)
    ts_first = int(datetime(2026, 5, 1, 14, 30, tzinfo=timezone.utc).timestamp())
    ts_dup_url = int(datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc).timestamp())
    finnhub_mock.company_news.return_value = [
        {
            "headline": "Apple Unveils New Silicon Roadmap",
            "summary": "Finnhub summary",
            "source": "Reuters",
            "url": "https://news.example.com/apple-silicon/",
            "datetime": ts_first,
        },
        {
            "headline": "Different Headline Same URL",
            "summary": "Duplicate by URL",
            "source": "MarketWatch",
            "url": "https://news.example.com/apple-silicon",
            "datetime": ts_dup_url,
        },
    ]

    alpaca_news_set = NewsSet(
        raw_data={
            "news": [
                {
                    "id": 101,
                    "headline": "  apple unveils   new silicon roadmap ",
                    "source": "Bloomberg",
                    "url": "https://other.example.com/duplicate-headline",
                    "summary": "Duplicate by normalized headline",
                    "created_at": "2026-05-01T14:45:00Z",
                    "updated_at": "2026-05-01T14:45:00Z",
                    "symbols": ["AAPL"],
                    "author": "Reporter",
                    "content": "Body",
                },
                {
                    "id": 102,
                    "headline": "Institutional Buyers Accumulate AAPL",
                    "source": "Benzinga",
                    "url": "https://news.example.com/aapl-buyers",
                    "summary": "Volume surges past 20D average.",
                    "created_at": "2026-05-01T16:15:00Z",
                    "updated_at": "2026-05-01T16:15:00Z",
                    "symbols": ["AAPL"],
                    "author": "Desk",
                    "content": "Body",
                },
            ]
        }
    )
    alpaca_mock = MagicMock()
    alpaca_mock.get_news.return_value = alpaca_news_set

    as_of_utc = datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc)
    aggregator = NewsAggregator(finnhub_client=finnhub_mock, news_client=alpaca_mock)
    items = aggregator.fetch_news("aapl", lookback_hours=48, as_of=as_of_utc)

    # Verify provider call arguments
    finnhub_mock.company_news.assert_called_once_with(
        "AAPL",
        _from="2026-04-29",
        to="2026-05-01",
    )
    alpaca_mock.get_news.assert_called_once()
    sent_request = alpaca_mock.get_news.call_args.args[0]
    assert isinstance(sent_request, NewsRequest)
    assert sent_request.symbols == "AAPL"
    assert sent_request.end.tzinfo == NY_TIMEZONE
    assert sent_request.start.tzinfo == NY_TIMEZONE

    # Verify deduplication (4 raw -> 2 unique: one by URL, one by headline)
    assert len(items) == 2
    # Sorted newest-first: 16:15Z (12:15 NY) before 14:45Z (10:45 NY) or 15:00Z (11:00 NY)
    assert items[0].headline == "Institutional Buyers Accumulate AAPL"
    assert items[0].provider == "alpaca"
    assert items[0].published_at.tzinfo == ZoneInfo("America/New_York")
    assert items[0].published_at.hour == 12
    assert items[0].published_at.minute == 15

    assert items[1].provider == "finnhub"
    assert items[1].published_at.tzinfo == ZoneInfo("America/New_York")

    sources = NewsAggregator.distinct_sources(items)
    assert sources == ["Benzinga", "Reuters"]


def test_news_aggregator_env_fallback_and_partial_provider_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-finnhub-key")
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-alpaca-secret")

    with (
        patch("src.analyst.news.finnhub.Client") as mock_fh_cls,
        patch("src.analyst.news.NewsClient") as mock_alpaca_cls,
    ):
        mock_fh = mock_fh_cls.return_value
        mock_fh.company_news.side_effect = RuntimeError("Finnhub rate limit")
        mock_alpaca = mock_alpaca_cls.return_value
        mock_alpaca.get_news.return_value = SimpleNamespace(
            data={
                "news": [
                    SimpleNamespace(
                        headline="Fallback Alpaca Headline",
                        summary="Only Alpaca succeeded",
                        source="Reuters",
                        url="https://example.com/fallback",
                        created_at=datetime(2026, 5, 1, 13, 0, tzinfo=timezone.utc),
                    )
                ]
            }
        )

        aggregator = NewsAggregator()
        mock_fh_cls.assert_called_once_with(api_key="test-finnhub-key")
        mock_alpaca_cls.assert_called_once_with(
            api_key="test-alpaca-key",
            secret_key="test-alpaca-secret",
        )

        items = aggregator.fetch_news("TSLA")
        assert len(items) == 1
        assert items[0].source == "Reuters"
        assert items[0].published_at.tzinfo == NY_TIMEZONE
        assert NewsAggregator.distinct_sources(items) == ["Reuters"]


# ---------------------------------------------------------------------------
# Relative volume & sentiment helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current_vol", "avg_vol", "expected"),
    [
        (1_800_000, 1_000_000, 1.8),
        (1_200_000, 1_000_000, 1.2),
        (0, 1_000_000, 0.0),
        (500_000, 0, 0.0),
        (500_000, -100, 0.0),
        (-50, 1_000_000, 0.0),
        (math.nan, 1_000_000, 0.0),
        (1_000_000, math.inf, 0.0),
    ],
)
def test_compute_relative_volume_handles_zero_and_invalid_inputs(
    current_vol: float,
    avg_vol: float,
    expected: float,
) -> None:
    assert compute_relative_volume(current_vol, avg_vol) == pytest.approx(expected)


def test_map_sentiment_label() -> None:
    assert map_sentiment_label(0.5) == "bullish"
    assert map_sentiment_label(-0.5) == "bearish"
    assert map_sentiment_label(0.0) == "neutral"


# ---------------------------------------------------------------------------
# GeminiAnalyst call verification, deterministic gates, and SQLite audit
# ---------------------------------------------------------------------------


def test_gemini_analyst_call_verification_and_approved_persistence(
    tmp_path: pytest.TempPathFactory,
) -> None:
    store = TradingAuditStore(tmp_path / "analyst_audit.db")
    mock_news_agg = MagicMock(spec=NewsAggregator)
    mock_news_agg.fetch_news.return_value = _make_sample_news()

    expected_report = _make_report(
        ticker="AAPL",
        sentiment_score=0.72,
        confidence_score=0.70,  # Exact minimum boundary must pass
        independent_sources_count=3,  # Exact minimum boundary must pass
        volume_confirmed=True,
        manipulation_risk=False,
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=expected_report,
        text=expected_report.model_dump_json(),
    )

    analyst = GeminiAnalyst(
        audit_store=store,
        news_aggregator=mock_news_agg,
        client=mock_client,
        config=AnalystConfig(),
    )
    result = analyst.evaluate(
        symbol="aapl",
        current_volume=1_200_000,  # Exactly 1.2x avg_volume_20d
        avg_volume_20d=1_000_000,
    )

    assert result.approved is True
    assert result.rejection_reasons == []
    assert result.ticker == "AAPL"
    assert result.relative_volume == pytest.approx(1.2)
    assert result.news_count == 3
    assert result.sources == ["Benzinga", "Bloomberg", "Reuters"]
    assert result.signal_id is not None

    # Verify google-genai invocation signature and config
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.5-flash-lite"
    gen_config = call_kwargs["config"]
    assert isinstance(gen_config, types.GenerateContentConfig)
    assert gen_config.response_mime_type == "application/json"
    assert gen_config.response_schema is AnalystReport
    assert gen_config.temperature == pytest.approx(0.1)
    assert gen_config.http_options is not None
    assert gen_config.http_options.timeout == 35_000
    assert gen_config.http_options.retry_options is not None
    assert gen_config.http_options.retry_options.attempts == 1

    # Verify SQLite persistence in tmp_path DB
    records = store.get_ai_signals(symbol="AAPL")
    assert len(records) == 1
    rec = records[0]
    assert rec.id == result.signal_id
    assert rec.symbol == "AAPL"
    assert rec.sentiment == "bullish"
    assert rec.confidence_score == pytest.approx(0.70)
    assert rec.consensus_count == 3
    assert rec.sources == ["Benzinga", "Bloomberg", "Reuters"]
    assert rec.rationale == expected_report.rationale
    assert rec.passed_threshold is True


@pytest.mark.parametrize(
    ("case_name", "report_overrides", "current_volume", "avg_volume_20d", "expected_fragment"),
    [
        (
            "low_confidence",
            {"confidence_score": 0.69},
            1_500_000,
            1_000_000,
            "confidence_score",
        ),
        (
            "insufficient_sources",
            {"independent_sources_count": 2},
            1_500_000,
            1_000_000,
            "independent_sources_count",
        ),
        (
            "llm_volume_not_confirmed",
            {"volume_confirmed": False},
            1_800_000,
            1_000_000,
            "volume_confirmed is False",
        ),
        (
            "price_volume_divergence_overrides_llm",
            {"volume_confirmed": True},  # LLM claims confirmed, but relative_volume < 1.2
            1_190_000,
            1_000_000,
            "price-volume divergence",
        ),
        (
            "zero_avg_volume_triggers_divergence",
            {"volume_confirmed": True},
            1_000_000,
            0.0,
            "price-volume divergence",
        ),
        (
            "manipulation_risk",
            {"manipulation_risk": True},
            2_000_000,
            1_000_000,
            "manipulation_risk",
        ),
    ],
)
def test_gemini_analyst_deterministic_rejections_persisted_to_sqlite(
    tmp_path: pytest.TempPathFactory,
    case_name: str,
    report_overrides: dict[str, object],
    current_volume: float,
    avg_volume_20d: float,
    expected_fragment: str,
) -> None:
    store = TradingAuditStore(tmp_path / f"{case_name}.db")
    report = _make_report(sentiment_score=-0.55, **report_overrides)

    # Exercise JSON text fallback parsing when parsed is None
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        parsed=None,
        text=json.dumps(report.model_dump()),
    )

    analyst = GeminiAnalyst(
        audit_store=store,
        news_aggregator=MagicMock(spec=NewsAggregator),
        client=mock_client,
    )
    result = analyst.evaluate(
        symbol="AAPL",
        current_volume=current_volume,
        avg_volume_20d=avg_volume_20d,
        news_items=_make_sample_news(),
    )

    assert result.approved is False
    assert any(expected_fragment in reason for reason in result.rejection_reasons)
    assert result.signal_id is not None

    records = store.get_ai_signals(symbol="AAPL")
    assert len(records) == 1
    rec = records[0]
    assert rec.passed_threshold is False
    assert rec.sentiment == "bearish"
    assert rec.confidence_score == pytest.approx(report.confidence_score)
    assert rec.consensus_count == report.independent_sources_count


def test_gemini_analyst_collects_multiple_rejection_reasons_and_uses_env_key(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    store = TradingAuditStore(tmp_path / "multi_reject.db")
    bad_report = _make_report(
        sentiment_score=0.0,
        confidence_score=0.40,
        independent_sources_count=1,
        volume_confirmed=False,
        manipulation_risk=True,
    )

    with patch("src.analyst.client.genai.Client") as mock_genai_cls:
        mock_genai_cls.return_value.models.generate_content.return_value = SimpleNamespace(
            parsed=bad_report
        )
        analyst = GeminiAnalyst(
            audit_store=store,
            news_aggregator=MagicMock(spec=NewsAggregator),
        )
        mock_genai_cls.assert_called_once()
        client_kwargs = mock_genai_cls.call_args.kwargs
        assert client_kwargs["api_key"] == "fake-gemini-key"
        assert client_kwargs["http_options"].httpx_client is not None

        result = analyst.evaluate(
            symbol="GME",
            current_volume=500_000,
            avg_volume_20d=1_000_000,
            news_items=_make_sample_news(),
        )

    assert result.approved is False
    assert len(result.rejection_reasons) == 5
    records = store.get_ai_signals(symbol="GME")
    assert len(records) == 1
    assert records[0].sentiment == "neutral"
    assert records[0].passed_threshold is False


def test_gemini_analyst_falls_back_on_503_or_404_and_raises_on_400(
    tmp_path: pytest.TempPathFactory,
) -> None:
    store = TradingAuditStore(tmp_path / "fallback.db")
    ok_report = _make_report(ticker="AAPL", confidence_score=0.82)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        genai_errors.ServerError(503, {"error": {"message": "overloaded"}}),
        SimpleNamespace(parsed=ok_report),
    ]

    analyst = GeminiAnalyst(
        audit_store=store,
        news_aggregator=MagicMock(spec=NewsAggregator),
        client=mock_client,
        config=AnalystConfig(
            model_name="gemini-4-hydrogen",
            fallback_models=("gemini-3.5-flash-lite",),
        ),
    )
    result = analyst.evaluate(
        symbol="AAPL",
        current_volume=2_000_000,
        avg_volume_20d=1_000_000,
        news_items=_make_sample_news(),
    )
    assert result.approved is True
    called_models = [
        call.kwargs["model"] for call in mock_client.models.generate_content.call_args_list
    ]
    assert called_models == ["gemini-4-hydrogen", "gemini-3.5-flash-lite"]

    # Non-retryable 400 error must surface immediately without trying fallbacks
    mock_client_400 = MagicMock()
    mock_client_400.models.generate_content.side_effect = genai_errors.ClientError(
        400, {"error": {"message": "bad payload"}}
    )
    analyst_400 = GeminiAnalyst(
        audit_store=store,
        news_aggregator=MagicMock(spec=NewsAggregator),
        client=mock_client_400,
    )
    with pytest.raises(genai_errors.ClientError):
        analyst_400.evaluate(
            symbol="AAPL",
            current_volume=2_000_000,
            avg_volume_20d=1_000_000,
            news_items=_make_sample_news(),
        )
    assert mock_client_400.models.generate_content.call_count == 1


def test_gemini_analyst_falls_back_on_504_timeout_and_applies_model_cooldown(
    tmp_path: pytest.TempPathFactory,
) -> None:
    import httpx

    store = TradingAuditStore(tmp_path / "timeout_cooldown.db")
    ok_report = _make_report(ticker="WMT", confidence_score=0.81)
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        genai_errors.ServerError(504, {"error": {"message": "deadline exceeded"}}),
        httpx.ReadTimeout("read timed out"),
        SimpleNamespace(parsed=ok_report),
        # Second evaluate() call should try the healthy model first without re-hitting cooled-down models
        SimpleNamespace(parsed=ok_report),
    ]

    analyst = GeminiAnalyst(
        audit_store=store,
        news_aggregator=MagicMock(spec=NewsAggregator),
        client=mock_client,
        config=AnalystConfig(
            model_name="gemini-3.5-flash-lite",
            fallback_models=("gemini-3.8-flash", "gemini-4-hydrogen"),
            model_cooldown_seconds=300.0,
        ),
    )

    first = analyst.evaluate(
        symbol="WMT",
        current_volume=2_000_000,
        avg_volume_20d=1_000_000,
        news_items=_make_sample_news(),
    )
    second = analyst.evaluate(
        symbol="WMT",
        current_volume=2_000_000,
        avg_volume_20d=1_000_000,
        news_items=_make_sample_news(),
    )

    assert first.approved is True
    assert second.approved is True
    called_models = [
        call.kwargs["model"] for call in mock_client.models.generate_content.call_args_list
    ]
    assert called_models == [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
        "gemini-4-hydrogen",
    ]


def test_build_analysis_prompt_caps_articles_while_preserving_source_diversity() -> None:
    from datetime import timedelta

    from src.analyst.client import _build_analysis_prompt, _select_prompt_articles

    base = datetime(2026, 11, 19, 11, 0, tzinfo=NY_TIMEZONE)
    items: list[NewsItem] = []
    # 25 recent Reuters articles + 1 older article each from 3 other sources
    for i in range(25):
        items.append(
            NewsItem(
                headline=f"Reuters headline {i}",
                summary="A" * 400,
                source="Reuters",
                published_at=base - timedelta(minutes=i),
                provider="finnhub",
            )
        )
    for offset, src in enumerate(["Bloomberg", "CNBC", "WSJ"], start=30):
        items.append(
            NewsItem(
                headline=f"{src} breaking update",
                summary="B" * 350,
                source=src,
                published_at=base - timedelta(minutes=offset),
                provider="alpaca",
            )
        )

    selected = _select_prompt_articles(items, max_articles=10)
    assert len(selected) == 10
    assert {item.source for item in selected} == {"Reuters", "Bloomberg", "CNBC", "WSJ"}

    cfg = AnalystConfig(max_prompt_articles=10, max_summary_chars=120)
    prompt = _build_analysis_prompt(
        symbol="AMD",
        current_volume=2_000_000,
        avg_volume_20d=1_000_000,
        relative_volume=2.0,
        news_items=items,
        sources=["Bloomberg", "CNBC", "Reuters", "WSJ"],
        config=cfg,
    )
    assert "Total Deduplicated Articles: 28 (showing 10 most recent)" in prompt
    assert "A" * 130 not in prompt
    assert "..." in prompt
