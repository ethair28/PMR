from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pmr.config import MonitoringConfig
from pmr.detector import detect_significant_moves, evaluate_market_event
from pmr.models import Market, MarketSeries, MarketSnapshot
from pmr.pipeline import run_pipeline
from pmr.providers import JsonFileMarketDataProvider, MockResearchProvider, StaticMarketDataProvider


BASE_CONFIG = MonitoringConfig()


class DetectorTests(unittest.TestCase):
    def test_old_market_with_clear_weekly_repricing_scores_in_full_history_mode(self) -> None:
        series = _build_series(
            market_id="old-clear",
            category="macro",
            probabilities=(
                [0.40, 0.41, 0.39, 0.40] * 15
                + [0.40, 0.42, 0.46, 0.51, 0.56, 0.58, 0.57, 0.58]
            ),
        )

        event = evaluate_market_event(series, BASE_CONFIG)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.history_mode, "full_history")
        self.assertTrue(event.eligible_for_ranking)
        self.assertGreaterEqual(event.max_abs_move_24h, BASE_CONFIG.min_abs_move_24h)
        self.assertGreaterEqual(event.weekly_range, BASE_CONFIG.min_weekly_range)

    def test_spike_and_retrace_market_is_still_detected(self) -> None:
        series = _build_series(
            market_id="spike-retrace",
            category="geopolitics",
            probabilities=(
                [0.30, 0.31, 0.30, 0.31] * 15
                + [0.30, 0.31, 0.34, 0.55, 0.37, 0.33, 0.32, 0.33]
            ),
        )

        event = evaluate_market_event(series, BASE_CONFIG)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertTrue(event.eligible_for_ranking)
        self.assertLess(abs(event.close_to_open_move), BASE_CONFIG.min_abs_move_24h)
        self.assertGreaterEqual(event.weekly_range, BASE_CONFIG.min_weekly_range)
        self.assertGreaterEqual(event.max_abs_move_24h, BASE_CONFIG.min_abs_move_24h)

    def test_two_week_market_is_ranked_in_short_history_mode(self) -> None:
        series = _build_series(
            market_id="short-history",
            category="politics",
            probabilities=[
                0.41,
                0.42,
                0.41,
                0.42,
                0.41,
                0.42,
                0.41,
                0.42,
                0.47,
                0.49,
                0.52,
                0.54,
                0.55,
                0.56,
            ],
        )

        event = evaluate_market_event(series, BASE_CONFIG)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.history_mode, "short_history")
        self.assertTrue(event.eligible_for_ranking)
        self.assertLess(event.confidence_score, 0.75)

    def test_too_new_market_is_marked_insufficient(self) -> None:
        series = _build_series(
            market_id="too-new",
            category="economics",
            probabilities=[0.20, 0.22, 0.26, 0.31, 0.35, 0.39, 0.41, 0.43],
        )

        event = evaluate_market_event(series, BASE_CONFIG)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.history_mode, "insufficient_data")
        self.assertFalse(event.eligible_for_ranking)
        self.assertEqual(event.exclusion_reason, "insufficient_data")

    def test_noisy_illiquid_market_is_filtered_out(self) -> None:
        series = _build_series(
            market_id="illiquid-noisy",
            category="politics",
            probabilities=(
                [0.12, 0.16, 0.10, 0.15] * 15
                + [0.18, 0.27, 0.19, 0.31, 0.22, 0.29, 0.25, 0.30]
            ),
            volume_7d_usd=5_000,
            volume_24h_usd=800,
            open_interest_usd=2_000,
        )

        event = evaluate_market_event(series, BASE_CONFIG)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertFalse(event.eligible_for_ranking)
        self.assertEqual(event.exclusion_reason, "liquidity")


class PipelineTests(unittest.TestCase):
    def test_pipeline_report_includes_new_detector_fields(self) -> None:
        markets = (
            _build_series(
                market_id="pipeline-market",
                category="macro",
                probabilities=(
                    [0.40, 0.41, 0.39, 0.40] * 15
                    + [0.40, 0.43, 0.47, 0.52, 0.55, 0.57, 0.56, 0.58]
                ),
            ),
        )

        result = run_pipeline(
            market_data_provider=StaticMarketDataProvider(markets),
            research_provider=MockResearchProvider(),
            config=BASE_CONFIG,
        )

        self.assertEqual(len(result.event_reports), 1)
        self.assertIn("History mode:", result.markdown_report)
        self.assertIn("Max abs 24h move:", result.markdown_report)

    def test_json_provider_loads_market_file(self) -> None:
        payload = {
            "markets": [
                {
                    "market_id": "test-market",
                    "question": "Will inflation print below 2% this year?",
                    "category": "economics",
                    "tags": ["inflation"],
                    "volume_7d_usd": 100000,
                    "volume_24h_usd": 25000,
                    "open_interest_usd": 50000,
                    "snapshots": [
                        {"observed_at": "2026-03-10T00:00:00+00:00", "probability": 0.2},
                        {"observed_at": "2026-03-21T00:00:00+00:00", "probability": 0.4},
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "markets.json"
            path.write_text(json.dumps(payload))
            provider = JsonFileMarketDataProvider(path)

            markets = provider.list_market_series()

        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].market.market_id, "test-market")

    def test_detect_significant_moves_returns_ranked_candidates_only(self) -> None:
        markets = (
            _build_series(
                market_id="ranked",
                category="macro",
                probabilities=(
                    [0.40, 0.41, 0.39, 0.40] * 15
                    + [0.40, 0.43, 0.47, 0.52, 0.55, 0.57, 0.56, 0.58]
                ),
            ),
            _build_series(
                market_id="excluded",
                category="politics",
                probabilities=[0.20, 0.24, 0.31, 0.36, 0.38, 0.39, 0.42, 0.43],
            ),
        )

        events = detect_significant_moves(markets, BASE_CONFIG)

        self.assertEqual([event.market.market_id for event in events], ["ranked"])


def _build_series(
    market_id: str,
    category: str,
    probabilities: list[float] | tuple[float, ...],
    *,
    volume_7d_usd: float = 250_000,
    volume_24h_usd: float = 60_000,
    open_interest_usd: float = 90_000,
    start: datetime | None = None,
    interval_hours: int = 24,
) -> MarketSeries:
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshots = tuple(
        MarketSnapshot(
            observed_at=start + timedelta(hours=index * interval_hours),
            probability=probability,
        )
        for index, probability in enumerate(probabilities)
    )
    market = Market(
        market_id=market_id,
        question=f"Question for {market_id}?",
        category=category,
        tags=(category,),
        volume_7d_usd=volume_7d_usd,
        volume_24h_usd=volume_24h_usd,
        open_interest_usd=open_interest_usd,
    )
    return MarketSeries(market=market, snapshots=snapshots)


if __name__ == "__main__":
    unittest.main()
