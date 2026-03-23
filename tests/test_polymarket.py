from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from pmr.config import MonitoringConfig
from pmr.detector import detect_significant_moves
from pmr.providers import PolymarketMarketDataProvider
from pmr.research_payloads import build_research_input_payload
from pmr.storage import SnapshotBounds


class PolymarketProviderTests(unittest.TestCase):
    def test_provider_builds_market_series_from_live_like_payloads(self) -> None:
        client = StubPolymarketClient(
            markets=[
                {
                    "id": "market-1",
                    "question": "Will the Fed cut rates by June?",
                    "slug": "fed-cut-rates-june",
                    "conditionId": "condition-1",
                    "description": "Macro market about the Fed and inflation.",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "clobTokenIds": "[\"token-yes\", \"token-no\"]",
                    "volume24hr": 250000,
                    "volume1wk": 950000,
                    "liquidityNum": 300000,
                    "active": True,
                    "closed": False,
                    "events": [
                        {
                            "title": "Fed decision in June",
                            "eventMetadata": {
                                "context_description": "Inflation and FOMC commentary are driving repricing."
                            },
                        }
                    ],
                },
                {
                    "id": "market-2",
                    "question": "Utah State Aggies vs. Arizona Wildcats",
                    "slug": "sports-market",
                    "description": "Sports market",
                    "outcomes": "[\"Utah State Aggies\", \"Arizona Wildcats\"]",
                    "clobTokenIds": "[\"sports-a\", \"sports-b\"]",
                    "volume24hr": 900000,
                    "volume1wk": 1000000,
                    "liquidityNum": 500000,
                    "active": True,
                    "closed": False,
                    "sportsMarketType": "moneyline",
                },
            ],
            history_by_token={
                "token-yes": [
                    {"t": 1735689600, "p": 0.39},
                    {"t": 1735776000, "p": 0.40},
                    {"t": 1735862400, "p": 0.39},
                    {"t": 1735948800, "p": 0.40},
                ]
            },
            open_interest_by_condition={"condition-1": 420000.0},
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=10,
            page_size=10,
            max_pages=1,
            history_days=30,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
        )

        market_series = provider.list_market_series()

        self.assertEqual(len(market_series), 1)
        self.assertEqual(market_series[0].market.market_id, "market-1")
        self.assertEqual(market_series[0].market.tracked_outcome, "Yes")
        self.assertEqual(market_series[0].market.open_interest_usd, 420000.0)
        self.assertEqual(market_series[0].market.category, "macro")

    def test_provider_balances_category_coverage_with_per_category_caps(self) -> None:
        client = StubPolymarketClient(
            markets=[
                _market_payload("pol-1", "Will Candidate A win the election?", "Politics question", "condition-pol-1", "token-pol-1"),
                _market_payload("pol-2", "Will Candidate B win the election?", "Politics question", "condition-pol-2", "token-pol-2"),
                _market_payload("pol-3", "Will Candidate C win the election?", "Politics question", "condition-pol-3", "token-pol-3"),
                _market_payload("macro-1", "Will the Fed cut rates this quarter?", "Macro question", "condition-macro-1", "token-macro-1"),
                _market_payload("geo-1", "US x Iran ceasefire by June?", "Geopolitics question", "condition-geo-1", "token-geo-1"),
            ],
            history_by_token={
                "token-pol-1": _history_points([0.20, 0.21, 0.22]),
                "token-pol-2": _history_points([0.30, 0.31, 0.32]),
                "token-pol-3": _history_points([0.40, 0.41, 0.42]),
                "token-macro-1": _history_points([0.35, 0.36, 0.40]),
                "token-geo-1": _history_points([0.45, 0.46, 0.50]),
            },
            open_interest_by_condition={
                "condition-pol-1": 200000.0,
                "condition-pol-2": 200000.0,
                "condition-pol-3": 200000.0,
                "condition-macro-1": 200000.0,
                "condition-geo-1": 200000.0,
            },
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=3,
            page_size=10,
            max_pages=1,
            history_days=30,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
            max_markets_per_category=1,
        )

        market_series = provider.list_market_series()

        self.assertEqual(len(market_series), 3)
        self.assertEqual(
            {series.market.category for series in market_series},
            {"politics", "macro", "geopolitics"},
        )

    def test_research_payload_contains_detector_fields(self) -> None:
        client = StubPolymarketClient(
            markets=[
                _market_payload(
                    "market-1",
                    "Will inflation print below 2.5% this quarter?",
                    "Macro market about CPI inflation.",
                    "condition-1",
                    "token-yes",
                    event_title="Inflation prints",
                )
            ],
            history_by_token={
                "token-yes": [
                    {"t": 1735689600, "p": 0.20},
                    {"t": 1735776000, "p": 0.21},
                    {"t": 1735862400, "p": 0.20},
                    {"t": 1735948800, "p": 0.21},
                    {"t": 1736035200, "p": 0.20},
                    {"t": 1736121600, "p": 0.21},
                    {"t": 1736208000, "p": 0.20},
                    {"t": 1736294400, "p": 0.21},
                    {"t": 1736380800, "p": 0.28},
                    {"t": 1736467200, "p": 0.33},
                    {"t": 1736553600, "p": 0.35},
                    {"t": 1736640000, "p": 0.36},
                    {"t": 1736726400, "p": 0.37},
                    {"t": 1736812800, "p": 0.36},
                ]
            },
            open_interest_by_condition={"condition-1": 300000.0},
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=10,
            page_size=10,
            max_pages=1,
            history_days=30,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
        )

        events = detect_significant_moves(provider.list_market_series(), MonitoringConfig())
        payload = build_research_input_payload(events, MonitoringConfig(), source_name="polymarket")

        self.assertEqual(len(payload["anomalies"]), 1)
        anomaly = payload["anomalies"][0]
        self.assertEqual(anomaly["market"]["tracked_outcome"], "Yes")
        self.assertIn("features", anomaly)
        self.assertIn("baseline_stats", anomaly)
        self.assertIn("normalized_scores", anomaly)

    def test_incremental_refresh_uses_recent_tail_instead_of_full_window(self) -> None:
        current_time = datetime(2026, 3, 23, tzinfo=timezone.utc)
        latest_observed_at = current_time - timedelta(hours=2)
        client = StubPolymarketClient(
            markets=[
                _market_payload(
                    "market-1",
                    "Will inflation print below 2.5% this quarter?",
                    "Macro market about CPI inflation.",
                    "condition-1",
                    "token-yes",
                )
            ],
            history_by_token={"token-yes": _history_points([0.30, 0.31, 0.32])},
            open_interest_by_condition={"condition-1": 300000.0},
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=1,
            page_size=10,
            max_pages=1,
            history_days=90,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
            refresh_mode="incremental",
            existing_snapshot_bounds={
                "market-1": SnapshotBounds(
                    earliest_observed_at=current_time - timedelta(days=20),
                    latest_observed_at=latest_observed_at,
                )
            },
            incremental_overlap_minutes=180,
            current_time=current_time,
        )

        provider.list_market_series()

        request = client.history_requests[0]
        self.assertEqual(request["token_id"], "token-yes")
        self.assertEqual(
            request["start_ts"],
            int((latest_observed_at - timedelta(minutes=180)).timestamp()),
        )
        self.assertEqual(request["end_ts"], int(current_time.timestamp()))

    def test_backfill_refresh_targets_older_gap_first(self) -> None:
        current_time = datetime(2026, 3, 23, tzinfo=timezone.utc)
        earliest_observed_at = current_time - timedelta(days=20)
        client = StubPolymarketClient(
            markets=[
                _market_payload(
                    "market-1",
                    "US x Iran ceasefire by June?",
                    "Geopolitics question",
                    "condition-1",
                    "token-yes",
                )
            ],
            history_by_token={"token-yes": _history_points([0.45, 0.46, 0.47])},
            open_interest_by_condition={"condition-1": 300000.0},
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=1,
            page_size=10,
            max_pages=1,
            history_days=90,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
            refresh_mode="backfill",
            existing_snapshot_bounds={
                "market-1": SnapshotBounds(
                    earliest_observed_at=earliest_observed_at,
                    latest_observed_at=current_time - timedelta(hours=1),
                )
            },
            incremental_overlap_minutes=180,
            backfill_chunk_days=7,
            current_time=current_time,
        )

        provider.list_market_series()

        request = client.history_requests[0]
        self.assertEqual(
            request["start_ts"],
            int((earliest_observed_at - timedelta(days=7)).timestamp()),
        )
        self.assertEqual(
            request["end_ts"],
            int((earliest_observed_at + timedelta(minutes=180)).timestamp()),
        )


class StubPolymarketClient:
    def __init__(
        self,
        *,
        markets: Sequence[dict[str, Any]],
        history_by_token: dict[str, list[dict[str, Any]]],
        open_interest_by_condition: dict[str, float],
    ) -> None:
        self._markets = list(markets)
        self._history_by_token = history_by_token
        self._open_interest_by_condition = open_interest_by_condition
        self.history_requests: list[dict[str, Any]] = []

    def list_markets(self, limit: int, offset: int) -> Sequence[dict[str, Any]]:
        return self._markets[offset : offset + limit]

    def get_price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity_minutes: int,
    ) -> Sequence[dict[str, Any]]:
        self.history_requests.append(
            {
                "token_id": token_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "fidelity_minutes": fidelity_minutes,
            }
        )
        return self._history_by_token.get(token_id, [])

    def get_open_interest(self, condition_id: str) -> float | None:
        return self._open_interest_by_condition.get(condition_id)


def _market_payload(
    market_id: str,
    question: str,
    description: str,
    condition_id: str,
    tracked_token_id: str,
    *,
    event_title: str | None = None,
) -> dict[str, Any]:
    return {
        "id": market_id,
        "question": question,
        "slug": market_id,
        "conditionId": condition_id,
        "description": description,
        "outcomes": "[\"Yes\", \"No\"]",
        "clobTokenIds": json_string([tracked_token_id, f"{tracked_token_id}-no"]),
        "volume24hr": 150000,
        "volume1wk": 800000,
        "liquidityNum": 250000,
        "active": True,
        "closed": False,
        "events": [{"title": event_title or question}],
    }


def _history_points(probabilities: list[float]) -> list[dict[str, Any]]:
    base = 1735689600
    return [{"t": base + 86400 * index, "p": probability} for index, probability in enumerate(probabilities)]


def json_string(values: list[str]) -> str:
    return "[%s]" % ", ".join(f'"{value}"' for value in values)


if __name__ == "__main__":
    unittest.main()
