from __future__ import annotations

import io
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from urllib.error import HTTPError

from pmr.config import MonitoringConfig
from pmr.detector import detect_significant_moves
from pmr.polymarket import HttpPolymarketClient
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

    def test_provider_does_not_use_event_context_for_topic_matching(self) -> None:
        client = StubPolymarketClient(
            markets=[
                {
                    "id": "social-market",
                    "question": "Will Elon Musk post 380-399 tweets this week?",
                    "slug": "elon-musk-tweet-count",
                    "conditionId": "condition-social",
                    "description": "This market resolves according to the number of times Elon Musk posts on X this week.",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "clobTokenIds": "[\"token-social-yes\", \"token-social-no\"]",
                    "volume24hr": 250000,
                    "volume1wk": 900000,
                    "liquidityNum": 300000,
                    "active": True,
                    "closed": False,
                    "events": [
                        {
                            "title": "Elon Musk # tweets this week",
                            "eventMetadata": {
                                "context_description": "Recent election surges imply traders expect heavy posting."
                            },
                        }
                    ],
                }
            ],
            history_by_token={"token-social-yes": _history_points([0.30, 0.32, 0.35])},
            open_interest_by_condition={"condition-social": 250000.0},
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

        scan = provider.scan_market_series()

        self.assertEqual(scan.markets, ())
        self.assertEqual(scan.diagnostics.rejection_counts["social_activity_tracker"], 1)

    def test_provider_uses_category_floors_then_global_overflow(self) -> None:
        client = StubPolymarketClient(
            markets=[
                _market_payload("pol-1", "Will Candidate A win the election?", "Politics question", "condition-pol-1", "token-pol-1"),
                _market_payload("pol-2", "Will Candidate B win the election?", "Politics question", "condition-pol-2", "token-pol-2"),
                _market_payload("pol-3", "Will Candidate C win the election?", "Politics question", "condition-pol-3", "token-pol-3"),
                _market_payload("pol-4", "Will Candidate D win the election?", "Politics question", "condition-pol-4", "token-pol-4"),
                _market_payload("macro-1", "Will the Fed cut rates this quarter?", "Macro question", "condition-macro-1", "token-macro-1"),
                _market_payload("geo-1", "US x Iran ceasefire by June?", "Geopolitics question", "condition-geo-1", "token-geo-1"),
            ],
            history_by_token={
                "token-pol-1": _history_points([0.20, 0.21, 0.22]),
                "token-pol-2": _history_points([0.30, 0.31, 0.32]),
                "token-pol-3": _history_points([0.40, 0.41, 0.42]),
                "token-pol-4": _history_points([0.50, 0.51, 0.52]),
                "token-macro-1": _history_points([0.35, 0.36, 0.40]),
                "token-geo-1": _history_points([0.45, 0.46, 0.50]),
            },
            open_interest_by_condition={
                "condition-pol-1": 200000.0,
                "condition-pol-2": 200000.0,
                "condition-pol-3": 200000.0,
                "condition-pol-4": 200000.0,
                "condition-macro-1": 200000.0,
                "condition-geo-1": 200000.0,
            },
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=4,
            page_size=10,
            max_pages=1,
            history_days=30,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
            min_markets_per_category=1,
            max_category_share_of_universe=0.75,
        )

        market_series = provider.list_market_series()

        self.assertEqual(len(market_series), 4)
        self.assertEqual(
            {series.market.market_id for series in market_series},
            {"pol-1", "pol-2", "macro-1", "geo-1"},
        )
        self.assertEqual([series.market.category for series in market_series].count("politics"), 2)

    def test_provider_uses_event_aggregated_liquidity_and_caps_sibling_contracts(self) -> None:
        client = StubPolymarketClient(
            markets=[
                _market_payload(
                    "slovenia-golob",
                    "Will Robert Golob be the next Prime Minister of Slovenia?",
                    "Politics question",
                    "condition-slo-1",
                    "token-slo-1",
                    event_title="Next Prime Minister of Slovenia",
                    volume_24h=35_000,
                    volume_1wk=310_000,
                ),
                _market_payload(
                    "slovenia-jansa",
                    "Will Janez Janša be the next Prime Minister of Slovenia?",
                    "Politics question",
                    "condition-slo-2",
                    "token-slo-2",
                    event_title="Next Prime Minister of Slovenia",
                    volume_24h=18_000,
                    volume_1wk=286_000,
                ),
                _market_payload(
                    "slovenia-logar",
                    "Will Anže Logar be the next Prime Minister of Slovenia?",
                    "Politics question",
                    "condition-slo-3",
                    "token-slo-3",
                    event_title="Next Prime Minister of Slovenia",
                    volume_24h=14_000,
                    volume_1wk=166_000,
                ),
                _market_payload(
                    "other-politics",
                    "Will Candidate A win the election?",
                    "Politics question",
                    "condition-pol-1",
                    "token-pol-1",
                    event_title="Candidate A election",
                    volume_24h=60_000,
                    volume_1wk=400_000,
                ),
                _market_payload(
                    "geo-market",
                    "US x Iran ceasefire by June?",
                    "Geopolitics question",
                    "condition-geo-1",
                    "token-geo-1",
                    volume_24h=50_000,
                    volume_1wk=340_000,
                ),
                _market_payload(
                    "macro-market",
                    "Will the Fed cut rates this quarter?",
                    "Macro question",
                    "condition-macro-1",
                    "token-macro-1",
                    volume_24h=45_000,
                    volume_1wk=330_000,
                ),
            ],
            history_by_token={
                "token-slo-1": _history_points([0.20, 0.35, 0.58]),
                "token-slo-2": _history_points([0.70, 0.48, 0.19]),
                "token-slo-3": _history_points([0.05, 0.03, 0.01]),
                "token-pol-1": _history_points([0.40, 0.42, 0.44]),
                "token-geo-1": _history_points([0.30, 0.32, 0.35]),
                "token-macro-1": _history_points([0.25, 0.27, 0.29]),
            },
            open_interest_by_condition={
                "condition-slo-1": 200000.0,
                "condition-slo-2": 200000.0,
                "condition-slo-3": 200000.0,
                "condition-pol-1": 200000.0,
                "condition-geo-1": 200000.0,
                "condition-macro-1": 200000.0,
            },
        )
        provider = PolymarketMarketDataProvider(
            client=client,
            max_markets=4,
            page_size=10,
            max_pages=1,
            history_days=30,
            fidelity_minutes=60,
            target_categories=MonitoringConfig().target_categories,
            category_aliases=MonitoringConfig().category_aliases,
            min_markets_per_category=1,
            max_category_share_of_universe=0.75,
            max_contracts_per_event=2,
        )

        market_series = provider.list_market_series()

        self.assertEqual(
            {series.market.market_id for series in market_series},
            {"slovenia-golob", "slovenia-jansa", "geo-market", "macro-market"},
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
        self.assertEqual(len(payload["research_jobs"]), 1)
        anomaly = payload["anomalies"][0]
        research_job = payload["research_jobs"][0]
        self.assertEqual(anomaly["market"]["tracked_outcome"], "Yes")
        self.assertIn("story", anomaly)
        self.assertIn("features", anomaly)
        self.assertIn("baseline_stats", anomaly)
        self.assertIn("normalized_scores", anomaly)
        self.assertIn("story_type_hint", anomaly["story"])
        self.assertIn("distance_from_extremes", anomaly["story"])
        self.assertIn("entered_extreme_zone", anomaly["story"])
        self.assertIn("investigation", research_job)
        self.assertIn("primary_market", research_job)
        self.assertIn("editorial_priority_hint", research_job["story"])

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

    def test_scan_diagnostics_report_rejection_reasons(self) -> None:
        client = StubPolymarketClient(
            markets=[
                _market_payload(
                    "good-market",
                    "Will inflation print below 2.5% this quarter?",
                    "Macro market about CPI inflation.",
                    "condition-good",
                    "token-good",
                ),
                {
                    "id": "off-topic",
                    "question": "Will ETH break $10k this year?",
                    "slug": "eth-10k",
                    "description": "Crypto market",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "clobTokenIds": "[\"token-eth-yes\", \"token-eth-no\"]",
                    "volume24hr": 250000,
                    "volume1wk": 900000,
                    "liquidityNum": 300000,
                    "active": True,
                    "closed": False,
                    "events": [{"title": "ETH price"}],
                },
                {
                    "id": "sports-market",
                    "question": "Lakers vs Celtics",
                    "slug": "lakers-celtics",
                    "description": "Sports market",
                    "outcomes": "[\"Lakers\", \"Celtics\"]",
                    "clobTokenIds": "[\"sports-a\", \"sports-b\"]",
                    "volume24hr": 500000,
                    "volume1wk": 1200000,
                    "liquidityNum": 400000,
                    "active": True,
                    "closed": False,
                    "sportsMarketType": "moneyline",
                    "events": [{"title": "Lakers vs Celtics"}],
                },
                _market_payload(
                    "thin-history",
                    "Will tariffs rise this quarter?",
                    "Economics market",
                    "condition-thin",
                    "token-thin",
                ),
                {
                    "id": "btc-market",
                    "question": "Will Bitcoin reach $80,000 in March?",
                    "slug": "will-bitcoin-reach-80000-in-march",
                    "description": "This market will immediately resolve to Yes if any Binance 1 minute candle for BTC/USDT during the month specified in the title has a final High price equal to or above the listed price.",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "clobTokenIds": "[\"token-btc-yes\", \"token-btc-no\"]",
                    "volume24hr": 400000,
                    "volume1wk": 1400000,
                    "liquidityNum": 350000,
                    "active": True,
                    "closed": False,
                    "events": [{"title": "What price will Bitcoin hit in March?"}],
                },
                {
                    "id": "social-market",
                    "question": "Will Elon Musk post 380-399 tweets this week?",
                    "slug": "elon-musk-tweet-count",
                    "description": "This market resolves according to the number of times Elon Musk posts on X this week. The resolution source is the Post Counter at xtracker.polymarket.com.",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "clobTokenIds": "[\"token-social-yes\", \"token-social-no\"]",
                    "volume24hr": 300000,
                    "volume1wk": 1100000,
                    "liquidityNum": 320000,
                    "active": True,
                    "closed": False,
                    "events": [
                        {
                            "title": "Elon Musk # tweets this week",
                            "eventMetadata": {
                                "context_description": "Recent election surges imply traders expect heavy posting."
                            },
                        }
                    ],
                },
            ],
            history_by_token={
                "token-good": _history_points([0.20, 0.23, 0.26]),
                "token-thin": [{"t": 1735689600, "p": 0.41}],
                "token-btc-yes": _history_points([0.60, 0.61, 0.65]),
                "token-social-yes": _history_points([0.30, 0.32, 0.35]),
            },
            open_interest_by_condition={
                "condition-good": 300000.0,
                "condition-thin": 250000.0,
            },
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

        scan = provider.scan_market_series()

        self.assertEqual([series.market.market_id for series in scan.markets], ["good-market"])
        self.assertEqual(scan.diagnostics.payloads_seen, 6)
        self.assertEqual(scan.diagnostics.topic_matches, 2)
        self.assertEqual(scan.diagnostics.selected_markets, 1)
        self.assertEqual(scan.diagnostics.history_requests, 2)
        self.assertEqual(scan.diagnostics.rejection_counts["off_topic"], 1)
        self.assertEqual(scan.diagnostics.rejection_counts["sports_market"], 1)
        self.assertEqual(scan.diagnostics.rejection_counts["insufficient_history"], 1)
        self.assertEqual(scan.diagnostics.rejection_counts["public_market_proxy"], 1)
        self.assertEqual(scan.diagnostics.rejection_counts["social_activity_tracker"], 1)


class HttpPolymarketClientTests(unittest.TestCase):
    def test_client_chunks_long_history_requests(self) -> None:
        client = RecordingHttpPolymarketClient(max_price_history_span_days=14)

        history = client.get_price_history(
            token_id="token-1",
            start_ts=0,
            end_ts=30 * 24 * 60 * 60,
            fidelity_minutes=60,
        )

        self.assertEqual(len(client.chunk_calls), 3)
        self.assertEqual(client.chunk_calls[0]["startTs"], "0")
        self.assertEqual(client.chunk_calls[-1]["endTs"], str(30 * 24 * 60 * 60))
        self.assertEqual(
            [item["t"] for item in history],
            [0, 1209600, 1209601, 2419201, 2419202, 2592000],
        )

    def test_client_splits_again_when_endpoint_rejects_chunk_span(self) -> None:
        client = IntervalLimitedHttpPolymarketClient(max_allowed_span_days=7, max_price_history_span_days=14)

        history = client.get_price_history(
            token_id="token-1",
            start_ts=0,
            end_ts=14 * 24 * 60 * 60,
            fidelity_minutes=60,
        )

        self.assertGreater(len(client.chunk_calls), 2)
        self.assertEqual(history[0]["t"], 0)
        self.assertEqual(history[-1]["t"], 14 * 24 * 60 * 60)


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
    volume_24h: float = 150000,
    volume_1wk: float = 800000,
    liquidity: float = 250000,
) -> dict[str, Any]:
    return {
        "id": market_id,
        "question": question,
        "slug": market_id,
        "conditionId": condition_id,
        "description": description,
        "outcomes": "[\"Yes\", \"No\"]",
        "clobTokenIds": json_string([tracked_token_id, f"{tracked_token_id}-no"]),
        "volume24hr": volume_24h,
        "volume1wk": volume_1wk,
        "liquidityNum": liquidity,
        "active": True,
        "closed": False,
        "events": [{"title": event_title or question}],
    }


def _history_points(probabilities: list[float]) -> list[dict[str, Any]]:
    base = 1735689600
    return [{"t": base + 86400 * index, "p": probability} for index, probability in enumerate(probabilities)]


def json_string(values: list[str]) -> str:
    return "[%s]" % ", ".join(f'"{value}"' for value in values)


class RecordingHttpPolymarketClient(HttpPolymarketClient):
    def __init__(self, *, max_price_history_span_days: int = 14) -> None:
        super().__init__(max_price_history_span_days=max_price_history_span_days)
        self.chunk_calls: list[dict[str, str]] = []

    def _get_json(
        self,
        *,
        base_url: str,
        path: str,
        params: dict[str, str],
    ) -> Any:
        if path != "/prices-history":
            raise AssertionError(f"Unexpected path: {path}")
        self.chunk_calls.append(dict(params))
        return {
            "history": [
                {"t": int(params["startTs"]), "p": 0.4},
                {"t": int(params["endTs"]), "p": 0.6},
            ]
        }


class IntervalLimitedHttpPolymarketClient(RecordingHttpPolymarketClient):
    def __init__(
        self,
        *,
        max_allowed_span_days: int,
        max_price_history_span_days: int = 14,
    ) -> None:
        super().__init__(max_price_history_span_days=max_price_history_span_days)
        self.max_allowed_span_seconds = max_allowed_span_days * 24 * 60 * 60

    def _get_json(
        self,
        *,
        base_url: str,
        path: str,
        params: dict[str, str],
    ) -> Any:
        self.chunk_calls.append(dict(params))
        start_ts = int(params["startTs"])
        end_ts = int(params["endTs"])
        if end_ts - start_ts > self.max_allowed_span_seconds:
            raise HTTPError(
                url="https://clob.polymarket.com/prices-history",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"invalid filters: \'startTs\' and \'endTs\' interval is too long"}'),
            )
        return {
            "history": [
                {"t": start_ts, "p": 0.4},
                {"t": end_ts, "p": 0.6},
            ]
        }


if __name__ == "__main__":
    unittest.main()
