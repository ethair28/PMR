from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pmr.models import Market, MarketSeries, MarketSnapshot
from pmr.storage import SnapshotStore


class SnapshotStoreTests(unittest.TestCase):
    def test_store_persists_loads_and_prunes_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pmr.sqlite3"
            store = SnapshotStore(db_path)
            store.initialize()

            old_series = _build_series(
                market_id="old-market",
                category="politics",
                start=datetime(2025, 10, 1, tzinfo=timezone.utc),
                probabilities=[0.20, 0.21, 0.22],
            )
            recent_series = _build_series(
                market_id="recent-market",
                category="macro",
                start=datetime(2026, 2, 20, tzinfo=timezone.utc),
                probabilities=[0.30, 0.31, 0.33, 0.37, 0.36],
            )

            store.upsert_market_series(
                [old_series],
                fetched_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            store.upsert_market_series(
                [recent_series],
                fetched_at=datetime(2026, 3, 22, tzinfo=timezone.utc),
            )
            store.prune(
                retention_days=30,
                now=datetime(2026, 3, 23, tzinfo=timezone.utc),
            )

            loaded = store.load_market_series(
                target_categories=("politics", "geopolitics", "economics", "macro"),
                history_days=120,
                staleness_hours=48,
                max_markets=10,
                min_markets_per_category=1,
                max_category_share_of_universe=0.6,
                max_markets_per_category=None,
                max_contracts_per_event=2,
                now=datetime(2026, 3, 23, tzinfo=timezone.utc),
            )

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].market.market_id, "recent-market")
        self.assertTrue(all(snapshot.observed_at >= datetime(2026, 2, 21, tzinfo=timezone.utc) for snapshot in loaded[0].snapshots))

    def test_store_load_skips_public_market_proxy_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pmr.sqlite3"
            store = SnapshotStore(db_path)
            store.initialize()

            public_market_proxy = _build_series(
                market_id="btc-price-market",
                category="economics",
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                probabilities=[0.35, 0.42, 0.51],
                question="Will Bitcoin reach $80,000 in March?",
                description="This market will immediately resolve to Yes if any Binance 1 minute candle for BTC/USDT has a final High price equal to or above the listed price.",
                event_title="What price will Bitcoin hit in March?",
            )
            policy_market = _build_series(
                market_id="fed-market",
                category="macro",
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                probabilities=[0.25, 0.30, 0.38],
                question="Will the Fed cut rates by June?",
            )

            store.upsert_market_series(
                [public_market_proxy, policy_market],
                fetched_at=datetime(2026, 3, 22, tzinfo=timezone.utc),
            )

            loaded = store.load_market_series(
                target_categories=("politics", "geopolitics", "economics", "macro"),
                history_days=120,
                staleness_hours=48,
                max_markets=10,
                min_markets_per_category=1,
                max_category_share_of_universe=0.6,
                max_markets_per_category=None,
                max_contracts_per_event=2,
                now=datetime(2026, 3, 23, tzinfo=timezone.utc),
            )

        self.assertEqual([series.market.market_id for series in loaded], ["fed-market"])

    def test_store_load_skips_social_activity_tracker_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pmr.sqlite3"
            store = SnapshotStore(db_path)
            store.initialize()

            social_tracker = _build_series(
                market_id="elon-tweet-market",
                category="politics",
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                probabilities=[0.10, 0.24, 0.41],
                question="Will Elon Musk post 380-399 tweets this week?",
                description="This market resolves according to the number of times Elon Musk posts on X this week. The resolution source is the Post Counter at xtracker.polymarket.com.",
                event_title="Elon Musk # tweets this week",
            )
            policy_market = _build_series(
                market_id="fed-market",
                category="macro",
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                probabilities=[0.25, 0.30, 0.38],
                question="Will the Fed cut rates by June?",
            )

            store.upsert_market_series(
                [social_tracker, policy_market],
                fetched_at=datetime(2026, 3, 22, tzinfo=timezone.utc),
            )

            loaded = store.load_market_series(
                target_categories=("politics", "geopolitics", "economics", "macro"),
                history_days=120,
                staleness_hours=48,
                max_markets=10,
                min_markets_per_category=1,
                max_category_share_of_universe=0.6,
                max_markets_per_category=None,
                max_contracts_per_event=2,
                now=datetime(2026, 3, 23, tzinfo=timezone.utc),
            )

        self.assertEqual([series.market.market_id for series in loaded], ["fed-market"])

    def test_store_lists_market_ids_and_snapshot_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pmr.sqlite3"
            store = SnapshotStore(db_path)
            store.initialize()

            first_series = _build_series(
                market_id="market-a",
                category="politics",
                start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                probabilities=[0.20, 0.26, 0.31],
            )
            second_series = _build_series(
                market_id="market-b",
                category="macro",
                start=datetime(2026, 3, 5, tzinfo=timezone.utc),
                probabilities=[0.45, 0.44, 0.47],
            )

            store.upsert_market_series(
                [first_series, second_series],
                fetched_at=datetime(2026, 3, 22, tzinfo=timezone.utc),
            )

            market_ids = store.list_market_ids()
            bounds = store.get_snapshot_bounds(market_ids)

        self.assertEqual(set(market_ids), {"market-a", "market-b"})
        self.assertEqual(
            bounds["market-a"].earliest_observed_at,
            datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(
            bounds["market-a"].latest_observed_at,
            datetime(2026, 3, 21, tzinfo=timezone.utc),
        )

    def test_store_uses_event_aggregated_liquidity_and_caps_sibling_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pmr.sqlite3"
            store = SnapshotStore(db_path)
            store.initialize()

            grouped_series = [
                _build_series(
                    market_id="slovenia-golob",
                    category="politics",
                    start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    probabilities=[0.20, 0.35, 0.58],
                    question="Will Robert Golob be the next Prime Minister of Slovenia?",
                    event_title="Next Prime Minister of Slovenia",
                    volume_7d_usd=310_000,
                    volume_24h_usd=35_000,
                ),
                _build_series(
                    market_id="slovenia-jansa",
                    category="politics",
                    start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    probabilities=[0.70, 0.48, 0.19],
                    question="Will Janez Janša be the next Prime Minister of Slovenia?",
                    event_title="Next Prime Minister of Slovenia",
                    volume_7d_usd=286_000,
                    volume_24h_usd=18_000,
                ),
                _build_series(
                    market_id="slovenia-logar",
                    category="politics",
                    start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    probabilities=[0.05, 0.03, 0.01],
                    question="Will Anže Logar be the next Prime Minister of Slovenia?",
                    event_title="Next Prime Minister of Slovenia",
                    volume_7d_usd=166_000,
                    volume_24h_usd=14_000,
                ),
            ]
            singletons = [
                _build_series(
                    market_id="other-politics",
                    category="politics",
                    start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    probabilities=[0.45, 0.46, 0.47],
                    question="Will Candidate A win the election?",
                    event_title="Candidate A election",
                    volume_7d_usd=400_000,
                    volume_24h_usd=60_000,
                ),
                _build_series(
                    market_id="geo-market",
                    category="geopolitics",
                    start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    probabilities=[0.30, 0.32, 0.35],
                    question="US x Iran ceasefire by June?",
                    event_title="US x Iran ceasefire by June?",
                    volume_7d_usd=340_000,
                    volume_24h_usd=50_000,
                ),
                _build_series(
                    market_id="macro-market",
                    category="macro",
                    start=datetime(2026, 3, 1, tzinfo=timezone.utc),
                    probabilities=[0.25, 0.28, 0.31],
                    question="Will the Fed cut rates by June?",
                    event_title="Fed cut rates by June?",
                    volume_7d_usd=330_000,
                    volume_24h_usd=45_000,
                ),
            ]

            store.upsert_market_series(
                [*grouped_series, *singletons],
                fetched_at=datetime(2026, 3, 22, tzinfo=timezone.utc),
            )

            loaded = store.load_market_series(
                target_categories=("politics", "geopolitics", "economics", "macro"),
                history_days=120,
                staleness_hours=48,
                max_markets=4,
                min_markets_per_category=1,
                max_category_share_of_universe=0.75,
                max_markets_per_category=None,
                max_contracts_per_event=2,
                now=datetime(2026, 3, 23, tzinfo=timezone.utc),
            )

        self.assertEqual(
            {series.market.market_id for series in loaded},
            {"slovenia-golob", "slovenia-jansa", "geo-market", "macro-market"},
        )


def _build_series(
    market_id: str,
    category: str,
    *,
    start: datetime,
    probabilities: list[float],
    question: str | None = None,
    description: str | None = None,
    event_title: str | None = None,
    volume_7d_usd: float = 200_000,
    volume_24h_usd: float = 50_000,
    open_interest_usd: float = 70_000,
) -> MarketSeries:
    market = Market(
        market_id=market_id,
        question=question or f"Question for {market_id}?",
        category=category,
        tags=(category,),
        description=description,
        event_title=event_title,
        volume_7d_usd=volume_7d_usd,
        volume_24h_usd=volume_24h_usd,
        open_interest_usd=open_interest_usd,
    )
    snapshots = tuple(
        MarketSnapshot(
            observed_at=start + timedelta(days=index * 10),
            probability=probability,
        )
        for index, probability in enumerate(probabilities)
    )
    return MarketSeries(market=market, snapshots=snapshots)


if __name__ == "__main__":
    unittest.main()
