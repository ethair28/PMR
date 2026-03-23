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
                history_days=120,
                staleness_hours=48,
                max_markets=10,
                max_markets_per_category=5,
                now=datetime(2026, 3, 23, tzinfo=timezone.utc),
            )

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].market.market_id, "recent-market")
        self.assertTrue(all(snapshot.observed_at >= datetime(2026, 2, 21, tzinfo=timezone.utc) for snapshot in loaded[0].snapshots))

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


def _build_series(
    market_id: str,
    category: str,
    *,
    start: datetime,
    probabilities: list[float],
) -> MarketSeries:
    market = Market(
        market_id=market_id,
        question=f"Question for {market_id}?",
        category=category,
        tags=(category,),
        volume_7d_usd=200_000,
        volume_24h_usd=50_000,
        open_interest_usd=70_000,
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
