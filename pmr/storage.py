from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from pmr.market_filters import classify_universe_market_exclusion
from pmr.models import Market, MarketSeries, MarketSnapshot
from pmr.universe_selection import UniverseCandidate, market_priority_sort_key, prioritize_universe_candidates


@dataclass(frozen=True, slots=True)
class SnapshotBounds:
    earliest_observed_at: datetime
    latest_observed_at: datetime


@dataclass(slots=True)
class SnapshotStore:
    """SQLite-backed store for bounded market metadata and snapshots."""

    path: Path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._initialize_schema(connection)

    def upsert_market_series(
        self,
        markets: Sequence[MarketSeries],
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        if not markets:
            return

        fetched_at = fetched_at or datetime.now(timezone.utc)
        with self._connect() as connection:
            self._initialize_schema(connection)
            for series in markets:
                self._upsert_market(connection, series, fetched_at=fetched_at)
                self._upsert_snapshots(connection, series)
            connection.commit()

    def prune(
        self,
        *,
        retention_days: int,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)
        snapshot_cutoff = now - timedelta(days=retention_days)
        last_seen_cutoff = snapshot_cutoff.isoformat()
        with self._connect() as connection:
            self._initialize_schema(connection)
            connection.execute(
                "DELETE FROM snapshots WHERE observed_at < ?",
                (snapshot_cutoff.isoformat(),),
            )
            connection.execute(
                """
                DELETE FROM markets
                WHERE last_seen_at < ?
                   OR NOT EXISTS (
                        SELECT 1
                        FROM snapshots
                        WHERE snapshots.market_id = markets.market_id
                   )
                """,
                (last_seen_cutoff,),
            )
            connection.commit()

    def load_market_series(
        self,
        *,
        target_categories: Sequence[str],
        history_days: int,
        staleness_hours: int,
        max_markets: int,
        min_markets_per_category: int,
        max_category_share_of_universe: float | None,
        max_markets_per_category: int | None,
        now: datetime | None = None,
    ) -> tuple[MarketSeries, ...]:
        now = now or datetime.now(timezone.utc)
        min_snapshot_time = now - timedelta(days=history_days)
        min_last_seen = now - timedelta(hours=staleness_hours)
        results: list[MarketSeries] = []

        with self._connect() as connection:
            self._initialize_schema(connection)
            market_rows = connection.execute(
                """
                SELECT *
                FROM markets
                WHERE last_seen_at >= ?
                ORDER BY volume_7d_usd DESC, volume_24h_usd DESC, market_id ASC
                """,
                (min_last_seen.isoformat(),),
            ).fetchall()

            prioritized_rows = prioritize_universe_candidates(
                [
                    UniverseCandidate(
                        item=row,
                        category=row["category"],
                        sort_key=market_priority_sort_key(
                            volume_7d=float(row["volume_7d_usd"]),
                            volume_24h=float(row["volume_24h_usd"]),
                            depth=float(row["open_interest_usd"]),
                            market_id=row["market_id"],
                        ),
                    )
                    for row in market_rows
                ],
                target_categories=tuple(target_categories),
                max_items=max_markets,
                min_items_per_category=min_markets_per_category,
                max_category_share=max_category_share_of_universe,
            )

            per_category_counts: dict[str, int] = {}
            for wrapper in prioritized_rows:
                row = wrapper.item
                category = row["category"]
                exclusion_reason = classify_universe_market_exclusion(
                    question=row["question"],
                    description=row["description"],
                    slug=row["slug"],
                    event_title=row["event_title"],
                )
                if exclusion_reason is not None:
                    continue
                if len(results) >= max_markets:
                    break
                if max_markets_per_category is not None and per_category_counts.get(category, 0) >= max_markets_per_category:
                    continue

                snapshot_rows = connection.execute(
                    """
                    SELECT observed_at, probability
                    FROM snapshots
                    WHERE market_id = ?
                      AND observed_at >= ?
                    ORDER BY observed_at ASC
                    """,
                    (row["market_id"], min_snapshot_time.isoformat()),
                ).fetchall()
                if len(snapshot_rows) < 2:
                    continue

                results.append(_row_to_market_series(row, snapshot_rows))
                per_category_counts[category] = per_category_counts.get(category, 0) + 1
                if len(results) >= max_markets:
                    break

        return tuple(results)

    def get_snapshot_bounds(
        self,
        market_ids: Sequence[str],
    ) -> dict[str, SnapshotBounds]:
        if not market_ids:
            return {}

        placeholders = ", ".join("?" for _ in market_ids)
        with self._connect() as connection:
            self._initialize_schema(connection)
            rows = connection.execute(
                f"""
                SELECT
                    market_id,
                    MIN(observed_at) AS earliest_observed_at,
                    MAX(observed_at) AS latest_observed_at
                FROM snapshots
                WHERE market_id IN ({placeholders})
                GROUP BY market_id
                """,
                tuple(market_ids),
            ).fetchall()

        return {
            row["market_id"]: SnapshotBounds(
                earliest_observed_at=datetime.fromisoformat(row["earliest_observed_at"]),
                latest_observed_at=datetime.fromisoformat(row["latest_observed_at"]),
            )
            for row in rows
        }

    def list_market_ids(self) -> tuple[str, ...]:
        """Return all tracked market identifiers currently stored in SQLite."""

        with self._connect() as connection:
            self._initialize_schema(connection)
            rows = connection.execute(
                """
                SELECT market_id
                FROM markets
                ORDER BY last_seen_at DESC, market_id ASC
                """
            ).fetchall()
        return tuple(row["market_id"] for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                category TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                slug TEXT,
                url TEXT,
                description TEXT,
                condition_id TEXT,
                tracked_outcome TEXT,
                tracked_token_id TEXT,
                event_title TEXT,
                volume_7d_usd REAL NOT NULL,
                volume_24h_usd REAL NOT NULL,
                open_interest_usd REAL NOT NULL,
                research_hints_json TEXT NOT NULL,
                notes TEXT,
                last_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                market_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                probability REAL NOT NULL,
                PRIMARY KEY (market_id, observed_at),
                FOREIGN KEY (market_id) REFERENCES markets(market_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_market_time
                ON snapshots (market_id, observed_at);
            CREATE INDEX IF NOT EXISTS idx_markets_last_seen
                ON markets (last_seen_at);
            """
        )

    def _upsert_market(
        self,
        connection: sqlite3.Connection,
        series: MarketSeries,
        *,
        fetched_at: datetime,
    ) -> None:
        market = series.market
        connection.execute(
            """
            INSERT INTO markets (
                market_id,
                question,
                category,
                tags_json,
                slug,
                url,
                description,
                condition_id,
                tracked_outcome,
                tracked_token_id,
                event_title,
                volume_7d_usd,
                volume_24h_usd,
                open_interest_usd,
                research_hints_json,
                notes,
                last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                question = excluded.question,
                category = excluded.category,
                tags_json = excluded.tags_json,
                slug = excluded.slug,
                url = excluded.url,
                description = excluded.description,
                condition_id = excluded.condition_id,
                tracked_outcome = excluded.tracked_outcome,
                tracked_token_id = excluded.tracked_token_id,
                event_title = excluded.event_title,
                volume_7d_usd = excluded.volume_7d_usd,
                volume_24h_usd = excluded.volume_24h_usd,
                open_interest_usd = excluded.open_interest_usd,
                research_hints_json = excluded.research_hints_json,
                notes = excluded.notes,
                last_seen_at = excluded.last_seen_at
            """,
            (
                market.market_id,
                market.question,
                market.category,
                json.dumps(list(market.tags)),
                market.slug,
                market.url,
                market.description,
                market.condition_id,
                market.tracked_outcome,
                market.tracked_token_id,
                market.event_title,
                market.volume_7d_usd,
                market.volume_24h_usd,
                market.open_interest_usd,
                json.dumps(list(series.research_hints)),
                series.notes,
                fetched_at.isoformat(),
            ),
        )

    def _upsert_snapshots(
        self,
        connection: sqlite3.Connection,
        series: MarketSeries,
    ) -> None:
        connection.executemany(
            """
            INSERT INTO snapshots (market_id, observed_at, probability)
            VALUES (?, ?, ?)
            ON CONFLICT(market_id, observed_at) DO UPDATE SET
                probability = excluded.probability
            """,
            [
                (
                    series.market.market_id,
                    snapshot.observed_at.isoformat(),
                    snapshot.probability,
                )
                for snapshot in series.snapshots
            ],
        )


def _row_to_market_series(
    market_row: sqlite3.Row,
    snapshot_rows: Sequence[sqlite3.Row],
) -> MarketSeries:
    market = Market(
        market_id=market_row["market_id"],
        question=market_row["question"],
        category=market_row["category"],
        tags=tuple(json.loads(market_row["tags_json"])),
        slug=market_row["slug"],
        url=market_row["url"],
        description=market_row["description"],
        condition_id=market_row["condition_id"],
        tracked_outcome=market_row["tracked_outcome"],
        tracked_token_id=market_row["tracked_token_id"],
        event_title=market_row["event_title"],
        volume_7d_usd=float(market_row["volume_7d_usd"]),
        volume_24h_usd=float(market_row["volume_24h_usd"]),
        open_interest_usd=float(market_row["open_interest_usd"]),
    )
    snapshots = tuple(
        MarketSnapshot(
            observed_at=datetime.fromisoformat(row["observed_at"]),
            probability=float(row["probability"]),
        )
        for row in snapshot_rows
    )
    return MarketSeries(
        market=market,
        snapshots=snapshots,
        research_hints=tuple(json.loads(market_row["research_hints_json"])),
        notes=market_row["notes"],
    )
