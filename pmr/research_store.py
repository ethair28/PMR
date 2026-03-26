from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from pmr.models import EvidenceItem, ResearchJob, ResearchResult


@dataclass(frozen=True, slots=True)
class ResearchCacheConfig:
    """Bounded-cache controls for persisted research evidence and results."""

    max_versions_per_job_provider: int = 2
    evidence_retention_days: int = 30
    result_retention_days: int = 90
    max_evidence_items_per_job: int = 50
    max_excerpt_chars: int = 2_000
    max_database_megabytes: int = 512


@dataclass(slots=True)
class ResearchStore:
    """SQLite-backed cache for normalized evidence and synthesized story outputs."""

    path: Path
    cache_config: ResearchCacheConfig = ResearchCacheConfig()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._initialize_schema(connection)

    def get_cached_result(self, cache_key: str) -> ResearchResult | None:
        with self._connect() as connection:
            self._initialize_schema(connection)
            row = connection.execute(
                """
                SELECT *
                FROM research_results
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            evidence_rows = connection.execute(
                """
                SELECT *
                FROM research_evidence
                WHERE cache_key = ?
                ORDER BY slot ASC
                """,
                (cache_key,),
            ).fetchall()
        key_evidence = tuple(_row_to_evidence_item(item) for item in evidence_rows if item["bucket"] == "key")
        contradictory_evidence = tuple(
            _row_to_evidence_item(item) for item in evidence_rows if item["bucket"] == "contradictory"
        )
        if not key_evidence:
            key_evidence = tuple(_deserialize_evidence_items(row["key_evidence_json"]))
        if not contradictory_evidence:
            contradictory_evidence = tuple(_deserialize_evidence_items(row["contradictory_evidence_json"]))
        return ResearchResult(
            job_id=row["job_id"],
            cache_key=row["cache_key"],
            provider=row["provider"],
            prompt_version=row["prompt_version"],
            model_name=row["model_name"] or "unknown",
            workflow_type=row["workflow_type"] or "repricing_story",
            status=row["status"],
            explanation_class=row["explanation_class"],
            confidence=float(row["confidence"]),
            most_plausible_explanation=row["most_plausible_explanation"],
            why_market_moved=row["why_market_moved"],
            price_action_summary=row["price_action_summary"] or "",
            surprise_assessment=row["surprise_assessment"] or "",
            main_narrative=row["main_narrative"] or "",
            alternative_explanations=tuple(json.loads(row["alternative_explanations_json"] or "[]")),
            note_to_editor=row["note_to_editor"] or "",
            draft_headline=row["draft_headline"] or "",
            draft_markdown=row["draft_markdown"] or "",
            key_evidence=key_evidence,
            contradictory_evidence=contradictory_evidence,
            open_questions=tuple(json.loads(row["open_questions_json"])),
            completed_at=datetime.fromisoformat(row["completed_at"]),
            error_message=row["error_message"],
            used_cache=True,
        )

    def upsert_result(self, job: ResearchJob, result: ResearchResult) -> None:
        key_evidence = tuple(result.key_evidence[: self.cache_config.max_evidence_items_per_job])
        contradictory_evidence = tuple(result.contradictory_evidence[: self.cache_config.max_evidence_items_per_job])
        with self._connect() as connection:
            self._initialize_schema(connection)
            connection.execute(
                """
                INSERT INTO research_results (
                    cache_key,
                    job_id,
                    provider,
                    prompt_version,
                    model_name,
                    workflow_type,
                    detection_window_end,
                    status,
                    explanation_class,
                    confidence,
                    most_plausible_explanation,
                    why_market_moved,
                    price_action_summary,
                    surprise_assessment,
                    main_narrative,
                    alternative_explanations_json,
                    note_to_editor,
                    draft_headline,
                    draft_markdown,
                    open_questions_json,
                    key_evidence_json,
                    contradictory_evidence_json,
                    completed_at,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider = excluded.provider,
                    prompt_version = excluded.prompt_version,
                    model_name = excluded.model_name,
                    workflow_type = excluded.workflow_type,
                    detection_window_end = excluded.detection_window_end,
                    status = excluded.status,
                    explanation_class = excluded.explanation_class,
                    confidence = excluded.confidence,
                    most_plausible_explanation = excluded.most_plausible_explanation,
                    why_market_moved = excluded.why_market_moved,
                    price_action_summary = excluded.price_action_summary,
                    surprise_assessment = excluded.surprise_assessment,
                    main_narrative = excluded.main_narrative,
                    alternative_explanations_json = excluded.alternative_explanations_json,
                    note_to_editor = excluded.note_to_editor,
                    draft_headline = excluded.draft_headline,
                    draft_markdown = excluded.draft_markdown,
                    open_questions_json = excluded.open_questions_json,
                    key_evidence_json = excluded.key_evidence_json,
                    contradictory_evidence_json = excluded.contradictory_evidence_json,
                    completed_at = excluded.completed_at,
                    error_message = excluded.error_message
                """,
                (
                    result.cache_key,
                    result.job_id,
                    result.provider,
                    result.prompt_version,
                    result.model_name,
                    result.workflow_type,
                    job.primary_market.detection_window_end.isoformat(),
                    result.status,
                    result.explanation_class,
                    result.confidence,
                    result.most_plausible_explanation,
                    result.why_market_moved,
                    result.price_action_summary,
                    result.surprise_assessment,
                    result.main_narrative,
                    json.dumps(list(result.alternative_explanations)),
                    result.note_to_editor,
                    result.draft_headline,
                    result.draft_markdown,
                    json.dumps(list(result.open_questions)),
                    json.dumps([_serialize_evidence_item(item) for item in key_evidence]),
                    json.dumps([_serialize_evidence_item(item) for item in contradictory_evidence]),
                    result.completed_at.isoformat(),
                    result.error_message,
                ),
            )
            connection.execute("DELETE FROM research_evidence WHERE cache_key = ?", (result.cache_key,))
            for bucket, items in (("key", key_evidence), ("contradictory", contradictory_evidence)):
                for slot, item in enumerate(items):
                    connection.execute(
                        """
                        INSERT INTO research_evidence (
                            cache_key,
                            bucket,
                            slot,
                            source_type,
                            url,
                            title_or_text,
                            author_or_publication,
                            published_at,
                            collected_at,
                            relevance_score,
                            temporal_proximity_score,
                            stance,
                            excerpt,
                            query
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            result.cache_key,
                            bucket,
                            slot,
                            item.source_type,
                            item.url,
                            item.title_or_text,
                            item.author_or_publication,
                            item.published_at.isoformat() if item.published_at else None,
                            item.collected_at.isoformat(),
                            item.relevance_score,
                            item.temporal_proximity_score,
                            item.stance,
                            _truncate(item.excerpt, self.cache_config.max_excerpt_chars),
                            item.query,
                        ),
                    )
            self._prune_versions(connection, job_id=result.job_id, provider=result.provider)
            connection.commit()

    def prune(self, *, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        evidence_cutoff = (now - timedelta(days=self.cache_config.evidence_retention_days)).isoformat()
        result_cutoff = (now - timedelta(days=self.cache_config.result_retention_days)).isoformat()
        with self._connect() as connection:
            self._initialize_schema(connection)
            connection.execute(
                "DELETE FROM research_evidence WHERE collected_at < ?",
                (evidence_cutoff,),
            )
            connection.execute(
                "DELETE FROM research_results WHERE completed_at < ?",
                (result_cutoff,),
            )
            rows = connection.execute(
                """
                SELECT DISTINCT job_id, provider
                FROM research_results
                """
            ).fetchall()
            for row in rows:
                self._prune_versions(connection, job_id=row["job_id"], provider=row["provider"])
            connection.commit()
        self._enforce_size_cap()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA auto_vacuum = INCREMENTAL")
        return connection

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_results (
                cache_key TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_name TEXT,
                workflow_type TEXT,
                detection_window_end TEXT NOT NULL,
                status TEXT NOT NULL,
                explanation_class TEXT,
                confidence REAL NOT NULL,
                most_plausible_explanation TEXT NOT NULL,
                why_market_moved TEXT NOT NULL,
                price_action_summary TEXT,
                surprise_assessment TEXT,
                main_narrative TEXT,
                alternative_explanations_json TEXT,
                note_to_editor TEXT,
                draft_headline TEXT,
                draft_markdown TEXT,
                open_questions_json TEXT NOT NULL,
                key_evidence_json TEXT NOT NULL,
                contradictory_evidence_json TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS research_evidence (
                cache_key TEXT NOT NULL,
                bucket TEXT NOT NULL,
                slot INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                url TEXT NOT NULL,
                title_or_text TEXT NOT NULL,
                author_or_publication TEXT,
                published_at TEXT,
                collected_at TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                temporal_proximity_score REAL NOT NULL,
                stance TEXT NOT NULL,
                excerpt TEXT NOT NULL,
                query TEXT,
                PRIMARY KEY (cache_key, bucket, slot),
                FOREIGN KEY (cache_key) REFERENCES research_results(cache_key) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_research_results_job_provider_completed
            ON research_results(job_id, provider, completed_at DESC);

            CREATE INDEX IF NOT EXISTS idx_research_results_completed
            ON research_results(completed_at);

            CREATE INDEX IF NOT EXISTS idx_research_evidence_collected
            ON research_evidence(collected_at);
            """
        )
        self._ensure_result_columns(connection)

    def _prune_versions(self, connection: sqlite3.Connection, *, job_id: str, provider: str) -> None:
        rows = connection.execute(
            """
            SELECT cache_key
            FROM research_results
            WHERE job_id = ? AND provider = ?
            ORDER BY completed_at DESC, detection_window_end DESC, cache_key DESC
            """,
            (job_id, provider),
        ).fetchall()
        to_delete = rows[self.cache_config.max_versions_per_job_provider :]
        for row in to_delete:
            connection.execute("DELETE FROM research_results WHERE cache_key = ?", (row["cache_key"],))

    def _enforce_size_cap(self) -> None:
        max_bytes = self.cache_config.max_database_megabytes * 1024 * 1024
        if not self.path.exists() or self.path.stat().st_size <= max_bytes:
            return
        with self._connect() as connection:
            self._initialize_schema(connection)
            while self.path.exists() and self.path.stat().st_size > max_bytes:
                row = connection.execute(
                    """
                    SELECT cache_key
                    FROM research_results
                    ORDER BY completed_at ASC, cache_key ASC
                    LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    break
                connection.execute("DELETE FROM research_results WHERE cache_key = ?", (row["cache_key"],))
                connection.commit()
            connection.execute("PRAGMA incremental_vacuum")
            connection.execute("VACUUM")

    def _ensure_result_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(research_results)").fetchall()
        }
        required_columns = {
            "model_name": "TEXT",
            "workflow_type": "TEXT",
            "price_action_summary": "TEXT",
            "surprise_assessment": "TEXT",
            "main_narrative": "TEXT",
            "alternative_explanations_json": "TEXT",
            "note_to_editor": "TEXT",
            "draft_headline": "TEXT",
            "draft_markdown": "TEXT",
        }
        for column_name, column_type in required_columns.items():
            if column_name in columns:
                continue
            connection.execute(
                f"ALTER TABLE research_results ADD COLUMN {column_name} {column_type}"
            )


def _serialize_evidence_item(item: EvidenceItem) -> dict[str, object]:
    return {
        "source_type": item.source_type,
        "url": item.url,
        "title_or_text": item.title_or_text,
        "author_or_publication": item.author_or_publication,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "collected_at": item.collected_at.isoformat(),
        "relevance_score": item.relevance_score,
        "temporal_proximity_score": item.temporal_proximity_score,
        "stance": item.stance,
        "excerpt": _truncate(item.excerpt, 2_000),
        "query": item.query,
    }


def _deserialize_evidence_items(raw: str) -> Sequence[EvidenceItem]:
    payload = json.loads(raw)
    return tuple(
        EvidenceItem(
            source_type=item["source_type"],
            url=item["url"],
            title_or_text=item["title_or_text"],
            author_or_publication=item.get("author_or_publication"),
            published_at=datetime.fromisoformat(item["published_at"]) if item.get("published_at") else None,
            collected_at=datetime.fromisoformat(item["collected_at"]),
            relevance_score=float(item["relevance_score"]),
            temporal_proximity_score=float(item["temporal_proximity_score"]),
            stance=item["stance"],
            excerpt=item["excerpt"],
            query=item.get("query"),
        )
        for item in payload
    )


def _row_to_evidence_item(row: sqlite3.Row) -> EvidenceItem:
    return EvidenceItem(
        source_type=row["source_type"],
        url=row["url"],
        title_or_text=row["title_or_text"],
        author_or_publication=row["author_or_publication"],
        published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
        collected_at=datetime.fromisoformat(row["collected_at"]),
        relevance_score=float(row["relevance_score"]),
        temporal_proximity_score=float(row["temporal_proximity_score"]),
        stance=row["stance"],
        excerpt=row["excerpt"],
        query=row["query"],
    )


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
