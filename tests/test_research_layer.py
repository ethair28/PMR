from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pmr.models import (
    EvidenceItem,
    RelatedMarket,
    ResearchJob,
    ResearchMarketContext,
    ResearchResult,
)
from pmr.research_engine import (
    HeuristicResearchSynthesizer,
    ResearchEngine,
    ResearchSource,
    build_research_query_plan,
    rank_evidence_for_job,
)
from pmr.research_cli import _load_dotenv_if_present
from pmr.research_payloads import build_research_results_payload, load_research_jobs_from_payload
from pmr.research_store import ResearchCacheConfig, ResearchStore
from pmr.research_xai import _choose_best_model_name, _normalize_api_host


class ResearchPayloadTests(unittest.TestCase):
    def test_load_research_jobs_from_payload_parses_current_shape(self) -> None:
        payload = {
            "research_jobs": [
                {
                    "job_id": "job-1",
                    "story": {
                        "family_key": "trump-visit-china",
                        "family_label": "Trump Visit China",
                        "story_type_hint": "live_repricing",
                        "distance_from_extremes": 0.22,
                        "entered_extreme_zone": False,
                        "editorial_priority_hint": "high",
                    },
                    "investigation": {
                        "question": "What explains the repricing?",
                        "why_flagged": "Flagged because of a 28-point move.",
                        "focus_points": ["Search X first", "Look near the move timestamp"],
                    },
                    "primary_market": {
                        "market": {
                            "market_id": "market-1",
                            "question": "Will Trump visit China by April 30?",
                            "slug": "trump-visit-china",
                            "url": "https://polymarket.com/event/trump-visit-china",
                            "description": "Test market",
                            "category": "politics",
                            "tags": ["china"],
                            "condition_id": "cond-1",
                            "tracked_outcome": "Yes",
                            "tracked_token_id": "token-1",
                            "event_title": "Trump visit China",
                        },
                        "story": {},
                        "detector": {
                            "detection_window_start": "2026-03-17T00:00:00+00:00",
                            "detection_window_end": "2026-03-24T00:00:00+00:00",
                            "history_mode": "full_history",
                            "confidence_level": "high",
                            "confidence_score": 0.86,
                            "composite_score": 8.47,
                            "eligible_for_ranking": True,
                            "exclusion_reason": None,
                            "max_move_timestamp": "2026-03-24T09:00:00+00:00",
                        },
                        "features": {
                            "close_to_open_move": -0.22,
                            "max_abs_move_24h": 0.28,
                            "weekly_range": 0.35,
                        },
                        "notes": ["note 1"],
                    },
                    "related_markets": [{"market_id": "market-2", "question": "Will Trump visit China by March 31?"}],
                }
            ]
        }

        jobs = load_research_jobs_from_payload(payload)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, "job-1")
        self.assertEqual(jobs[0].primary_market.question, "Will Trump visit China by April 30?")
        self.assertEqual(jobs[0].related_markets[0].market_id, "market-2")

    def test_dotenv_loader_trims_whitespace_and_preserves_existing_env(self) -> None:
        original = os.environ.get("XAI_API_KEY")
        os.environ.pop("XAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                env_path = Path(temp_dir) / ".env"
                env_path.write_text('XAI_API_KEY = "demo-key"\nPMR_XAI_MODEL = grok-4.20-reasoning-latest\n')
                _load_dotenv_if_present(env_path)
                self.assertEqual(os.environ["XAI_API_KEY"], "demo-key")
                self.assertEqual(os.environ["PMR_XAI_MODEL"], "grok-4.20-reasoning-latest")

                os.environ["XAI_API_KEY"] = "already-set"
                env_path.write_text("XAI_API_KEY=should-not-win\n")
                _load_dotenv_if_present(env_path)
                self.assertEqual(os.environ["XAI_API_KEY"], "already-set")
        finally:
            if original is None:
                os.environ.pop("XAI_API_KEY", None)
            else:
                os.environ["XAI_API_KEY"] = original
            os.environ.pop("PMR_XAI_MODEL", None)


class ResearchEngineTests(unittest.TestCase):
    def test_build_query_plan_uses_primary_market_and_related_markets(self) -> None:
        job = _build_job()

        plan = build_research_query_plan(job)

        self.assertIn("Will Trump visit China by April 30?", plan.x_queries)
        self.assertIn("Trump Visit China", plan.web_queries)
        self.assertIsNotNone(plan.focus_timestamp)

    def test_rank_evidence_dedupes_and_prefers_higher_score(self) -> None:
        job = _build_job()
        low = _build_evidence(
            url="https://example.com/item",
            relevance_score=0.4,
            temporal_score=0.2,
            title="Lower score",
        )
        high = _build_evidence(
            url="https://example.com/item",
            relevance_score=0.8,
            temporal_score=0.7,
            title="Higher score",
        )

        ranked = rank_evidence_for_job(job=job, evidence=(low, high), max_items=10)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].title_or_text, "Higher score")

    def test_heuristic_synthesizer_marks_insufficient_when_no_evidence(self) -> None:
        job = _build_job()
        synthesizer = HeuristicResearchSynthesizer()
        plan = build_research_query_plan(job)

        result = synthesizer.summarize(
            job,
            plan,
            (),
            cache_key="cache-1",
            provider_name="test",
            prompt_version="v1",
            generated_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.explanation_class, "speculative")
        self.assertIn("not surface enough evidence", result.most_plausible_explanation)

    def test_batch_continues_after_one_job_failure(self) -> None:
        jobs = (_build_job("job-1"), _build_job("job-2"))
        engine = ResearchEngine(
            source=_MixedSource(
                success_job_id="job-1",
                evidence=(_build_evidence(url="https://x.com/post/1"),),
            ),
            synthesizer=HeuristicResearchSynthesizer(),
            provider_name="test",
            prompt_version="v1",
        )

        batch = engine.run_batch(jobs)

        self.assertEqual(batch.processed_jobs, 2)
        self.assertEqual(batch.failed_jobs, 1)
        self.assertEqual(batch.results[0].status, "completed")
        self.assertEqual(batch.results[1].status, "failed")


class XaiSdkAdapterHelperTests(unittest.TestCase):
    def test_normalize_api_host_accepts_base_url(self) -> None:
        self.assertEqual(_normalize_api_host("https://api.x.ai/v1"), "api.x.ai")
        self.assertEqual(_normalize_api_host("https://api.x.ai:443/v1"), "api.x.ai:443")

    def test_choose_best_model_name_prefers_latest_reasoning_candidate(self) -> None:
        available = {"grok-4-1-fast-reasoning-latest", "grok-4.20", "grok-3-mini"}
        self.assertEqual(_choose_best_model_name(available), "grok-4.20")

    def test_choose_best_model_name_falls_back_to_default_when_unknown(self) -> None:
        self.assertEqual(_choose_best_model_name({"grok-3-mini"}), "grok-4.20-reasoning-latest")

    def test_choose_best_model_name_uses_any_grok4_reasoning_before_multi_agent(self) -> None:
        available = {"grok-4-fast-reasoning", "grok-4.20-multi-agent-latest"}
        self.assertEqual(_choose_best_model_name(available), "grok-4-fast-reasoning")


class ResearchStoreTests(unittest.TestCase):
    def test_store_returns_cached_result(self) -> None:
        job = _build_job()
        result = _build_result(job, cache_key="cache-1", completed_at=datetime(2026, 3, 25, tzinfo=timezone.utc))
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(Path(temp_dir) / "research.sqlite3")
            store.initialize()
            store.upsert_result(job, result)

            cached = store.get_cached_result("cache-1")

        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertTrue(cached.used_cache)
        self.assertEqual(cached.job_id, "job-1")

    def test_store_prunes_old_versions_per_job_provider(self) -> None:
        job = _build_job()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(
                Path(temp_dir) / "research.sqlite3",
                cache_config=ResearchCacheConfig(max_versions_per_job_provider=2),
            )
            store.initialize()
            store.upsert_result(job, _build_result(job, cache_key="cache-1", completed_at=datetime(2026, 3, 20, tzinfo=timezone.utc)))
            store.upsert_result(job, _build_result(job, cache_key="cache-2", completed_at=datetime(2026, 3, 21, tzinfo=timezone.utc)))
            store.upsert_result(job, _build_result(job, cache_key="cache-3", completed_at=datetime(2026, 3, 22, tzinfo=timezone.utc)))

            with store._connect() as connection:
                rows = connection.execute("SELECT cache_key FROM research_results ORDER BY completed_at ASC").fetchall()

        self.assertEqual([row["cache_key"] for row in rows], ["cache-2", "cache-3"])

    def test_store_prunes_old_results_by_age(self) -> None:
        job = _build_job()
        old_result = _build_result(job, cache_key="cache-old", completed_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(
                Path(temp_dir) / "research.sqlite3",
                cache_config=ResearchCacheConfig(result_retention_days=30),
            )
            store.initialize()
            store.upsert_result(job, old_result)
            store.prune(now=datetime(2026, 3, 25, tzinfo=timezone.utc))

            cached = store.get_cached_result("cache-old")

        self.assertIsNone(cached)

    def test_store_enforces_size_cap(self) -> None:
        job = _build_job()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ResearchStore(
                Path(temp_dir) / "research.sqlite3",
                cache_config=ResearchCacheConfig(max_database_megabytes=0),
            )
            store.initialize()
            store.upsert_result(job, _build_result(job, cache_key="cache-1", completed_at=datetime(2026, 3, 25, tzinfo=timezone.utc)))
            store.prune(now=datetime(2026, 3, 25, tzinfo=timezone.utc))

            with store._connect() as connection:
                remaining = connection.execute("SELECT COUNT(*) AS count FROM research_results").fetchone()["count"]

        self.assertEqual(remaining, 0)


class ResearchResultPayloadTests(unittest.TestCase):
    def test_build_research_results_payload_is_json_canonical(self) -> None:
        job = _build_job()
        result = _build_result(job, cache_key="cache-1", completed_at=datetime(2026, 3, 25, tzinfo=timezone.utc))
        from pmr.models import ResearchBatchResult

        batch = ResearchBatchResult(
            provider="test",
            prompt_version="v1",
            generated_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
            results=(result,),
        )

        payload = build_research_results_payload(batch)

        self.assertEqual(payload["processed_jobs"], 1)
        self.assertEqual(payload["results"][0]["job_id"], "job-1")
        self.assertNotIn("brief_markdown", payload["results"][0])


def _build_job(job_id: str = "job-1") -> ResearchJob:
    return ResearchJob(
        job_id=job_id,
        family_key="trump-visit-china",
        family_label="Trump Visit China",
        story_type_hint="live_repricing",
        distance_from_extremes=0.22,
        entered_extreme_zone=False,
        editorial_priority_hint="high",
        investigation_question="What most plausibly explains the repricing?",
        why_flagged="Flagged because the market moved 28 points in 24 hours.",
        focus_points=("Search X first", "Look near the move timestamp"),
        primary_market=ResearchMarketContext(
            market_id="market-1",
            question="Will Trump visit China by April 30?",
            detection_window_start=datetime(2026, 3, 17, tzinfo=timezone.utc),
            detection_window_end=datetime(2026, 3, 24, tzinfo=timezone.utc),
            history_mode="full_history",
            confidence_level="high",
            confidence_score=0.86,
            composite_score=8.47,
            close_to_open_move=-0.22,
            max_abs_move_24h=0.28,
            weekly_range=0.35,
            max_move_timestamp=datetime(2026, 3, 24, 9, tzinfo=timezone.utc),
            category="politics",
            url="https://polymarket.com/event/trump-visit-china",
            notes=("Detector note",),
        ),
        related_markets=(RelatedMarket(market_id="market-2", question="Will Trump visit China by March 31?"),),
    )


def _build_evidence(
    *,
    url: str,
    relevance_score: float = 0.8,
    temporal_score: float = 0.8,
    title: str = "Evidence item",
    stance: str = "supporting",
) -> EvidenceItem:
    return EvidenceItem(
        source_type="web_article",
        url=url,
        title_or_text=title,
        author_or_publication="Reporter",
        published_at=datetime(2026, 3, 24, 8, tzinfo=timezone.utc),
        collected_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        relevance_score=relevance_score,
        temporal_proximity_score=temporal_score,
        stance=stance,
        excerpt="Short excerpt",
        query="Trump Visit China",
    )


def _build_result(job: ResearchJob, *, cache_key: str, completed_at: datetime) -> ResearchResult:
    return ResearchResult(
        job_id=job.job_id,
        cache_key=cache_key,
        provider="test",
        prompt_version="v1",
        status="completed",
        explanation_class="plausible",
        confidence=0.61,
        most_plausible_explanation="Likely driven by diplomatic signaling and new chatter around a possible visit.",
        why_market_moved=job.why_flagged,
        key_evidence=(_build_evidence(url="https://example.com/1"),),
        contradictory_evidence=(),
        open_questions=("Was there a direct leak or only rumor-driven speculation?",),
        completed_at=completed_at,
    )


class _MixedSource(ResearchSource):
    def __init__(self, *, success_job_id: str, evidence: tuple[EvidenceItem, ...]) -> None:
        self.success_job_id = success_job_id
        self.evidence = evidence

    def search(self, job: ResearchJob, query_plan: object) -> tuple[EvidenceItem, ...]:
        if job.job_id != self.success_job_id:
            raise RuntimeError("synthetic failure")
        return self.evidence
