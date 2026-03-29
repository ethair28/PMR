from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pmr.models import (
    EvidenceItem,
    FollowUpQuery,
    HypothesisAssessment,
    InvestigationLead,
    InvestigationPlan,
    PriceTracePoint,
    RelatedMarket,
    ResearchJob,
    ResearchMarketContext,
    ResearchPriceContext,
    ResearchResult,
)
from pmr.research_engine import (
    HeuristicResearchSynthesizer,
    ResearchEngine,
    ResearchPlanner,
    ResearchSource,
    build_research_query_plan,
    rank_evidence_for_job,
    select_follow_up_queries,
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
                        "workflow_type": "repricing_story",
                        "story_type_hint": "live_repricing",
                        "distance_from_extremes": 0.22,
                        "entered_extreme_zone": False,
                        "editorial_priority_hint": "high",
                        "story_role_hint": "standalone",
                        "overlap_group_key": None,
                        "overlap_summary": None,
                        "suggested_merge_with": [],
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
                            "window_open_probability": 0.62,
                            "window_close_probability": 0.40,
                            "window_high_probability": 0.68,
                            "window_low_probability": 0.33,
                            "close_to_open_move": -0.22,
                            "max_abs_move_6h": 0.14,
                            "max_abs_move_24h": 0.28,
                            "largest_6h_move": -0.14,
                            "largest_24h_move": -0.28,
                            "weekly_range": 0.35,
                            "persistence_of_largest_move": 0.61,
                            "jump_count_over_threshold": 2,
                        },
                        "price_context": {
                            "interval_hours": 8,
                            "chart_interval_minutes": 60,
                            "largest_move_window_hours": 24,
                            "largest_move_window_start": "2026-03-23T09:00:00+00:00",
                            "largest_move_window_end": "2026-03-24T09:00:00+00:00",
                            "surprise_reference_probability": None,
                            "surprise_points": None,
                            "surprise_label": None,
                            "trace_points": [
                                {
                                    "observed_at": "2026-03-17T00:00:00+00:00",
                                    "probability": 0.62,
                                    "move_since_previous": None
                                },
                                {
                                    "observed_at": "2026-03-17T08:00:00+00:00",
                                    "probability": 0.58,
                                    "move_since_previous": -0.04
                                }
                            ],
                            "chart_trace_points": [
                                {
                                    "observed_at": "2026-03-17T00:00:00+00:00",
                                    "probability": 0.62,
                                    "move_since_previous": None
                                },
                                {
                                    "observed_at": "2026-03-17T01:00:00+00:00",
                                    "probability": 0.60,
                                    "move_since_previous": -0.02
                                }
                            ],
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
        self.assertEqual(jobs[0].workflow_type, "repricing_story")
        self.assertEqual(jobs[0].story_role_hint, "standalone")
        self.assertEqual(jobs[0].primary_market.question, "Will Trump visit China by April 30?")
        self.assertEqual(jobs[0].primary_market.price_context.interval_hours, 8)
        self.assertEqual(jobs[0].primary_market.price_context.chart_interval_minutes, 60)
        self.assertEqual(len(jobs[0].primary_market.price_context.chart_trace_points), 2)
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

    def test_rank_evidence_penalizes_wikipedia_and_market_commentary(self) -> None:
        job = _build_job()
        strong_news = _build_evidence(
            url="https://www.reuters.com/world/example",
            title="Reuters report",
            relevance_score=0.7,
            temporal_score=0.7,
        )
        wikipedia = _build_evidence(
            url="https://en.wikipedia.org/wiki/Example",
            title="Wikipedia page",
            relevance_score=0.9,
            temporal_score=0.9,
        )
        market_commentary = _build_evidence(
            url="https://x.com/AgentOnChain/status/1",
            title="Polymarket odds moved on this story",
            relevance_score=0.9,
            temporal_score=0.9,
            source_type="x_post",
            author="AgentOnChain",
        )

        ranked = rank_evidence_for_job(job=job, evidence=(wikipedia, market_commentary, strong_news), max_items=3)

        self.assertEqual(ranked[0].url, "https://www.reuters.com/world/example")
        self.assertEqual(ranked[0].quality_tier, "primary")
        self.assertEqual(ranked[1].quality_tier, "weak_context")
        self.assertEqual(ranked[2].quality_tier, "weak_context")

    def test_rank_evidence_excludes_recursive_grok_posts(self) -> None:
        job = _build_job()
        recursive = _build_evidence(
            url="https://x.com/grok/status/1",
            title="Grok explains why the market moved",
            source_type="x_post",
            author="Grok",
        )
        strong_news = _build_evidence(
            url="https://www.reuters.com/world/example",
            title="Reuters report",
        )

        ranked = rank_evidence_for_job(job=job, evidence=(recursive, strong_news), max_items=3)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].url, "https://www.reuters.com/world/example")

    def test_heuristic_synthesizer_marks_insufficient_when_no_evidence(self) -> None:
        job = _build_job()
        synthesizer = HeuristicResearchSynthesizer()
        plan = build_research_query_plan(job)

        result = synthesizer.summarize(
            job,
            plan,
            (),
            investigation_plan=None,
            cache_key="cache-1",
            provider_name="test",
            prompt_version="v1",
            generated_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.explanation_class, "speculative")
        self.assertIn("not surface enough evidence", result.most_plausible_explanation)
        self.assertEqual(result.belief_shift_drivers, ())
        self.assertEqual(result.signal_types, ())
        self.assertTrue(result.why_now)

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

    def test_repricing_job_uses_planner_and_follow_up_search(self) -> None:
        job = _build_job()
        source = _PlannerAwareSource(
            initial_evidence=(_build_evidence(url="https://example.com/initial", title="Initial report"),),
            follow_up_evidence=(
                _build_evidence(
                    url="https://example.com/follow-up",
                    title="Follow-up validation",
                    temporal_score=0.9,
                ),
            ),
        )
        planner = _StaticPlanner(
            InvestigationPlan(
                job_id=job.job_id,
                candidate_explanations=(
                    InvestigationLead(
                        label="Lead 1",
                        hypothesis="Diplomatic signaling deteriorated.",
                        supporting_signals=("Initial report",),
                        missing_evidence=("Need confirmation from follow-up reporting.",),
                        priority="high",
                    ),
                ),
                leading_hypothesis="Diplomatic signaling deteriorated.",
                follow_up_queries=(
                    FollowUpQuery(
                        query="Trump visit China Reuters follow-up",
                        source_type="web",
                        reason="Validate the strongest initial lead.",
                    ),
                ),
                skeptical_query=FollowUpQuery(
                    query="Trump visit China denial",
                    source_type="x",
                    reason="Check for direct pushback against the lead explanation.",
                    skeptical=True,
                ),
                assessments=(
                    HypothesisAssessment(
                        hypothesis="Diplomatic signaling deteriorated.",
                        support_level="mixed",
                        contradictions=("No official confirmation yet.",),
                        open_uncertainty=("Timing still depends on informal chatter.",),
                    ),
                ),
                needs_more_research=True,
            )
        )
        engine = ResearchEngine(
            source=source,
            synthesizer=HeuristicResearchSynthesizer(),
            planner=planner,
            provider_name="test",
            prompt_version="v1",
        )

        result = engine.investigate_job(job, refresh=True, now=datetime(2026, 3, 25, tzinfo=timezone.utc))

        self.assertEqual(source.search_calls, 1)
        self.assertEqual(source.follow_up_calls, 1)
        self.assertEqual(planner.plan_calls, 1)
        self.assertIsNotNone(result.investigation_plan)
        assert result.investigation_plan is not None
        self.assertEqual(result.investigation_plan.leading_hypothesis, "Diplomatic signaling deteriorated.")
        self.assertEqual(len(result.key_evidence), 2)

    def test_resolution_job_skips_planner_and_follow_up(self) -> None:
        job = _build_job(job_id="job-resolution", workflow_type="resolution_story")
        source = _PlannerAwareSource(
            initial_evidence=(_build_evidence(url="https://example.com/result", title="Resolution report"),),
            follow_up_evidence=(),
        )
        planner = _StaticPlanner(
            InvestigationPlan(
                job_id=job.job_id,
                candidate_explanations=(),
                leading_hypothesis="",
                follow_up_queries=(),
                needs_more_research=False,
            )
        )
        engine = ResearchEngine(
            source=source,
            synthesizer=HeuristicResearchSynthesizer(),
            planner=planner,
            provider_name="test",
            prompt_version="v1",
        )

        result = engine.investigate_job(job, refresh=True, now=datetime(2026, 3, 25, tzinfo=timezone.utc))

        self.assertEqual(source.search_calls, 1)
        self.assertEqual(source.follow_up_calls, 0)
        self.assertEqual(planner.plan_calls, 0)
        self.assertIsNone(result.investigation_plan)

    def test_follow_up_failure_degrades_gracefully_to_first_pass_synthesis(self) -> None:
        job = _build_job()
        source = _PlannerAwareSource(
            initial_evidence=(_build_evidence(url="https://example.com/initial", title="Initial report"),),
            follow_up_evidence=(),
            fail_on_follow_up=True,
        )
        planner = _StaticPlanner(
            InvestigationPlan(
                job_id=job.job_id,
                candidate_explanations=(
                    InvestigationLead(
                        label="Lead 1",
                        hypothesis="Rumor momentum faded.",
                        supporting_signals=("Initial report",),
                        missing_evidence=("Need direct disconfirmation.",),
                        priority="high",
                    ),
                ),
                leading_hypothesis="Rumor momentum faded.",
                follow_up_queries=(
                    FollowUpQuery(
                        query="Trump visit China follow-up",
                        source_type="web",
                        reason="Validate the rumor-fade hypothesis.",
                    ),
                ),
                needs_more_research=True,
            )
        )
        engine = ResearchEngine(
            source=source,
            synthesizer=HeuristicResearchSynthesizer(),
            planner=planner,
            provider_name="test",
            prompt_version="v1",
        )

        result = engine.investigate_job(job, refresh=True, now=datetime(2026, 3, 25, tzinfo=timezone.utc))

        self.assertEqual(result.status, "completed")
        self.assertIsNone(result.investigation_plan)
        self.assertEqual(len(result.key_evidence), 1)

    def test_select_follow_up_queries_preserves_skeptical_query_within_budget(self) -> None:
        plan = InvestigationPlan(
            job_id="job-1",
            candidate_explanations=(),
            leading_hypothesis="Example",
            follow_up_queries=(
                FollowUpQuery(query="q1", source_type="web", reason="first"),
                FollowUpQuery(query="q2", source_type="web", reason="second"),
                FollowUpQuery(query="q3", source_type="x", reason="third"),
            ),
            skeptical_query=FollowUpQuery(
                query="q-skeptical",
                source_type="x",
                reason="disconfirm",
                skeptical=True,
            ),
            needs_more_research=True,
        )

        selected = select_follow_up_queries(plan, max_queries=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[-1].query, "q-skeptical")
        self.assertTrue(selected[-1].skeptical)


class XaiSdkAdapterHelperTests(unittest.TestCase):
    def test_normalize_api_host_accepts_base_url(self) -> None:
        self.assertEqual(_normalize_api_host("https://api.x.ai/v1"), "api.x.ai")
        self.assertEqual(_normalize_api_host("https://api.x.ai:443/v1"), "api.x.ai:443")

    def test_choose_best_model_name_prefers_latest_reasoning_candidate(self) -> None:
        available = {"grok-4-1-fast-reasoning-latest", "grok-4.20", "grok-3-mini"}
        self.assertEqual(_choose_best_model_name(available, workflow_type="resolution_story"), "grok-4.20")

    def test_choose_best_model_name_falls_back_to_default_when_unknown(self) -> None:
        self.assertEqual(
            _choose_best_model_name({"grok-3-mini"}, workflow_type="resolution_story"),
            "grok-4.20-reasoning-latest",
        )

    def test_choose_best_model_name_uses_any_grok4_reasoning_before_multi_agent(self) -> None:
        available = {"grok-4-fast-reasoning", "grok-4.20-multi-agent-latest"}
        self.assertEqual(
            _choose_best_model_name(available, workflow_type="resolution_story"),
            "grok-4-fast-reasoning",
        )

    def test_choose_best_model_name_prefers_multi_agent_for_repricing(self) -> None:
        available = {"grok-4-fast-reasoning", "grok-4.20-multi-agent-latest"}
        self.assertEqual(
            _choose_best_model_name(available, workflow_type="repricing_story"),
            "grok-4.20-multi-agent-latest",
        )


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
        self.assertIsNotNone(cached.investigation_plan)
        self.assertEqual(cached.key_evidence[0].quality_tier, "primary")

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
        self.assertEqual(payload["story_drafts"][0]["job_id"], "job-1")
        self.assertNotIn("brief_markdown", payload["story_drafts"][0])
        self.assertIn("investigation_plan", payload["story_drafts"][0])
        self.assertEqual(payload["story_drafts"][0]["key_evidence"][0]["quality_tier"], "primary")


def _build_job(job_id: str = "job-1", *, workflow_type: str = "repricing_story") -> ResearchJob:
    return ResearchJob(
        job_id=job_id,
        family_key="trump-visit-china",
        family_label="Trump Visit China",
        workflow_type=workflow_type,
        story_type_hint="live_repricing" if workflow_type == "repricing_story" else "resolved_surprise",
        distance_from_extremes=0.22,
        entered_extreme_zone=False,
        editorial_priority_hint="high",
        story_role_hint="standalone",
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
            window_open_probability=0.62,
            window_close_probability=0.40,
            window_high_probability=0.68,
            window_low_probability=0.33,
            close_to_open_move=-0.22,
            max_abs_move_6h=0.14,
            max_abs_move_24h=0.28,
            largest_6h_move=-0.14,
            largest_24h_move=-0.28,
            weekly_range=0.35,
            persistence_of_largest_move=0.61,
            jump_count_over_threshold=2,
            max_move_timestamp=datetime(2026, 3, 24, 9, tzinfo=timezone.utc),
            category="politics",
            price_context=ResearchPriceContext(
                interval_hours=8,
                trace_points=(
                    PriceTracePoint(
                        observed_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                        probability=0.62,
                        move_since_previous=None,
                    ),
                    PriceTracePoint(
                        observed_at=datetime(2026, 3, 17, 8, tzinfo=timezone.utc),
                        probability=0.58,
                        move_since_previous=-0.04,
                    ),
                ),
                chart_interval_minutes=60,
                chart_trace_points=(
                    PriceTracePoint(
                        observed_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                        probability=0.62,
                        move_since_previous=None,
                    ),
                    PriceTracePoint(
                        observed_at=datetime(2026, 3, 17, 1, tzinfo=timezone.utc),
                        probability=0.60,
                        move_since_previous=-0.02,
                    ),
                ),
                largest_move_window_hours=24,
                largest_move_window_start=datetime(2026, 3, 23, 9, tzinfo=timezone.utc),
                largest_move_window_end=datetime(2026, 3, 24, 9, tzinfo=timezone.utc),
            ),
            url="https://polymarket.com/event/trump-visit-china",
            notes=("Detector note",),
        ),
        overlap_group_key=None,
        overlap_summary=None,
        suggested_merge_with=(),
        related_markets=(RelatedMarket(market_id="market-2", question="Will Trump visit China by March 31?"),),
    )


def _build_evidence(
    *,
    url: str,
    relevance_score: float = 0.8,
    temporal_score: float = 0.8,
    title: str = "Evidence item",
    stance: str = "supporting",
    source_type: str = "web_article",
    author: str = "Reporter",
    quality_tier: str = "secondary",
) -> EvidenceItem:
    return EvidenceItem(
        source_type=source_type,
        url=url,
        title_or_text=title,
        author_or_publication=author,
        published_at=datetime(2026, 3, 24, 8, tzinfo=timezone.utc),
        collected_at=datetime(2026, 3, 25, tzinfo=timezone.utc),
        relevance_score=relevance_score,
        temporal_proximity_score=temporal_score,
        stance=stance,
        excerpt="Short excerpt",
        query="Trump Visit China",
        quality_tier=quality_tier,
    )


def _build_result(job: ResearchJob, *, cache_key: str, completed_at: datetime) -> ResearchResult:
    return ResearchResult(
        job_id=job.job_id,
        cache_key=cache_key,
        provider="test",
        prompt_version="v1",
        model_name="heuristic",
        workflow_type=job.workflow_type,
        story_role_hint=job.story_role_hint,
        status="completed",
        explanation_class="plausible",
        confidence=0.61,
        most_plausible_explanation="Likely driven by diplomatic signaling and new chatter around a possible visit.",
        why_market_moved=job.why_flagged,
        price_action_summary="The market sold off sharply over the week after a concentrated 24h move.",
        surprise_assessment="This is a repricing story rather than a resolved outcome.",
        main_narrative="Traders appear to have downgraded the odds after the strongest signals failed to materialize.",
        belief_shift_drivers=("Diplomatic signaling weakened.", "Rumor momentum faded."),
        signal_types=("reporting", "social_commentary"),
        why_now="The move accelerated during the 24h window when the strongest follow-up reporting undercut the earlier narrative.",
        alternative_explanations=("The move may also reflect fading rumor momentum.",),
        unresolved_points=("It is still unclear whether there was a direct leak or only narrative decay.",),
        note_to_editor="Overlap with broader US-China diplomatic narratives.",
        draft_headline="Trump Visit China: Odds Slide as Narrative Weakens",
        draft_markdown="# Trump Visit China: Odds Slide as Narrative Weakens",
        overlap_group_key=job.overlap_group_key,
        overlap_summary=job.overlap_summary,
        suggested_merge_with=job.suggested_merge_with,
        key_evidence=(
            _build_evidence(
                url="https://www.reuters.com/example/1",
                quality_tier="primary",
            ),
        ),
        contradictory_evidence=(),
        open_questions=("Was there a direct leak or only rumor-driven speculation?",),
        completed_at=completed_at,
        investigation_plan=InvestigationPlan(
            job_id=job.job_id,
            candidate_explanations=(
                InvestigationLead(
                    label="Lead 1",
                    hypothesis="Diplomatic signaling weakened.",
                    supporting_signals=("Diplomatic signaling weakened.",),
                    missing_evidence=("Need stronger official confirmation.",),
                    priority="high",
                ),
            ),
            leading_hypothesis="Diplomatic signaling weakened.",
            follow_up_queries=(
                FollowUpQuery(
                    query="Trump visit China Reuters follow-up",
                    source_type="web",
                    reason="Validate the lead explanation.",
                ),
            ),
            skeptical_query=FollowUpQuery(
                query="Trump visit China denial",
                source_type="x",
                reason="Look for direct contradiction.",
                skeptical=True,
            ),
            assessments=(
                HypothesisAssessment(
                    hypothesis="Diplomatic signaling weakened.",
                    support_level="mixed",
                    contradictions=("No official confirmation yet.",),
                    open_uncertainty=("Timing remains uncertain.",),
                ),
            ),
            needs_more_research=True,
        ),
    )


class _MixedSource(ResearchSource):
    def __init__(self, *, success_job_id: str, evidence: tuple[EvidenceItem, ...]) -> None:
        self.success_job_id = success_job_id
        self.evidence = evidence

    def search(self, job: ResearchJob, query_plan: object) -> tuple[EvidenceItem, ...]:
        if job.job_id != self.success_job_id:
            raise RuntimeError("synthetic failure")
        return self.evidence

    def search_follow_up(
        self,
        job: ResearchJob,
        query_plan: object,
        follow_up_queries: tuple[FollowUpQuery, ...],
    ) -> tuple[EvidenceItem, ...]:
        return ()


class _PlannerAwareSource(ResearchSource):
    def __init__(
        self,
        *,
        initial_evidence: tuple[EvidenceItem, ...],
        follow_up_evidence: tuple[EvidenceItem, ...],
        fail_on_follow_up: bool = False,
    ) -> None:
        self.initial_evidence = initial_evidence
        self.follow_up_evidence = follow_up_evidence
        self.fail_on_follow_up = fail_on_follow_up
        self.search_calls = 0
        self.follow_up_calls = 0

    def search(self, job: ResearchJob, query_plan: object) -> tuple[EvidenceItem, ...]:
        self.search_calls += 1
        return self.initial_evidence

    def search_follow_up(
        self,
        job: ResearchJob,
        query_plan: object,
        follow_up_queries: tuple[FollowUpQuery, ...],
    ) -> tuple[EvidenceItem, ...]:
        self.follow_up_calls += 1
        if self.fail_on_follow_up:
            raise RuntimeError("synthetic follow-up failure")
        return self.follow_up_evidence


class _StaticPlanner(ResearchPlanner):
    def __init__(self, plan: InvestigationPlan) -> None:
        self.plan_result = plan
        self.plan_calls = 0

    def plan(
        self,
        job: ResearchJob,
        query_plan: object,
        evidence: tuple[EvidenceItem, ...],
        *,
        generated_at: datetime,
    ) -> InvestigationPlan:
        self.plan_calls += 1
        return self.plan_result
