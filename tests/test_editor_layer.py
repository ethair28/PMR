from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pmr.editor_engine import EditorEngine, HeuristicEditorComposer
from pmr.editor_payloads import build_weekly_report_payload, load_editor_story_packets_from_payloads
from pmr.editor_reporting import render_editor_decisions_markdown, render_weekly_report_markdown
from pmr.models import (
    EditorStoryPacket,
    EvidenceItem,
    PriceTracePoint,
    RelatedMarket,
    ResearchMarketContext,
    ResearchPriceContext,
)


class EditorPayloadTests(unittest.TestCase):
    def test_load_editor_story_packets_merges_results_with_rich_context(self) -> None:
        packets = load_editor_story_packets_from_payloads(
            research_results_payload=_sample_research_results_payload(),
            research_inputs_payload=_sample_research_inputs_payload(),
        )

        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet.job_id, "job-1")
        self.assertEqual(packet.family_label, "Trump Visit China")
        self.assertEqual(packet.primary_market.question, "Will Trump visit China by April 30?")
        self.assertEqual(packet.primary_market.price_context.interval_hours, 8)
        self.assertEqual(packet.root_cluster_key, "politics:china")
        self.assertEqual(packet.draft_headline, "Trump-Xi thaw")
        self.assertEqual(packet.key_evidence[0].url, "https://example.com/story")


class HeuristicEditorTests(unittest.TestCase):
    def test_heuristic_editor_merges_secondary_overlap_story(self) -> None:
        primary = _build_packet(
            job_id="slovenia-election",
            family_label="Slovenia Election Winner",
            story_role_hint="primary",
            overlap_group_key="politics:slovenia:government",
            suggested_merge_with=("slovenia-pm",),
            workflow_type="resolution_story",
            confidence=0.82,
        )
        secondary = _build_packet(
            job_id="slovenia-pm",
            family_label="Next Prime Minister of Slovenia",
            story_role_hint="secondary",
            overlap_group_key="politics:slovenia:government",
            suggested_merge_with=("slovenia-election",),
            workflow_type="repricing_story",
            confidence=0.68,
        )
        engine = EditorEngine(
            composer=HeuristicEditorComposer(),
            provider_name="heuristic_editor",
            prompt_version="v1",
        )

        report = engine.run((primary, secondary), now=datetime(2026, 3, 28, tzinfo=timezone.utc))

        self.assertEqual(report.included_story_count, 1)
        self.assertEqual(report.merged_story_count, 1)
        self.assertEqual(report.excluded_story_count, 0)
        self.assertEqual(len(report.sections), 1)
        self.assertEqual(report.sections[0].included_job_ids, ("slovenia-election", "slovenia-pm"))
        include_decision = next(decision for decision in report.decisions if decision.action == "include")
        self.assertEqual(include_decision.section_rank, 1)

    def test_heuristic_editor_can_return_no_story_week(self) -> None:
        failed = _build_packet(
            job_id="weak-story",
            family_label="Weak Story",
            status="failed",
            confidence=0.0,
        )
        engine = EditorEngine(
            composer=HeuristicEditorComposer(),
            provider_name="heuristic_editor",
            prompt_version="v1",
        )

        report = engine.run((failed,), now=datetime(2026, 3, 28, tzinfo=timezone.utc))

        self.assertEqual(len(report.sections), 0)
        self.assertEqual(report.included_story_count, 0)
        self.assertEqual(report.excluded_story_count, 1)

    def test_heuristic_editor_compresses_dense_root_cluster(self) -> None:
        stories = (
            _build_packet(
                job_id="iran-ceasefire",
                family_label="US x Iran ceasefire",
                overlap_group_key="geo:iran:ceasefire",
                root_cluster_key="geopolitics:iran",
                confidence=0.82,
            ),
            _build_packet(
                job_id="iran-meeting",
                family_label="US x Iran meeting",
                overlap_group_key="geo:iran:meeting",
                root_cluster_key="geopolitics:iran",
                confidence=0.78,
            ),
            _build_packet(
                job_id="iran-military",
                family_label="Military action against Iran ends",
                overlap_group_key="geo:iran:military",
                root_cluster_key="geopolitics:iran",
                confidence=0.74,
            ),
        )
        report = EditorEngine(
            composer=HeuristicEditorComposer(),
            provider_name="heuristic_editor",
            prompt_version="v1",
        ).run(stories, now=datetime(2026, 3, 28, tzinfo=timezone.utc))

        self.assertEqual(len(report.sections), 2)
        self.assertEqual(report.included_story_count, 2)
        self.assertEqual(report.merged_story_count, 1)
        merged = [decision for decision in report.decisions if decision.action == "merge"]
        self.assertTrue(merged)
        self.assertIsNotNone(merged[0].section_headline)
        self.assertIsNotNone(merged[0].section_rank)


class EditorReportingTests(unittest.TestCase):
    def test_report_and_decision_renderers_use_structured_output(self) -> None:
        packet = _build_packet(job_id="job-1", family_label="Trump Visit China", confidence=0.76)
        report = EditorEngine(
            composer=HeuristicEditorComposer(),
            provider_name="heuristic_editor",
            prompt_version="v1",
        ).run((packet,), now=datetime(2026, 3, 28, tzinfo=timezone.utc))

        payload = build_weekly_report_payload(report)
        markdown = render_weekly_report_markdown(report)
        decisions = render_editor_decisions_markdown(report)

        self.assertEqual(payload["included_story_count"], 1)
        self.assertIn("# PMR Weekly Report", markdown)
        self.assertIn("Trump Visit China", decisions)


def _sample_research_inputs_payload() -> dict:
    return {
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
                                "move_since_previous": None,
                            },
                            {
                                "observed_at": "2026-03-17T08:00:00+00:00",
                                "probability": 0.58,
                                "move_since_previous": -0.04,
                            },
                        ],
                    },
                    "notes": ["note 1"],
                },
                "related_markets": [{"market_id": "market-2", "question": "Will Trump visit China by March 31?"}],
            }
        ]
    }


def _sample_research_results_payload() -> dict:
    return {
        "generated_at": "2026-03-28T00:00:00+00:00",
        "provider": "xai",
        "prompt_version": "v1",
        "processed_jobs": 1,
        "cached_jobs": 0,
        "failed_jobs": 0,
        "story_drafts": [
            {
                "job_id": "job-1",
                "cache_key": "cache-1",
                "provider": "xai",
                "prompt_version": "v1",
                "model_name": "grok",
                "workflow_type": "repricing_story",
                "story_role_hint": "standalone",
                "status": "completed",
                "explanation_class": "plausible",
                "confidence": 0.71,
                "most_plausible_explanation": "Thaw in rhetoric",
                "why_market_moved": "Improved rhetoric and signaling.",
                "price_action_summary": "The market moved down sharply over 24h.",
                "surprise_assessment": "Not a clean resolution surprise.",
                "main_narrative": "The market repriced as traders reacted to softer diplomatic signals.",
                "alternative_explanations": ["Noise in deadline markets."],
                "note_to_editor": "Keep if the weekly set is light.",
                "draft_headline": "Trump-Xi thaw",
                "draft_markdown": "A real repricing story.",
                "overlap_group_key": None,
                "overlap_summary": None,
                "suggested_merge_with": [],
                "key_evidence": [
                    {
                        "source_type": "news_article",
                        "url": "https://example.com/story",
                        "title_or_text": "Diplomatic thaw",
                        "author_or_publication": "Example News",
                        "published_at": "2026-03-24T10:00:00+00:00",
                        "collected_at": "2026-03-28T00:00:00+00:00",
                        "relevance_score": 0.9,
                        "temporal_proximity_score": 0.8,
                        "stance": "supporting",
                        "excerpt": "Signals improved",
                        "query": "Trump China",
                    }
                ],
                "contradictory_evidence": [],
                "open_questions": ["How durable is the thaw?"],
                "completed_at": "2026-03-28T00:00:00+00:00",
                "error_message": None,
                "used_cache": False,
            }
        ],
    }


def _build_packet(
    *,
    job_id: str = "job-1",
    family_label: str = "Story",
    story_role_hint: str = "standalone",
    overlap_group_key: str | None = None,
    suggested_merge_with: tuple[str, ...] = (),
    workflow_type: str = "repricing_story",
    confidence: float = 0.7,
    status: str = "completed",
    root_cluster_key: str | None = None,
) -> EditorStoryPacket:
    market_context = ResearchMarketContext(
        market_id=f"{job_id}-market",
        question=family_label,
        detection_window_start=datetime(2026, 3, 17, tzinfo=timezone.utc),
        detection_window_end=datetime(2026, 3, 24, tzinfo=timezone.utc),
        history_mode="full_history",
        confidence_level="high",
        confidence_score=0.8,
        composite_score=8.0,
        window_open_probability=0.35,
        window_close_probability=0.61,
        window_high_probability=0.66,
        window_low_probability=0.31,
        close_to_open_move=0.26,
        max_abs_move_6h=0.12,
        max_abs_move_24h=0.26,
        largest_6h_move=0.12,
        largest_24h_move=0.26,
        weekly_range=0.35,
        persistence_of_largest_move=0.7,
        jump_count_over_threshold=2,
        max_move_timestamp=datetime(2026, 3, 23, 12, tzinfo=timezone.utc),
        category="politics",
        price_context=ResearchPriceContext(
            interval_hours=8,
            trace_points=(
                PriceTracePoint(
                    observed_at=datetime(2026, 3, 17, tzinfo=timezone.utc),
                    probability=0.35,
                    move_since_previous=None,
                ),
                PriceTracePoint(
                    observed_at=datetime(2026, 3, 17, 8, tzinfo=timezone.utc),
                    probability=0.40,
                    move_since_previous=0.05,
                ),
            ),
            largest_move_window_hours=24,
            largest_move_window_start=datetime(2026, 3, 23, 0, tzinfo=timezone.utc),
            largest_move_window_end=datetime(2026, 3, 24, 0, tzinfo=timezone.utc),
            surprise_reference_probability=None,
            surprise_points=None,
            surprise_label=None,
        ),
        event_title=family_label,
    )
    return EditorStoryPacket(
        job_id=job_id,
        family_key=job_id,
        family_label=family_label,
        workflow_type=workflow_type,  # type: ignore[arg-type]
        story_type_hint="live_repricing" if workflow_type == "repricing_story" else "resolved_surprise",
        editorial_priority_hint="high" if workflow_type == "repricing_story" else "medium",
        story_role_hint=story_role_hint,  # type: ignore[arg-type]
        investigation_question="Why did the market move?",
        why_flagged="Large weekly repricing.",
        focus_points=("Use price action",),
        primary_market=market_context,
        overlap_group_key=overlap_group_key,
        overlap_summary="Overlap cluster" if overlap_group_key else None,
        root_cluster_key=root_cluster_key,
        root_cluster_summary="Root cluster" if root_cluster_key else None,
        suggested_merge_with=suggested_merge_with,
        related_markets=(RelatedMarket(market_id="related-1", question="Related market"),),
        status=status,  # type: ignore[arg-type]
        explanation_class="plausible" if status == "completed" else None,
        confidence=confidence,
        model_name="grok",
        most_plausible_explanation="Most plausible explanation.",
        why_market_moved="Why the market moved.",
        price_action_summary="Price moved sharply over the week.",
        surprise_assessment="Moderate surprise." if workflow_type != "repricing_story" else "",
        main_narrative="This is the main narrative for the story.",
        alternative_explanations=("Alternative angle.",),
        note_to_editor="Editor note.",
        draft_headline=family_label,
        draft_markdown="Draft body.",
        key_evidence=(
            EvidenceItem(
                source_type="news_article",
                url="https://example.com/story",
                title_or_text="Example story",
                author_or_publication="Example News",
                published_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
                collected_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
                relevance_score=0.9,
                temporal_proximity_score=0.8,
                stance="supporting",
                excerpt="Example excerpt",
            ),
        ),
        contradictory_evidence=(),
        open_questions=("Open question.",),
        completed_at=datetime(2026, 3, 28, tzinfo=timezone.utc),
        error_message=None if status != "failed" else "failed",
        used_cache=False,
    )
