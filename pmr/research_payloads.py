from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pmr.config import MonitoringConfig
from pmr.models import (
    EvidenceItem,
    RelatedMarket,
    RepricingEvent,
    ResearchBatchResult,
    ResearchJob,
    ResearchMarketContext,
    ResearchResult,
)


def build_research_input_payload(
    events: Sequence[RepricingEvent],
    config: MonitoringConfig,
    *,
    source_name: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable anomaly payloads for a downstream research agent."""

    stamp = generated_at or datetime.now(timezone.utc)
    return {
        "generated_at": stamp.isoformat(),
        "source": {
            "provider": source_name,
            "detection_window_days": config.detection_window_days,
            "preferred_baseline_window_days": config.preferred_baseline_window_days,
            "feature_interval_minutes": config.feature_interval_minutes,
            "min_abs_move_24h": config.min_abs_move_24h,
            "min_weekly_range": config.min_weekly_range,
        },
        "research_jobs": [serialize_research_job(event) for event in events],
        "anomalies": [serialize_event_for_research(event) for event in events],
    }


def serialize_event_for_research(event: RepricingEvent) -> dict[str, Any]:
    """Serialize one detected anomaly into the shape a research agent needs."""

    return {
        "market": {
            "market_id": event.market.market_id,
            "question": event.market.question,
            "slug": event.market.slug,
            "url": event.market.url,
            "description": event.market.description,
            "category": event.market.category,
            "tags": list(event.market.tags),
            "condition_id": event.market.condition_id,
            "tracked_outcome": event.market.tracked_outcome,
            "tracked_token_id": event.market.tracked_token_id,
            "event_title": event.market.event_title,
        },
        "story": {
            "family_key": event.story_group_key,
            "family_label": event.story_group_label,
            "story_type_hint": event.story_type_hint,
            "distance_from_extremes": event.distance_from_extremes,
            "entered_extreme_zone": event.entered_extreme_zone,
            "related_market_ids": list(event.related_market_ids),
            "related_market_questions": list(event.related_market_questions),
        },
        "detector": {
            "detection_window_start": event.detection_window_start.isoformat(),
            "detection_window_end": event.detection_window_end.isoformat(),
            "history_mode": event.history_mode,
            "confidence_level": event.confidence_level,
            "confidence_score": event.confidence_score,
            "composite_score": event.composite_score,
            "eligible_for_ranking": event.eligible_for_ranking,
            "exclusion_reason": event.exclusion_reason,
            "max_move_timestamp": event.max_move_timestamp.isoformat() if event.max_move_timestamp else None,
        },
        "features": {
            "window_open_probability": event.window_open_probability,
            "window_close_probability": event.window_close_probability,
            "window_high_probability": event.window_high_probability,
            "window_low_probability": event.window_low_probability,
            "close_to_open_move": event.close_to_open_move,
            "max_abs_move_6h": event.max_abs_move_6h,
            "max_abs_move_24h": event.max_abs_move_24h,
            "largest_6h_move": event.largest_6h_move,
            "largest_24h_move": event.largest_24h_move,
            "weekly_range": event.weekly_range,
            "persistence_of_largest_move": event.persistence_of_largest_move,
            "jump_count_over_threshold": event.jump_count_over_threshold,
        },
        "baseline_stats": {
            "usable_history_days": event.baseline_stats.usable_history_days,
            "snapshot_count": event.baseline_stats.snapshot_count,
            "median_abs_move_6h": event.baseline_stats.median_abs_move_6h,
            "median_abs_move_24h": event.baseline_stats.median_abs_move_24h,
            "median_weekly_range": event.baseline_stats.median_weekly_range,
            "sample_count_6h": event.baseline_stats.sample_count_6h,
            "sample_count_24h": event.baseline_stats.sample_count_24h,
            "sample_count_weekly_range": event.baseline_stats.sample_count_weekly_range,
        },
        "normalized_scores": {
            "z_6h": event.z_6h,
            "z_24h": event.z_24h,
            "z_range": event.z_range,
        },
        "liquidity": {
            "volume_7d_usd": event.liquidity_metrics.volume_7d_usd,
            "volume_24h_usd": event.liquidity_metrics.volume_24h_usd,
            "open_interest_usd": event.liquidity_metrics.open_interest_usd,
            "liquidity_score": event.liquidity_metrics.liquidity_score,
        },
        "notes": list(event.notes),
    }


def serialize_research_job(event: RepricingEvent) -> dict[str, Any]:
    """Serialize one anomaly into a story-oriented research brief."""

    return {
        "job_id": event.story_group_key,
        "story": {
            "family_key": event.story_group_key,
            "family_label": event.story_group_label,
            "story_type_hint": event.story_type_hint,
            "distance_from_extremes": event.distance_from_extremes,
            "entered_extreme_zone": event.entered_extreme_zone,
            "editorial_priority_hint": _editorial_priority_hint(event),
        },
        "investigation": {
            "question": _build_investigation_question(event),
            "why_flagged": _build_why_flagged(event),
            "focus_points": _build_focus_points(event),
        },
        "primary_market": serialize_event_for_research(event),
        "related_markets": [
            {"market_id": market_id, "question": question}
            for market_id, question in zip(event.related_market_ids, event.related_market_questions)
        ],
    }


def _build_investigation_question(event: RepricingEvent) -> str:
    return (
        f"What most plausibly explains the repricing in '{event.market.question}' "
        f"between {event.detection_window_start.isoformat()} and {event.detection_window_end.isoformat()}?"
    )


def _build_why_flagged(event: RepricingEvent) -> str:
    direction = "up" if event.close_to_open_move >= 0 else "down"
    return (
        f"Flagged as a {event.story_type_hint} candidate because the market moved {direction} by "
        f"{abs(event.close_to_open_move) * 100:.1f} pts from window open to close, reached a "
        f"{event.weekly_range * 100:.1f} pt weekly range, and posted a max 24h move of "
        f"{event.max_abs_move_24h * 100:.1f} pts."
    )


def _build_focus_points(event: RepricingEvent) -> list[str]:
    points = [
        "Prioritize evidence near the largest move timestamp and compare clear news against rumor-driven repricing.",
        "Decide whether the move reflects a genuine surprise, a plausible but uncertain explanation, or mostly speculative chatter.",
    ]
    if event.story_type_hint == "live_repricing":
        points.append(
            "Look for incremental developments, negotiations, leaks, or commentary that shifted odds without fully resolving the market."
        )
    elif event.story_type_hint == "resolved_surprise":
        points.append(
            "Confirm the concluding outcome and explain why the market appears to have been caught off guard before resolution."
        )
    else:
        points.append(
            "Check whether this looks like a late-stage confirmation of an already-likely outcome rather than fresh surprise."
        )
    if event.related_market_ids:
        points.append(
            "Use the related market variants as supporting context, but avoid duplicating the same story in the final writeup."
        )
    return points


def _editorial_priority_hint(event: RepricingEvent) -> str:
    if event.story_type_hint == "live_repricing":
        return "high"
    if event.story_type_hint == "resolved_surprise":
        return "medium"
    return "low"


def load_research_jobs_from_file(path: Path) -> tuple[ResearchJob, ...]:
    """Load typed research jobs from a previously exported JSON payload."""

    payload = json.loads(path.read_text())
    return load_research_jobs_from_payload(payload)


def load_research_jobs_from_payload(payload: dict[str, Any]) -> tuple[ResearchJob, ...]:
    """Parse typed research jobs from the current JSON handoff contract."""

    return tuple(_parse_research_job(item) for item in payload.get("research_jobs", ()))


def build_research_results_payload(batch: ResearchBatchResult) -> dict[str, Any]:
    """Serialize a batch of research results for storage or downstream tooling."""

    return {
        "generated_at": batch.generated_at.isoformat(),
        "provider": batch.provider,
        "prompt_version": batch.prompt_version,
        "processed_jobs": batch.processed_jobs,
        "cached_jobs": batch.cached_jobs,
        "failed_jobs": batch.failed_jobs,
        "results": [serialize_research_result(result) for result in batch.results],
    }


def serialize_research_result(result: ResearchResult) -> dict[str, Any]:
    """Serialize one structured research result."""

    return {
        "job_id": result.job_id,
        "cache_key": result.cache_key,
        "provider": result.provider,
        "prompt_version": result.prompt_version,
        "status": result.status,
        "explanation_class": result.explanation_class,
        "confidence": result.confidence,
        "most_plausible_explanation": result.most_plausible_explanation,
        "why_market_moved": result.why_market_moved,
        "key_evidence": [serialize_evidence_item(item) for item in result.key_evidence],
        "contradictory_evidence": [serialize_evidence_item(item) for item in result.contradictory_evidence],
        "open_questions": list(result.open_questions),
        "completed_at": result.completed_at.isoformat(),
        "error_message": result.error_message,
        "used_cache": result.used_cache,
    }


def serialize_evidence_item(item: EvidenceItem) -> dict[str, Any]:
    """Serialize a normalized evidence record."""

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
        "excerpt": item.excerpt,
        "query": item.query,
    }


def _parse_research_job(payload: dict[str, Any]) -> ResearchJob:
    story = payload["story"]
    investigation = payload["investigation"]
    primary_market = _parse_primary_market(payload["primary_market"])
    related_markets = tuple(
        RelatedMarket(
            market_id=str(item["market_id"]),
            question=str(item["question"]),
        )
        for item in payload.get("related_markets", ())
    )
    return ResearchJob(
        job_id=str(payload["job_id"]),
        family_key=str(story["family_key"]),
        family_label=str(story["family_label"]),
        story_type_hint=story["story_type_hint"],
        distance_from_extremes=float(story["distance_from_extremes"]),
        entered_extreme_zone=bool(story["entered_extreme_zone"]),
        editorial_priority_hint=story["editorial_priority_hint"],
        investigation_question=str(investigation["question"]),
        why_flagged=str(investigation["why_flagged"]),
        focus_points=tuple(str(point) for point in investigation.get("focus_points", ())),
        primary_market=primary_market,
        related_markets=related_markets,
    )


def _parse_primary_market(payload: dict[str, Any]) -> ResearchMarketContext:
    market = payload["market"]
    detector = payload["detector"]
    features = payload["features"]
    return ResearchMarketContext(
        market_id=str(market["market_id"]),
        question=str(market["question"]),
        detection_window_start=datetime.fromisoformat(detector["detection_window_start"]),
        detection_window_end=datetime.fromisoformat(detector["detection_window_end"]),
        history_mode=detector["history_mode"],
        confidence_level=detector["confidence_level"],
        confidence_score=float(detector["confidence_score"]),
        composite_score=float(detector["composite_score"]),
        close_to_open_move=float(features["close_to_open_move"]),
        max_abs_move_24h=float(features["max_abs_move_24h"]),
        weekly_range=float(features["weekly_range"]),
        max_move_timestamp=(
            datetime.fromisoformat(detector["max_move_timestamp"])
            if detector.get("max_move_timestamp")
            else None
        ),
        category=str(market["category"]),
        slug=market.get("slug"),
        url=market.get("url"),
        description=market.get("description"),
        tags=tuple(str(tag) for tag in market.get("tags", ())),
        condition_id=market.get("condition_id"),
        tracked_outcome=market.get("tracked_outcome"),
        tracked_token_id=market.get("tracked_token_id"),
        event_title=market.get("event_title"),
        notes=tuple(str(note) for note in payload.get("notes", ())),
    )
