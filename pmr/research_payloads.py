from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any, Sequence

from pmr.config import MonitoringConfig
from pmr.models import (
    EvidenceItem,
    PriceTracePoint,
    RelatedMarket,
    RepricingEvent,
    ResearchBatchResult,
    ResearchJob,
    ResearchMarketContext,
    ResearchPriceContext,
    ResearchResult,
    StoryRoleHint,
    StoryWorkflowType,
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
    overlap_assignments = _build_overlap_assignments(events)
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
        "research_jobs": [
            serialize_research_job(event, overlap_assignments=overlap_assignments) for event in events
        ],
        "anomalies": [
            serialize_event_for_research(event, overlap_assignments=overlap_assignments) for event in events
        ],
    }


def serialize_event_for_research(
    event: RepricingEvent,
    *,
    overlap_assignments: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize one detected anomaly into the shape a research agent needs."""

    workflow_type = _workflow_type_from_story_hint(event.story_type_hint)
    price_context = _build_price_context(event)
    overlap = (overlap_assignments or {}).get(event.story_group_key, _default_overlap_assignment())
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
            "workflow_type": workflow_type,
            "story_type_hint": event.story_type_hint,
            "distance_from_extremes": event.distance_from_extremes,
            "entered_extreme_zone": event.entered_extreme_zone,
            "story_role_hint": overlap["story_role_hint"],
            "overlap_group_key": overlap["overlap_group_key"],
            "overlap_summary": overlap["overlap_summary"],
            "suggested_merge_with": list(overlap["suggested_merge_with"]),
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
        "price_context": serialize_price_context(price_context),
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


def serialize_research_job(
    event: RepricingEvent,
    *,
    overlap_assignments: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize one anomaly into a story-oriented research brief."""

    workflow_type = _workflow_type_from_story_hint(event.story_type_hint)
    overlap = (overlap_assignments or {}).get(event.story_group_key, _default_overlap_assignment())
    return {
        "job_id": event.story_group_key,
        "story": {
            "family_key": event.story_group_key,
            "family_label": event.story_group_label,
            "workflow_type": workflow_type,
            "story_type_hint": event.story_type_hint,
            "distance_from_extremes": event.distance_from_extremes,
            "entered_extreme_zone": event.entered_extreme_zone,
            "editorial_priority_hint": _editorial_priority_hint(event),
            "story_role_hint": overlap["story_role_hint"],
            "overlap_group_key": overlap["overlap_group_key"],
            "overlap_summary": overlap["overlap_summary"],
            "suggested_merge_with": list(overlap["suggested_merge_with"]),
        },
        "investigation": {
            "question": _build_investigation_question(event),
            "why_flagged": _build_why_flagged(event),
            "focus_points": _build_focus_points(event, overlap=overlap),
        },
        "primary_market": serialize_event_for_research(event, overlap_assignments=overlap_assignments),
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


def _build_focus_points(
    event: RepricingEvent,
    *,
    overlap: dict[str, Any] | None = None,
) -> list[str]:
    points = [
        "Prioritize evidence near the largest move timestamp and compare clear news against rumor-driven repricing.",
        "Decide whether the move reflects a genuine surprise, a plausible but uncertain explanation, or mostly speculative chatter.",
    ]
    if event.story_type_hint == "live_repricing":
        points.append(
            "Explain what changed market perception, not just what happened: separate hard news, rumor, negotiation signals, social commentary, and sentiment when multiple forces are in play."
        )
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
    if overlap and overlap.get("story_role_hint") != "standalone":
        points.append(
            "This overlaps with another weekly candidate. Keep the angle distinct and leave a clear note to the editor if the final report should merge the stories."
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
    """Serialize a batch of story-development outputs for storage or downstream tooling."""

    return {
        "generated_at": batch.generated_at.isoformat(),
        "provider": batch.provider,
        "prompt_version": batch.prompt_version,
        "processed_jobs": batch.processed_jobs,
        "cached_jobs": batch.cached_jobs,
        "failed_jobs": batch.failed_jobs,
        "story_drafts": [serialize_research_result(result) for result in batch.results],
    }


def serialize_research_result(result: ResearchResult) -> dict[str, Any]:
    """Serialize one structured story-development output."""

    return {
        "job_id": result.job_id,
        "cache_key": result.cache_key,
        "provider": result.provider,
        "prompt_version": result.prompt_version,
        "model_name": result.model_name,
        "workflow_type": result.workflow_type,
        "story_role_hint": result.story_role_hint,
        "status": result.status,
        "explanation_class": result.explanation_class,
        "confidence": result.confidence,
        "most_plausible_explanation": result.most_plausible_explanation,
        "why_market_moved": result.why_market_moved,
        "price_action_summary": result.price_action_summary,
        "surprise_assessment": result.surprise_assessment,
        "main_narrative": result.main_narrative,
        "alternative_explanations": list(result.alternative_explanations),
        "note_to_editor": result.note_to_editor,
        "draft_headline": result.draft_headline,
        "draft_markdown": result.draft_markdown,
        "overlap_group_key": result.overlap_group_key,
        "overlap_summary": result.overlap_summary,
        "suggested_merge_with": list(result.suggested_merge_with),
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


def serialize_price_context(context: ResearchPriceContext) -> dict[str, Any]:
    """Serialize the compact weekly price context passed into story development."""

    return {
        "interval_hours": context.interval_hours,
        "largest_move_window_hours": context.largest_move_window_hours,
        "largest_move_window_start": (
            context.largest_move_window_start.isoformat() if context.largest_move_window_start else None
        ),
        "largest_move_window_end": (
            context.largest_move_window_end.isoformat() if context.largest_move_window_end else None
        ),
        "surprise_reference_probability": context.surprise_reference_probability,
        "surprise_points": context.surprise_points,
        "surprise_label": context.surprise_label,
        "trace_points": [serialize_price_trace_point(item) for item in context.trace_points],
    }


def serialize_price_trace_point(item: PriceTracePoint) -> dict[str, Any]:
    return {
        "observed_at": item.observed_at.isoformat(),
        "probability": item.probability,
        "move_since_previous": item.move_since_previous,
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
        workflow_type=story.get("workflow_type", _workflow_type_from_story_hint(story["story_type_hint"])),
        story_type_hint=story["story_type_hint"],
        distance_from_extremes=float(story["distance_from_extremes"]),
        entered_extreme_zone=bool(story["entered_extreme_zone"]),
        editorial_priority_hint=story["editorial_priority_hint"],
        story_role_hint=story.get("story_role_hint", "standalone"),
        investigation_question=str(investigation["question"]),
        why_flagged=str(investigation["why_flagged"]),
        focus_points=tuple(str(point) for point in investigation.get("focus_points", ())),
        primary_market=primary_market,
        overlap_group_key=story.get("overlap_group_key"),
        overlap_summary=story.get("overlap_summary"),
        suggested_merge_with=tuple(str(item) for item in story.get("suggested_merge_with", ())),
        related_markets=related_markets,
    )


def _parse_primary_market(payload: dict[str, Any]) -> ResearchMarketContext:
    market = payload["market"]
    detector = payload["detector"]
    features = payload["features"]
    price_context = _parse_price_context(payload.get("price_context", {}), payload)
    return ResearchMarketContext(
        market_id=str(market["market_id"]),
        question=str(market["question"]),
        detection_window_start=datetime.fromisoformat(detector["detection_window_start"]),
        detection_window_end=datetime.fromisoformat(detector["detection_window_end"]),
        history_mode=detector["history_mode"],
        confidence_level=detector["confidence_level"],
        confidence_score=float(detector["confidence_score"]),
        composite_score=float(detector["composite_score"]),
        window_open_probability=float(features.get("window_open_probability", 0.0)),
        window_close_probability=float(features.get("window_close_probability", 0.0)),
        window_high_probability=float(features.get("window_high_probability", 0.0)),
        window_low_probability=float(features.get("window_low_probability", 0.0)),
        close_to_open_move=float(features["close_to_open_move"]),
        max_abs_move_6h=float(features.get("max_abs_move_6h", 0.0)),
        max_abs_move_24h=float(features["max_abs_move_24h"]),
        largest_6h_move=float(features.get("largest_6h_move", 0.0)),
        largest_24h_move=float(features.get("largest_24h_move", 0.0)),
        weekly_range=float(features["weekly_range"]),
        persistence_of_largest_move=float(features.get("persistence_of_largest_move", 0.0)),
        jump_count_over_threshold=int(features.get("jump_count_over_threshold", 0)),
        max_move_timestamp=(
            datetime.fromisoformat(detector["max_move_timestamp"])
            if detector.get("max_move_timestamp")
            else None
        ),
        category=str(market["category"]),
        price_context=price_context,
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


def _parse_price_context(raw: dict[str, Any], market_payload: dict[str, Any]) -> ResearchPriceContext:
    trace_points = tuple(
        PriceTracePoint(
            observed_at=datetime.fromisoformat(item["observed_at"]),
            probability=float(item["probability"]),
            move_since_previous=(
                float(item["move_since_previous"]) if item.get("move_since_previous") is not None else None
            ),
        )
        for item in raw.get("trace_points", ())
    )
    if not raw:
        detector = market_payload["detector"]
        return ResearchPriceContext(
            interval_hours=8,
            trace_points=trace_points,
            largest_move_window_hours=24,
            largest_move_window_start=(
                datetime.fromisoformat(detector["max_move_timestamp"]) if detector.get("max_move_timestamp") else None
            ),
            largest_move_window_end=(
                datetime.fromisoformat(detector["max_move_timestamp"]) if detector.get("max_move_timestamp") else None
            ),
        )
    return ResearchPriceContext(
        interval_hours=int(raw.get("interval_hours", 8)),
        trace_points=trace_points,
        largest_move_window_hours=int(raw.get("largest_move_window_hours", 24)),
        largest_move_window_start=(
            datetime.fromisoformat(raw["largest_move_window_start"])
            if raw.get("largest_move_window_start")
            else None
        ),
        largest_move_window_end=(
            datetime.fromisoformat(raw["largest_move_window_end"])
            if raw.get("largest_move_window_end")
            else None
        ),
        surprise_reference_probability=(
            float(raw["surprise_reference_probability"])
            if raw.get("surprise_reference_probability") is not None
            else None
        ),
        surprise_points=(
            float(raw["surprise_points"]) if raw.get("surprise_points") is not None else None
        ),
        surprise_label=raw.get("surprise_label"),
    )


def _workflow_type_from_story_hint(story_type_hint: str) -> StoryWorkflowType:
    if story_type_hint == "live_repricing":
        return "repricing_story"
    return "resolution_story"


def _default_overlap_assignment() -> dict[str, Any]:
    return {
        "overlap_group_key": None,
        "overlap_summary": None,
        "suggested_merge_with": (),
        "story_role_hint": "standalone",
    }


def _build_overlap_assignments(events: Sequence[RepricingEvent]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[RepricingEvent]] = {}
    for event in events:
        signature = _overlap_signature(event)
        if signature is None:
            continue
        grouped.setdefault(signature, []).append(event)

    assignments: dict[str, dict[str, Any]] = {}
    for signature, grouped_events in grouped.items():
        if len(grouped_events) < 2:
            continue
        ordered = sorted(grouped_events, key=lambda item: item.composite_score, reverse=True)
        primary = ordered[0]
        _, geography, theme = signature
        for index, event in enumerate(ordered):
            role_hint: StoryRoleHint = "primary" if index == 0 else "secondary"
            assignments[event.story_group_key] = {
                "overlap_group_key": f"{signature[0]}:{geography}:{theme}",
                "overlap_summary": _build_overlap_summary(
                    event=event,
                    group_size=len(ordered),
                    geography=geography,
                    theme=theme,
                    role_hint=role_hint,
                    primary=primary,
                ),
                "suggested_merge_with": tuple(
                    other.story_group_key for other in ordered if other.story_group_key != event.story_group_key
                ),
                "story_role_hint": role_hint,
            }
    return assignments


def _build_overlap_summary(
    *,
    event: RepricingEvent,
    group_size: int,
    geography: str,
    theme: str,
    role_hint: StoryRoleHint,
    primary: RepricingEvent,
) -> str:
    readable_theme = theme.replace("_", " ")
    if role_hint == "primary":
        return (
            f"This looks like the primary angle in a {group_size}-story {geography} {readable_theme} cluster. "
            "Adjacent stories probably belong in the same editor review set."
        )
    return (
        f"This overlaps with '{primary.story_group_label}' inside a {group_size}-story {geography} {readable_theme} cluster. "
        "It may fit better as a merged or secondary angle than as a fully standalone story."
    )


def _overlap_signature(event: RepricingEvent) -> tuple[str, str, str] | None:
    text = " ".join(
        part for part in (event.story_group_label, event.market.question, event.market.event_title or "") if part
    )
    tokens = _normalized_tokens(text)
    geography = _dominant_geography_token(tokens)
    theme = _overlap_theme(tokens)
    if geography is None and theme == "general":
        return None
    return (event.market.category, geography or "global", theme)


def _normalized_tokens(text: str) -> tuple[str, ...]:
    raw_tokens = re.findall(r"[a-z0-9']+", text.lower())
    normalized = []
    for token in raw_tokens:
        mapped = _TOKEN_NORMALIZATION.get(token, token)
        if mapped in _OVERLAP_STOPWORDS:
            continue
        normalized.append(mapped)
    return tuple(normalized)


def _dominant_geography_token(tokens: Sequence[str]) -> str | None:
    for token in tokens:
        if token in _GEOGRAPHY_TOKENS:
            return token
    return None


def _overlap_theme(tokens: Sequence[str]) -> str:
    token_set = set(tokens)
    for theme, theme_tokens in _THEME_TOKENS.items():
        if token_set & theme_tokens:
            return theme
    return "general"


def _build_price_context(event: RepricingEvent) -> ResearchPriceContext:
    trace_points = _sample_trace_points(
        snapshots=event.series.snapshots,
        start=event.detection_window_start,
        end=event.detection_window_end,
        interval_hours=8,
    )
    largest_window_hours = 24 if event.max_abs_move_24h >= event.max_abs_move_6h else 6
    largest_move_end = event.max_move_timestamp
    largest_move_start = (
        largest_move_end - timedelta(hours=largest_window_hours) if largest_move_end else None
    )
    surprise_reference_probability = None
    surprise_points = None
    surprise_label = None
    workflow_type = _workflow_type_from_story_hint(event.story_type_hint)
    if workflow_type == "resolution_story":
        surprise_reference_probability = event.window_open_probability
        surprise_points = _resolution_surprise_points(event)
        surprise_label = _surprise_label(surprise_points)
    return ResearchPriceContext(
        interval_hours=8,
        trace_points=trace_points,
        largest_move_window_hours=largest_window_hours,
        largest_move_window_start=largest_move_start,
        largest_move_window_end=largest_move_end,
        surprise_reference_probability=surprise_reference_probability,
        surprise_points=surprise_points,
        surprise_label=surprise_label,
    )


def _sample_trace_points(
    *,
    snapshots: Sequence[Any],
    start: datetime,
    end: datetime,
    interval_hours: int,
) -> tuple[PriceTracePoint, ...]:
    selected = []
    step_seconds = interval_hours * 3600
    latest_probability = None
    cursor = 0
    ordered = tuple(sorted(snapshots, key=lambda item: item.observed_at))
    total_points = max(1, int(((end - start).total_seconds() // step_seconds)) + 1)
    for index in range(total_points + 1):
        target = start + timedelta(seconds=min(index * step_seconds, (end - start).total_seconds()))
        while cursor < len(ordered) and ordered[cursor].observed_at <= target:
            latest_probability = ordered[cursor].probability
            cursor += 1
        if latest_probability is None:
            continue
        move_since_previous = (
            latest_probability - selected[-1].probability if selected else None
        )
        selected.append(
            PriceTracePoint(
                observed_at=target,
                probability=latest_probability,
                move_since_previous=move_since_previous,
            )
        )
    return tuple(selected)


def _resolution_surprise_points(event: RepricingEvent) -> float:
    resolved_up = event.window_close_probability >= 0.5
    reference = event.window_open_probability
    if resolved_up:
        return max(0.0, (1.0 - reference) * 100.0)
    return max(0.0, reference * 100.0)


def _surprise_label(points: float | None) -> str | None:
    if points is None:
        return None
    if points >= 35.0:
        return "high_surprise"
    if points >= 15.0:
        return "moderate_surprise"
    return "low_surprise"


_TOKEN_NORMALIZATION = {
    "slovenian": "slovenia",
    "italian": "italy",
    "iranian": "iran",
    "israeli": "israel",
    "lebanese": "lebanon",
    "american": "us",
    "u.s": "us",
    "u.s.": "us",
    "fed": "us",
}

_OVERLAP_STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "of",
    "by",
    "in",
    "to",
    "be",
    "most",
    "next",
    "party",
    "2026",
}

_GEOGRAPHY_TOKENS = {
    "slovenia",
    "italy",
    "iran",
    "israel",
    "lebanon",
    "china",
    "us",
    "somalia",
    "lyon",
}

_THEME_TOKENS = {
    "politics_government": {
        "election",
        "parliament",
        "parliamentary",
        "minister",
        "prime",
        "government",
        "coalition",
        "referendum",
        "president",
        "mayor",
        "winner",
    },
    "military_conflict": {
        "ceasefire",
        "strike",
        "strikes",
        "offensive",
        "forces",
        "war",
        "missile",
        "attack",
        "invade",
        "military",
        "ground",
    },
    "macro_rates": {
        "rates",
        "rate",
        "inflation",
        "cpi",
        "interest",
        "hike",
        "cut",
    },
    "diplomacy": {
        "visit",
        "meeting",
        "summit",
        "talks",
        "xi",
        "trump",
    },
}
