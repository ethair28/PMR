from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from pmr.config import MonitoringConfig
from pmr.models import RepricingEvent
from pmr.story_groups import build_story_family_key, build_story_family_label


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
            "family_key": build_story_family_key(event.market),
            "family_label": build_story_family_label(event.market),
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
