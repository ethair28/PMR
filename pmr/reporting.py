from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from pmr.config import MonitoringConfig
from pmr.models import RepricingEvent, ResearchFinding


@dataclass(frozen=True, slots=True)
class EventReport:
    event: RepricingEvent
    research: ResearchFinding


def build_markdown_report(
    event_reports: Sequence[EventReport],
    config: MonitoringConfig,
    generated_at: datetime | None = None,
) -> str:
    stamp = generated_at or datetime.now(timezone.utc)
    lines = [
        "# PMR Repricing Report",
        "",
        f"Generated: {stamp.isoformat()}",
        f"Detection window: last {config.detection_window_days} days",
        f"Preferred baseline window: {config.preferred_baseline_window_days} days",
        "",
        "## Detection Rules",
        "",
        f"- Categories: {', '.join(config.target_categories)}",
        f"- Minimum 24h move: {config.min_abs_move_24h * 100:.1f} pts",
        f"- Minimum weekly range: {config.min_weekly_range * 100:.1f} pts",
        f"- Minimum 7d volume: ${config.min_volume_7d_usd:,.0f}",
        f"- Minimum open interest: ${config.min_open_interest_usd:,.0f}",
        f"- Full-history scoring: {config.min_history_days_for_full_scoring}+ days",
        f"- Short-history scoring: {config.min_history_days_for_short_scoring} to {config.min_history_days_for_full_scoring - 1} days",
        "",
    ]

    if not event_reports:
        lines.extend(
            [
                "## Result",
                "",
                "No markets passed the weekly-event detector in this run.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["## Headline Moves", ""])
    for item in event_reports:
        event = item.event
        lines.append(
            "- "
            f"{event.market.question} "
            f"(max 24h move {event.max_abs_move_24h * 100:.1f} pts, "
            f"weekly range {event.weekly_range * 100:.1f} pts, "
            f"{event.story_type_hint}, "
            f"{event.history_mode}, score {event.composite_score:.2f}, "
            f"{item.research.explanation_type}, confidence {item.research.confidence:.0%})"
        )

    lines.extend(["", "## Detailed Analysis", ""])
    for index, item in enumerate(event_reports, start=1):
        event = item.event
        research = item.research

        lines.extend(
            [
                f"### {index}. {event.market.question}",
                "",
                f"- Category: {event.market.category}",
                f"- Market ID: `{event.market.market_id}`",
                f"- Story group: {event.story_group_label} (`{event.story_group_key}`)",
                f"- Story type hint: {event.story_type_hint}",
                f"- Detection window: {event.detection_window_start.isoformat()} to {event.detection_window_end.isoformat()}",
                f"- History mode: {event.history_mode}",
                f"- Detector confidence: {event.confidence_level} ({event.confidence_score:.0%})",
                f"- Window open / close: {event.window_open_probability:.1%} / {event.window_close_probability:.1%}",
                f"- Close-to-open move: {event.close_to_open_move * 100:.1f} percentage points",
                f"- Weekly high / low / range: {event.window_high_probability:.1%} / {event.window_low_probability:.1%} / {event.weekly_range * 100:.1f} pts",
                f"- Entered extreme zone: {'yes' if event.entered_extreme_zone else 'no'}",
                f"- Distance from extremes at close: {event.distance_from_extremes * 100:.1f} pts",
                f"- Max abs 6h move: {event.max_abs_move_6h * 100:.1f} pts",
                f"- Max abs 24h move: {event.max_abs_move_24h * 100:.1f} pts",
                f"- Max move timestamp: {event.max_move_timestamp.isoformat() if event.max_move_timestamp else 'n/a'}",
                f"- Persistence of largest move: {event.persistence_of_largest_move:.0%}",
                f"- Jump count over threshold: {event.jump_count_over_threshold}",
                f"- Baseline medians (6h / 24h / weekly range): "
                f"{_fmt_pct(event.baseline_stats.median_abs_move_6h)} / "
                f"{_fmt_pct(event.baseline_stats.median_abs_move_24h)} / "
                f"{_fmt_pct(event.baseline_stats.median_weekly_range)}",
                f"- Baseline history used: {event.baseline_stats.usable_history_days:.1f} days across {event.baseline_stats.snapshot_count} snapshots",
                f"- Normalized scores (z_6h / z_24h / z_range): {event.z_6h:.1f} / {event.z_24h:.1f} / {event.z_range:.1f}",
                f"- Liquidity (7d / 24h / OI): "
                f"${event.liquidity_metrics.volume_7d_usd:,.0f} / "
                f"${event.liquidity_metrics.volume_24h_usd:,.0f} / "
                f"${event.liquidity_metrics.open_interest_usd:,.0f}",
                f"- Liquidity score: {event.liquidity_metrics.liquidity_score:.2f}",
                f"- Composite score: {event.composite_score:.2f}",
                f"- Research view: {research.explanation_type} ({research.confidence:.0%} confidence)",
                f"- Summary: {research.summary}",
            ]
        )

        if event.notes:
            lines.extend(["- Detector notes:"] + [f"  - {point}" for point in event.notes])
        if event.related_market_questions:
            lines.extend(["- Related markets:"] + [f"  - {question}" for question in event.related_market_questions])
        if research.evidence:
            lines.extend(["- Evidence candidates:"] + [f"  - {point}" for point in research.evidence])
        if research.caveats:
            lines.extend(["- Caveats:"] + [f"  - {point}" for point in research.caveats])
        lines.append("")

    return "\n".join(lines)


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f} pts"
