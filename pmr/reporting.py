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
        f"Window: last {config.lookback_days} days",
        "",
        "## Detection Rules",
        "",
        f"- Categories: {', '.join(config.target_categories)}",
        f"- Minimum 7d volume: ${config.min_volume_7d_usd:,.0f}",
        f"- Minimum open interest: ${config.min_open_interest_usd:,.0f}",
        f"- Minimum absolute move: {config.min_abs_move * 100:.1f} pts",
        f"- Minimum anomaly ratio: {config.min_anomaly_ratio:.1f}x baseline daily move",
        "",
    ]

    if not event_reports:
        lines.extend(
            [
                "## Result",
                "",
                "No markets passed the significance thresholds in this run.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["## Headline Moves", ""])
    for item in event_reports:
        event = item.event
        direction = "up" if event.move >= 0 else "down"
        lines.append(
            "- "
            f"{event.market.question} "
            f"({event.start_snapshot.probability:.0%} to {event.end_snapshot.probability:.0%}, "
            f"{direction} {event.abs_move * 100:.1f} pts, "
            f"{item.research.explanation_type}, "
            f"confidence {item.research.confidence:.0%})"
        )

    lines.extend(["", "## Detailed Analysis", ""])
    for index, item in enumerate(event_reports, start=1):
        event = item.event
        research = item.research
        direction = "increase" if event.move >= 0 else "decrease"

        lines.extend(
            [
                f"### {index}. {event.market.question}",
                "",
                f"- Category: {event.market.category}",
                f"- Market ID: `{event.market.market_id}`",
                f"- Move: {direction} from {event.start_snapshot.probability:.1%} to {event.end_snapshot.probability:.1%}",
                f"- Absolute move: {event.abs_move * 100:.1f} percentage points",
                f"- Baseline daily move: {event.baseline_abs_daily_move * 100:.1f} percentage points",
                f"- Anomaly ratio: {event.anomaly_ratio:.1f}x",
                f"- 7d volume / OI: ${event.market.volume_7d_usd:,.0f} / ${event.market.open_interest_usd:,.0f}",
                f"- Research view: {research.explanation_type} ({research.confidence:.0%} confidence)",
                f"- Summary: {research.summary}",
            ]
        )

        if research.evidence:
            lines.extend(["- Evidence candidates:"] + [f"  - {point}" for point in research.evidence])
        if research.caveats:
            lines.extend(["- Caveats:"] + [f"  - {point}" for point in research.caveats])
        lines.append("")

    return "\n".join(lines)
