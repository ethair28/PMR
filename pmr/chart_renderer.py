from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from pmr.models import ChartAsset, ComposedSection, EditorStoryPacket, WeeklyReport


CHART_WIDTH = 1600
CHART_HEIGHT = 900
CHART_DPI = 100


def render_report_charts(
    report: WeeklyReport,
    stories: Sequence[EditorStoryPacket],
    *,
    output_dir: Path,
) -> WeeklyReport:
    """Render selected section charts and attach the resulting assets to the report."""

    story_by_id = {story.job_id: story for story in stories}
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[ChartAsset] = []
    updated_sections: list[ComposedSection] = []

    for section in report.sections:
        chart_asset = None
        if (
            section.primary_chart_job_id is not None
            and section.primary_chart_job_id in story_by_id
        ):
            story = story_by_id[section.primary_chart_job_id]
            chart_asset = _render_section_chart(section, story, output_dir=output_dir)
            if chart_asset is not None:
                assets.append(chart_asset)
        updated_sections.append(
            replace(
                section,
                chart_asset_id=chart_asset.asset_id if chart_asset is not None else None,
            )
        )

    updated_report = replace(
        report,
        sections=tuple(updated_sections),
        chart_assets=tuple(assets),
    )
    return updated_report


def build_chart_manifest_payload(report: WeeklyReport) -> dict[str, object]:
    return {
        "chart_count": len(report.chart_assets),
        "assets": [
            {
                "asset_id": asset.asset_id,
                "job_id": asset.job_id,
                "section_headline": asset.section_headline,
                "local_path": asset.local_path,
                "alt_text": asset.alt_text,
                "width": asset.width,
                "height": asset.height,
            }
            for asset in report.chart_assets
        ],
    }


def _render_section_chart(
    section: ComposedSection,
    story: EditorStoryPacket,
    *,
    output_dir: Path,
) -> ChartAsset | None:
    points = story.primary_market.price_context.chart_trace_points
    if len(points) < 2:
        return None

    dates = [point.observed_at for point in points]
    probabilities = [point.probability * 100.0 for point in points]
    fig, ax = plt.subplots(figsize=(CHART_WIDTH / CHART_DPI, CHART_HEIGHT / CHART_DPI), dpi=CHART_DPI)
    ax.plot(dates, probabilities, linewidth=2.8, color="#2563eb")
    ax.set_title(story.primary_market.question, fontsize=17, fontweight="bold", loc="left")
    ax.set_ylabel("")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", alpha=0.18)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.autofmt_xdate()

    asset_id = _slugify(section.headline)
    output_path = output_dir / f"{asset_id}.png"
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=CHART_DPI)
    plt.close(fig)

    return ChartAsset(
        asset_id=asset_id,
        job_id=story.job_id,
        section_headline=section.headline,
        local_path=str(output_path),
        alt_text=_build_alt_text(section, story),
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
    )


def _build_alt_text(section: ComposedSection, story: EditorStoryPacket) -> str:
    return (
        f"7-day Polymarket probability chart for {story.primary_market.question}. "
        f"The market moved from {story.primary_market.window_open_probability * 100:.1f}% "
        f"to {story.primary_market.window_close_probability * 100:.1f}% over the week."
    )


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "chart"
