from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from pmr.models import ComposedSection, EditorStoryPacket, WeeklyReport


def package_weekly_report(
    report: WeeklyReport,
    stories: Sequence[EditorStoryPacket],
) -> WeeklyReport:
    """Attach deterministic render-time metadata without authoring report prose."""

    story_by_id = {story.job_id: story for story in stories}
    return _attach_primary_chart_targets(report, story_by_id)


def _attach_primary_chart_targets(
    report: WeeklyReport,
    story_by_id: dict[str, EditorStoryPacket],
) -> WeeklyReport:
    updated_sections: list[ComposedSection] = []
    for section in report.sections:
        primary_story = _primary_story(section, story_by_id)
        updated_sections.append(
            replace(
                section,
                primary_chart_job_id=primary_story.job_id if primary_story is not None else None,
                chart_asset_id=None,
            )
        )
    return replace(report, sections=tuple(updated_sections))


def _primary_story(
    section: ComposedSection,
    story_by_id: dict[str, EditorStoryPacket],
) -> EditorStoryPacket | None:
    for job_id in section.included_job_ids:
        if job_id in story_by_id:
            return story_by_id[job_id]
    return None
