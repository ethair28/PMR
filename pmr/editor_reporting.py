from __future__ import annotations

from pmr.models import WeeklyReport


def render_weekly_report_markdown(report: WeeklyReport) -> str:
    """Render the reader-facing weekly report from the structured editor output."""

    lines = [f"# {report.report_title}"]
    if report.report_subtitle:
        lines.extend(["", report.report_subtitle])
    if report.editorial_summary:
        lines.extend(["", report.editorial_summary])
    if report.overall_note_to_reader:
        lines.extend(["", f"_Editor note: {report.overall_note_to_reader}_"])

    if not report.sections:
        lines.extend(
            [
                "",
                "## No Strong Stories",
                "",
                "The editor/composer did not find a strong enough set of weekly stories to publish this run.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    for section in report.sections:
        lines.extend(["", f"## {section.headline}"])
        if section.dek:
            lines.extend(["", section.dek])
        lines.extend(["", section.body_markdown.strip()])

    return "\n".join(lines).strip() + "\n"


def render_editor_decisions_markdown(report: WeeklyReport) -> str:
    """Render a separate document that explains the editor/composer's choices."""

    lines = ["# Editor Decisions"]
    lines.extend(["", f"- Included: {report.included_story_count}"])
    lines.extend(["- Merged: " + str(report.merged_story_count)])
    lines.extend(["- Excluded: " + str(report.excluded_story_count)])
    if report.editorial_summary:
        lines.extend(["", report.editorial_summary])

    if not report.decisions:
        lines.extend(["", "No explicit editorial decisions were returned."])
        return "\n".join(lines).strip() + "\n"

    for decision in report.decisions:
        title = decision.job_id
        lines.extend(["", f"## {title}", ""])
        lines.append(f"- Action: `{decision.action}`")
        lines.append(f"- Detail level: `{decision.detail_level}`")
        if decision.section_headline:
            lines.append(f"- Section: {decision.section_headline}")
        if decision.section_rank is not None:
            lines.append(f"- Section rank: {decision.section_rank}")
        if decision.merge_with:
            lines.append("- Merge with: " + ", ".join(decision.merge_with))
        lines.extend(["", decision.rationale])

    return "\n".join(lines).strip() + "\n"
