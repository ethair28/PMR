from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

from pmr.models import ComposedSection, EditorDecision, EditorStoryPacket, WeeklyReport


class EditorComposer(Protocol):
    def compose(
        self,
        stories: Sequence[EditorStoryPacket],
        *,
        provider_name: str,
        prompt_version: str,
        generated_at: datetime,
    ) -> WeeklyReport:
        """Compose a weekly report from the story-development outputs."""


@dataclass(slots=True)
class HeuristicEditorComposer:
    """Deterministic editor/composer used for tests and local fallback."""

    minimum_confidence: float = 0.4

    def compose(
        self,
        stories: Sequence[EditorStoryPacket],
        *,
        provider_name: str,
        prompt_version: str,
        generated_at: datetime,
    ) -> WeeklyReport:
        groups = _group_stories(stories)
        sections: list[tuple[float, ComposedSection]] = []
        decisions: list[EditorDecision] = []

        for group_key, members in groups:
            ordered = sorted(members, key=_editor_score, reverse=True)
            primary = _choose_primary_story(ordered)
            if primary is None:
                for story in ordered:
                    decisions.append(
                        EditorDecision(
                            job_id=story.job_id,
                            action="exclude",
                            rationale="The story draft failed or did not carry enough confidence for inclusion.",
                            detail_level="brief",
                        )
                    )
                continue

            group_score = _editor_score(primary)
            merge_candidates = [
                story for story in ordered if story.job_id != primary.job_id and story.status == "completed"
            ]
            detail_level = _detail_level_for_story(primary)
            section_headline = primary.draft_headline or primary.family_label
            body_markdown = primary.draft_markdown.strip() or _fallback_body(primary)
            if merge_candidates:
                related_bits = []
                for story in merge_candidates[:2]:
                    related_bits.append(
                        f"- **{story.draft_headline or story.family_label}:** {story.main_narrative or story.note_to_editor}"
                    )
                if related_bits:
                    body_markdown = body_markdown.rstrip() + "\n\n### Related Angle\n" + "\n".join(related_bits)

            dek = _build_dek(primary, merge_candidates)
            sections.append(
                (
                    group_score,
                    ComposedSection(
                        headline=section_headline,
                        dek=dek,
                        body_markdown=body_markdown,
                        included_job_ids=tuple([primary.job_id, *[item.job_id for item in merge_candidates]]),
                        detail_level=detail_level,
                    ),
                )
            )
            decisions.append(
                EditorDecision(
                    job_id=primary.job_id,
                    action="include",
                    rationale=_include_rationale(primary, merge_candidates),
                    detail_level=detail_level,
                    merge_with=tuple(item.job_id for item in merge_candidates),
                    section_headline=section_headline,
                )
            )
            for story in merge_candidates:
                decisions.append(
                    EditorDecision(
                        job_id=story.job_id,
                        action="merge",
                        rationale="This story overlaps materially with a stronger weekly story and works better as supporting context than as a standalone section.",
                        detail_level="brief",
                        merge_with=(primary.job_id,),
                        section_headline=section_headline,
                    )
                )
            for story in ordered:
                if story.job_id == primary.job_id or story in merge_candidates:
                    continue
                decisions.append(
                    EditorDecision(
                        job_id=story.job_id,
                        action="exclude",
                        rationale="The story was weaker than the selected story in the same overlap cluster or not strong enough to justify standalone inclusion.",
                        detail_level="brief",
                    )
                )

        ordered_sections = tuple(section for _, section in sorted(sections, key=lambda item: item[0], reverse=True))
        decisions = _assign_section_ranks(decisions, ordered_sections)
        report_title = "PMR Weekly Report"
        report_subtitle = generated_at.strftime("Generated %Y-%m-%d %H:%M UTC")
        if not ordered_sections:
            editorial_summary = "No story drafts were strong enough to justify inclusion in this weekly report."
            overall_note = "The editor/composer did not find a sufficiently strong set of stories for publication."
        else:
            editorial_summary = (
                f"Selected {sum(1 for item in decisions if item.action == 'include')} primary stories, "
                f"merged {sum(1 for item in decisions if item.action == 'merge')} overlaps, and "
                f"excluded {sum(1 for item in decisions if item.action == 'exclude')} weaker candidates. "
                "Repricing stories were favored when they offered stronger belief-shift insight than straightforward resolution stories."
            )
            overall_note = ""
        return WeeklyReport(
            provider=provider_name,
            prompt_version=prompt_version,
            model_name="heuristic_editor",
            generated_at=generated_at,
            report_title=report_title,
            report_subtitle=report_subtitle,
            editorial_summary=editorial_summary,
            overall_note_to_reader=overall_note,
            sections=ordered_sections,
            decisions=tuple(decisions),
        )


@dataclass(slots=True)
class EditorEngine:
    """Coordinate the editor/composer stage over a batch of story drafts."""

    composer: EditorComposer
    provider_name: str
    prompt_version: str

    def run(
        self,
        stories: Sequence[EditorStoryPacket],
        *,
        now: datetime | None = None,
    ) -> WeeklyReport:
        generated_at = now or datetime.now(timezone.utc)
        raw_report = self.composer.compose(
            tuple(stories),
            provider_name=self.provider_name,
            prompt_version=self.prompt_version,
            generated_at=generated_at,
        )
        return _normalize_weekly_report(raw_report, stories)


def _group_stories(stories: Sequence[EditorStoryPacket]) -> tuple[tuple[str, tuple[EditorStoryPacket, ...]], ...]:
    grouped: dict[str, list[EditorStoryPacket]] = {}
    for story in stories:
        key = story.overlap_group_key or story.job_id
        grouped.setdefault(key, []).append(story)
    return tuple((key, tuple(items)) for key, items in grouped.items())


def _editor_score(story: EditorStoryPacket) -> float:
    priority_weight = {"high": 0.35, "medium": 0.2, "low": 0.05}[story.editorial_priority_hint]
    workflow_weight = 0.25 if story.workflow_type == "repricing_story" else 0.15
    status_weight = 0.25 if story.status == "completed" else (-0.25 if story.status == "failed" else 0.0)
    role_weight = 0.05 if story.story_role_hint == "primary" else 0.0
    return (
        story.primary_market.composite_score * 0.12
        + story.confidence * 1.6
        + priority_weight
        + workflow_weight
        + status_weight
        + role_weight
    )


def _choose_primary_story(stories: Sequence[EditorStoryPacket]) -> EditorStoryPacket | None:
    for story in stories:
        if story.status != "completed":
            continue
        if story.confidence < 0.4 and story.workflow_type != "repricing_story":
            continue
        return story
    return None


def _detail_level_for_story(story: EditorStoryPacket) -> str:
    if story.workflow_type == "repricing_story" and story.confidence >= 0.65:
        return "extended"
    if story.editorial_priority_hint == "high" or story.confidence >= 0.55:
        return "standard"
    return "brief"


def _fallback_body(story: EditorStoryPacket) -> str:
    lines = [f"**Why it matters:** {story.main_narrative or story.most_plausible_explanation}"]
    if story.price_action_summary:
        lines.append(f"\n**Price action:** {story.price_action_summary}")
    if story.surprise_assessment:
        lines.append(f"\n**Surprise:** {story.surprise_assessment}")
    return "\n".join(lines).strip()


def _build_dek(primary: EditorStoryPacket, merge_candidates: Sequence[EditorStoryPacket]) -> str:
    role = "repricing" if primary.workflow_type == "repricing_story" else "resolution"
    if merge_candidates:
        return (
            f"{role.capitalize()} story selected as the primary angle for a broader overlap cluster "
            f"with {len(merge_candidates)} supporting market variant(s)."
        )
    return (
        f"{role.capitalize()} story selected as a standalone weekly section because it carried a strong "
        "combination of signal value, explanatory quality, and reader usefulness."
    )


def _include_rationale(primary: EditorStoryPacket, merge_candidates: Sequence[EditorStoryPacket]) -> str:
    if merge_candidates:
        return (
            "Selected as the strongest story in its overlap cluster. The underlying signal was strong enough "
            "to merit inclusion, while related drafts are better treated as supporting context."
        )
    return (
        "Selected as a standalone story because it offers strong reader value, a usable explanatory narrative, "
        "and enough signal to justify dedicated space in the weekly report."
    )


def _assign_section_ranks(
    decisions: Sequence[EditorDecision],
    sections: Sequence[ComposedSection],
) -> list[EditorDecision]:
    section_rank_by_headline = {section.headline: index + 1 for index, section in enumerate(sections)}
    updated: list[EditorDecision] = []
    for decision in decisions:
        rank = (
            section_rank_by_headline.get(decision.section_headline)
            if decision.section_headline is not None
            else None
        )
        updated.append(
            EditorDecision(
                job_id=decision.job_id,
                action=decision.action,
                rationale=decision.rationale,
                detail_level=decision.detail_level,
                merge_with=decision.merge_with,
                section_headline=decision.section_headline,
                section_rank=rank,
            )
        )
    return updated


def _normalize_weekly_report(
    report: WeeklyReport,
    stories: Sequence[EditorStoryPacket],
) -> WeeklyReport:
    story_by_id = {story.job_id: story for story in stories}
    report = _compress_root_clusters(report, story_by_id)
    report = _assign_section_links(report, story_by_id)
    report = _promote_top_section(report, story_by_id)
    report = _assign_section_links(report, story_by_id)
    return report


def _compress_root_clusters(
    report: WeeklyReport,
    story_by_id: dict[str, EditorStoryPacket],
) -> WeeklyReport:
    sections = list(report.sections)
    decisions_by_job = {decision.job_id: decision for decision in report.decisions}
    root_to_indices: dict[str, list[int]] = {}
    for index, section in enumerate(sections):
        root_key = _dominant_root_cluster(section, story_by_id)
        if root_key is None:
            continue
        root_to_indices.setdefault(root_key, []).append(index)

    removed_indices: set[int] = set()
    for _, indices in root_to_indices.items():
        if len(indices) <= 2:
            continue
        ordered = sorted(indices, key=lambda idx: _section_score(sections[idx], story_by_id), reverse=True)
        keeper_indices = ordered[:2]
        primary_index = keeper_indices[0]
        folded_indices = ordered[2:]
        for folded_index in folded_indices:
            sections[primary_index] = _merge_section_into_target(sections[primary_index], sections[folded_index])
            removed_indices.add(folded_index)
            primary_job_id = sections[primary_index].included_job_ids[0] if sections[primary_index].included_job_ids else None
            for job_id in sections[folded_index].included_job_ids:
                existing = decisions_by_job.get(job_id)
                if existing is None:
                    decisions_by_job[job_id] = EditorDecision(
                        job_id=job_id,
                        action="merge",
                        rationale="Compressed into the strongest root-cluster section to avoid over-segmentation.",
                        detail_level="brief",
                        merge_with=(primary_job_id,) if primary_job_id else (),
                    )
                    continue
                decisions_by_job[job_id] = EditorDecision(
                    job_id=existing.job_id,
                    action="merge" if existing.action == "include" else existing.action,
                    rationale=(
                        existing.rationale.rstrip(".")
                        + ". Compressed into the strongest root-cluster section to avoid over-segmentation."
                    ),
                    detail_level="brief" if existing.action == "include" else existing.detail_level,
                    merge_with=existing.merge_with or ((primary_job_id,) if primary_job_id else ()),
                    section_headline=existing.section_headline,
                    section_rank=existing.section_rank,
                )

    if not removed_indices:
        return report

    compressed_sections = tuple(
        section for index, section in enumerate(sections) if index not in removed_indices
    )
    return WeeklyReport(
        provider=report.provider,
        prompt_version=report.prompt_version,
        model_name=report.model_name,
        generated_at=report.generated_at,
        report_title=report.report_title,
        report_subtitle=report.report_subtitle,
        editorial_summary=report.editorial_summary,
        overall_note_to_reader=report.overall_note_to_reader,
        sections=compressed_sections,
        decisions=tuple(decisions_by_job.values()),
    )


def _assign_section_links(
    report: WeeklyReport,
    story_by_id: dict[str, EditorStoryPacket],
) -> WeeklyReport:
    del story_by_id
    section_for_job: dict[str, tuple[str, int]] = {}
    for rank, section in enumerate(report.sections, start=1):
        for job_id in section.included_job_ids:
            section_for_job[job_id] = (section.headline, rank)

    updated_decisions: list[EditorDecision] = []
    for decision in report.decisions:
        headline = decision.section_headline
        rank = decision.section_rank
        if decision.job_id in section_for_job:
            headline, rank = section_for_job[decision.job_id]
        elif decision.merge_with:
            for target in decision.merge_with:
                if target in section_for_job:
                    headline, rank = section_for_job[target]
                    break
        updated_decisions.append(
            EditorDecision(
                job_id=decision.job_id,
                action=decision.action,
                rationale=decision.rationale,
                detail_level=decision.detail_level,
                merge_with=decision.merge_with,
                section_headline=headline,
                section_rank=rank,
            )
        )

    return WeeklyReport(
        provider=report.provider,
        prompt_version=report.prompt_version,
        model_name=report.model_name,
        generated_at=report.generated_at,
        report_title=report.report_title,
        report_subtitle=report.report_subtitle,
        editorial_summary=report.editorial_summary,
        overall_note_to_reader=report.overall_note_to_reader,
        sections=report.sections,
        decisions=tuple(updated_decisions),
    )


def _promote_top_section(
    report: WeeklyReport,
    story_by_id: dict[str, EditorStoryPacket],
) -> WeeklyReport:
    if not report.sections:
        return report
    if any(section.detail_level == "lead" for section in report.sections):
        return report
    top_index = max(
        range(len(report.sections)),
        key=lambda idx: _section_score(report.sections[idx], story_by_id),
    )
    updated_sections: list[ComposedSection] = []
    for index, section in enumerate(report.sections):
        if index == top_index:
            updated_sections.append(
                ComposedSection(
                    headline=section.headline,
                    dek=section.dek,
                    body_markdown=section.body_markdown,
                    included_job_ids=section.included_job_ids,
                    detail_level="lead",
                )
            )
        else:
            updated_sections.append(section)
    return WeeklyReport(
        provider=report.provider,
        prompt_version=report.prompt_version,
        model_name=report.model_name,
        generated_at=report.generated_at,
        report_title=report.report_title,
        report_subtitle=report.report_subtitle,
        editorial_summary=report.editorial_summary,
        overall_note_to_reader=report.overall_note_to_reader,
        sections=tuple(updated_sections),
        decisions=report.decisions,
    )


def _dominant_root_cluster(
    section: ComposedSection,
    story_by_id: dict[str, EditorStoryPacket],
) -> str | None:
    counts: dict[str, int] = {}
    for job_id in section.included_job_ids:
        story = story_by_id.get(job_id)
        if story is None or story.root_cluster_key is None:
            continue
        counts[story.root_cluster_key] = counts.get(story.root_cluster_key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _section_score(
    section: ComposedSection,
    story_by_id: dict[str, EditorStoryPacket],
) -> float:
    scores = [_editor_score(story_by_id[job_id]) for job_id in section.included_job_ids if job_id in story_by_id]
    if not scores:
        return 0.0
    return max(scores) + 0.05 * max(len(section.included_job_ids) - 1, 0)


def _merge_section_into_target(target: ComposedSection, extra: ComposedSection) -> ComposedSection:
    extra_summary = extra.dek or extra.body_markdown.splitlines()[0].strip()
    appendix = (
        "\n\n### Additional Cluster Angle\n"
        f"- **{extra.headline}:** {extra_summary}"
    )
    merged_ids = tuple(dict.fromkeys((*target.included_job_ids, *extra.included_job_ids)))
    detail_level = "lead" if target.detail_level == "lead" else target.detail_level
    return ComposedSection(
        headline=target.headline,
        dek=target.dek,
        body_markdown=target.body_markdown.rstrip() + appendix,
        included_job_ids=merged_ids,
        detail_level=detail_level,
    )
