from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from xai_sdk import Client
from xai_sdk.chat import system, user

from pmr.models import ComposedSection, EditorDecision, EditorStoryPacket, WeeklyReport


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_EDITOR_MODEL = "grok-4.20-multi-agent-latest"


class _EditorDecisionRecord(BaseModel):
    job_id: str
    action: Literal["include", "merge", "exclude"]
    rationale: str
    detail_level: Literal["brief", "standard", "extended", "lead"] = "standard"
    merge_with: list[str] = Field(default_factory=list)
    section_headline: str | None = None
    section_rank: int | None = None


class _EditorSectionRecord(BaseModel):
    headline: str
    dek: str = ""
    bottom_line: str = ""
    summary_points: list[str] = Field(default_factory=list)
    body_markdown: str
    included_job_ids: list[str] = Field(default_factory=list)
    detail_level: Literal["brief", "standard", "extended", "lead"] = "standard"


class _EditorEnvelope(BaseModel):
    report_title: str
    report_subtitle: str = ""
    opening_markdown: str
    sections: list[_EditorSectionRecord] = Field(default_factory=list)
    decisions: list[_EditorDecisionRecord] = Field(default_factory=list)


@dataclass(slots=True)
class XaiEditorComposer:
    """xAI SDK-backed weekly editor/composer using Grok multi-agent."""

    api_key: str
    model: str = DEFAULT_EDITOR_MODEL
    base_url: str = DEFAULT_XAI_BASE_URL
    timeout_seconds: int = 90
    agent_count: int = 4
    _client: Client | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> "XaiEditorComposer":
        return cls(
            api_key=os.environ["XAI_API_KEY"],
            model=os.environ.get("PMR_XAI_EDITOR_MODEL", DEFAULT_EDITOR_MODEL),
            base_url=os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL),
        )

    def compose(
        self,
        stories: Sequence[EditorStoryPacket],
        *,
        provider_name: str,
        prompt_version: str,
        generated_at: datetime,
    ) -> WeeklyReport:
        chat = self._get_client().chat.create(
            model=self.model,
            temperature=0.1,
            store_messages=False,
            agent_count=self.agent_count,
        )
        chat.append(system(EDITOR_SYSTEM_PROMPT))
        chat.append(user(_build_editor_prompt(stories, generated_at=generated_at)))
        _, parsed = chat.parse(_EditorEnvelope)
        sections = tuple(
            ComposedSection(
                headline=item.headline.strip(),
                body_markdown=item.body_markdown.strip(),
                included_job_ids=tuple(job_id for job_id in item.included_job_ids if job_id),
                detail_level=item.detail_level,
                dek=item.dek.strip(),
                bottom_line=item.bottom_line.strip(),
                summary_points=tuple(point.strip() for point in item.summary_points if point.strip()),
                primary_chart_job_id=None,
                chart_asset_id=None,
            )
            for item in parsed.sections
        )
        return WeeklyReport(
            provider=provider_name,
            prompt_version=prompt_version,
            model_name=self.model,
            generated_at=generated_at,
            report_title=parsed.report_title.strip(),
            report_subtitle=parsed.report_subtitle.strip(),
            opening_markdown=parsed.opening_markdown.strip(),
            sections=sections,
            decisions=tuple(
                EditorDecision(
                    job_id=item.job_id,
                    action=item.action,
                    rationale=item.rationale.strip(),
                    detail_level=item.detail_level,
                    merge_with=tuple(job_id for job_id in item.merge_with if job_id),
                    section_headline=item.section_headline.strip() if item.section_headline else None,
                    section_rank=item.section_rank,
                )
                for item in parsed.decisions
            ),
        )

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(
                api_key=self.api_key,
                api_host=_normalize_api_host(self.base_url),
                timeout=float(self.timeout_seconds),
            )
        return self._client


EDITOR_SYSTEM_PROMPT = """You are the weekly editor/composer for PMR, an automated prediction-market news-intelligence product.

Product philosophy:
- PMR is about changes in belief, not about covering the entire news cycle.
- Prediction-market repricing is the relevance filter.
- Resolution stories add value by quantifying surprise.
- Repricing stories are the product's core differentiator because they explain why market perception shifted before there is a fully resolved public narrative.
- The goal is to maximize reader value and signal density, not to maximize story count.
- It is acceptable to publish zero stories in a weak week.
- It is acceptable to publish a longer report when the week genuinely contains many strong stories.
- The system is fully automated, but editorial judgment remains a first-class layer inside the system.

Editorial instructions:
- Use the supplied price context to judge whether the narrative actually fits the move.
- Prefer stories with real informational value, strong explanation quality, and genuine relevance to changing belief.
- Merge overlapping drafts when that improves clarity and avoids duplication.
- Default to one main section per broader root cluster. A second section from the same root cluster is allowed only when it adds clearly distinct reader value.
- Be ruthless about compression in dense geopolitical clusters. If three sections are all part of the same underlying episode, that is usually too many.
- Give more detail to stronger stories and less detail to marginal ones. Use `lead`, `standard`, `extended`, or `brief` intentionally.
- Resolution stories should emphasize surprise calibration.
- Repricing stories should emphasize what changed perception, what remains uncertain, and why the move matters.
- Make the report easy to read and high-signal. Readers should understand the main value quickly, but you do not need to force every section into the same template.
- You may vary pacing, section shape, and how directly you front-load the takeaway when that improves the report.
- A separate downstream layer only renders the report and attaches a simple primary-market weekly chart to each section. Do not optimize for packaging formats.
- Your single publication target is the final weekly report markdown. Do not optimize for X posts, thread structure, or cross-channel shareability.
- Return a decision for every candidate story.
- Every decision must include the section headline and section rank for included or merged stories.
- Return structured sections for the final report plus clear decision rationales.
"""


def _build_editor_prompt(stories: Sequence[EditorStoryPacket], *, generated_at: datetime) -> str:
    lines = [
        f"Weekly editor batch timestamp: {generated_at.isoformat()}",
        f"Candidate story count: {len(stories)}",
        "Decide which stories to include, merge, or exclude. Use dynamic judgment rather than quotas.",
        "",
        "Root clusters:",
        *_format_root_cluster_overview(stories),
        "",
        "Stories:",
    ]
    for index, story in enumerate(sorted(stories, key=_prompt_story_sort_key, reverse=True), start=1):
        lines.extend(
            [
                f"{index}. job_id={story.job_id}",
                f"   family_label={story.family_label}",
                f"   workflow_type={story.workflow_type}",
                f"   story_type_hint={story.story_type_hint}",
                f"   editorial_priority_hint={story.editorial_priority_hint}",
                f"   story_role_hint={story.story_role_hint}",
                f"   status={story.status}",
                f"   explanation_class={story.explanation_class or 'n/a'}",
                f"   confidence={story.confidence:.2f}",
                f"   composite_score={story.primary_market.composite_score:.2f}",
                f"   market_question={story.primary_market.question}",
                f"   event_title={story.primary_market.event_title or 'n/a'}",
                f"   why_flagged={story.why_flagged}",
                f"   investigation_question={story.investigation_question}",
                f"   price_action={story.price_action_summary}",
                f"   surprise_assessment={story.surprise_assessment or 'n/a'}",
                f"   belief_shift_drivers={'; '.join(story.belief_shift_drivers) or 'none'}",
                f"   signal_types={'; '.join(story.signal_types) or 'none'}",
                f"   why_now={story.why_now or 'n/a'}",
                f"   overlap_context={_format_overlap(story)}",
                f"   root_cluster={story.root_cluster_key or 'standalone'}",
                f"   root_cluster_summary={story.root_cluster_summary or 'n/a'}",
                f"   price_trace_8h={_format_price_trace(story)}",
                f"   largest_move_window={_format_largest_move_window(story)}",
                f"   note_to_editor={story.note_to_editor or 'n/a'}",
                f"   main_narrative={story.main_narrative or story.most_plausible_explanation}",
                f"   alternative_explanations={'; '.join(story.alternative_explanations) or 'none'}",
                f"   unresolved_points={'; '.join(story.unresolved_points) or 'none'}",
                f"   open_questions={'; '.join(story.open_questions) or 'none'}",
                "   key_evidence:",
                *_format_evidence_lines(story.key_evidence, indent="     - "),
                "   contradictory_evidence:",
                *_format_evidence_lines(story.contradictory_evidence, indent="     - "),
                "   related_markets:",
                *_format_related_market_lines(story, indent="     - "),
                "   draft_markdown:",
                story.draft_markdown.strip() or "     (none)",
                "",
            ]
        )
    lines.extend(
        [
            "Return:",
            "- a final report title and subtitle",
            "- an opening_markdown field that opens the report in the way you judge best",
            "- zero or more final sections with markdown bodies",
            "- optional bottom_line and summary_points on sections when they genuinely help readability",
            "- one explicit decision per job_id",
            "- section_headline and section_rank on every included or merged decision",
        ]
    )
    return "\n".join(lines)


def _format_overlap(story: EditorStoryPacket) -> str:
    if not story.overlap_group_key:
        return "standalone"
    merge_targets = ", ".join(story.suggested_merge_with) if story.suggested_merge_with else "none"
    return (
        f"group={story.overlap_group_key}; summary={story.overlap_summary or 'n/a'}; "
        f"suggested_merge_with={merge_targets}"
    )


def _format_price_trace(story: EditorStoryPacket) -> str:
    trace = story.primary_market.price_context.trace_points
    if not trace:
        return "none"
    return "; ".join(
        f"{item.observed_at.isoformat()}={item.probability * 100:.1f}%"
        + (f" ({item.move_since_previous * 100:+.1f} pts)" if item.move_since_previous is not None else "")
        for item in trace
    )


def _format_largest_move_window(story: EditorStoryPacket) -> str:
    context = story.primary_market.price_context
    return (
        f"{context.largest_move_window_hours}h "
        f"from {context.largest_move_window_start.isoformat() if context.largest_move_window_start else 'n/a'} "
        f"to {context.largest_move_window_end.isoformat() if context.largest_move_window_end else 'n/a'}"
    )


def _format_evidence_lines(evidence: Sequence, *, indent: str) -> list[str]:
    if not evidence:
        return [indent + "none"]
    lines = []
    for item in evidence[:5]:
        published_at = item.published_at.isoformat() if item.published_at else "n/a"
        lines.append(
            indent
            + f"[{item.source_type}] {item.title_or_text} | {item.author_or_publication or 'n/a'} | "
            + f"published_at={published_at} | relevance={item.relevance_score:.2f} | temporal={item.temporal_proximity_score:.2f}"
        )
    return lines


def _format_related_market_lines(story: EditorStoryPacket, *, indent: str) -> list[str]:
    if not story.related_markets:
        return [indent + "none"]
    return [indent + f"{item.market_id}: {item.question}" for item in story.related_markets]


def _format_root_cluster_overview(stories: Sequence[EditorStoryPacket]) -> list[str]:
    grouped: dict[str, list[EditorStoryPacket]] = {}
    for story in stories:
        key = story.root_cluster_key or f"standalone:{story.job_id}"
        grouped.setdefault(key, []).append(story)
    lines: list[str] = []
    for key, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        label = members[0].root_cluster_summary or "Standalone story."
        member_list = ", ".join(member.job_id for member in members)
        lines.append(f"- {key}: {label} Members: {member_list}")
    return lines


def _prompt_story_sort_key(story: EditorStoryPacket) -> tuple[float, float]:
    workflow_bias = 0.5 if story.workflow_type == "repricing_story" else 0.0
    return (story.confidence + workflow_bias, story.primary_market.composite_score)


def _normalize_api_host(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        return parsed.netloc or parsed.path
    return value.rstrip("/")
