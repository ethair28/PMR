from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from pmr.models import (
    ChartAsset,
    ComposedSection,
    EditorDecision,
    EditorStoryPacket,
    EvidenceItem,
    ResearchJob,
    WeeklyReport,
)
from pmr.research_payloads import load_research_jobs_from_payload


def load_editor_story_packets_from_files(
    research_results_path: Path,
    research_inputs_path: Path,
) -> tuple[EditorStoryPacket, ...]:
    """Load merged editor inputs from the story-development outputs plus research-job context."""

    results_payload = json.loads(research_results_path.read_text())
    inputs_payload = json.loads(research_inputs_path.read_text())
    return load_editor_story_packets_from_payloads(results_payload, inputs_payload)


def load_editor_story_packets_from_payloads(
    research_results_payload: dict[str, Any],
    research_inputs_payload: dict[str, Any],
) -> tuple[EditorStoryPacket, ...]:
    """Merge story drafts with their richer research-job context for the editor stage."""

    jobs_by_id = {job.job_id: job for job in load_research_jobs_from_payload(research_inputs_payload)}
    packets: list[EditorStoryPacket] = []
    for item in research_results_payload.get("story_drafts", ()):
        job_id = str(item["job_id"])
        if job_id not in jobs_by_id:
            raise ValueError(f"Missing research job context for editor packet '{job_id}'.")
        packets.append(_parse_editor_story_packet(item, jobs_by_id[job_id]))
    return tuple(packets)


def build_weekly_report_payload(report: WeeklyReport) -> dict[str, Any]:
    """Serialize the editor output into the JSON-first canonical weekly-report artifact."""

    return {
        "generated_at": report.generated_at.isoformat(),
        "provider": report.provider,
        "prompt_version": report.prompt_version,
        "model_name": report.model_name,
        "report_title": report.report_title,
        "report_subtitle": report.report_subtitle,
        "opening_markdown": report.opening_markdown,
        "included_story_count": report.included_story_count,
        "merged_story_count": report.merged_story_count,
        "excluded_story_count": report.excluded_story_count,
        "sections": [serialize_composed_section(section) for section in report.sections],
        "decisions": [serialize_editor_decision(decision) for decision in report.decisions],
        "chart_assets": [serialize_chart_asset(asset) for asset in report.chart_assets],
    }


def serialize_composed_section(section: ComposedSection) -> dict[str, Any]:
    return {
        "headline": section.headline,
        "dek": section.dek,
        "bottom_line": section.bottom_line,
        "summary_points": list(section.summary_points),
        "body_markdown": section.body_markdown,
        "included_job_ids": list(section.included_job_ids),
        "detail_level": section.detail_level,
        "primary_chart_job_id": section.primary_chart_job_id,
        "chart_asset_id": section.chart_asset_id,
    }


def serialize_editor_decision(decision: EditorDecision) -> dict[str, Any]:
    return {
        "job_id": decision.job_id,
        "action": decision.action,
        "rationale": decision.rationale,
        "detail_level": decision.detail_level,
        "merge_with": list(decision.merge_with),
        "section_headline": decision.section_headline,
        "section_rank": decision.section_rank,
    }


def serialize_chart_asset(asset: ChartAsset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "job_id": asset.job_id,
        "section_headline": asset.section_headline,
        "local_path": asset.local_path,
        "alt_text": asset.alt_text,
        "width": asset.width,
        "height": asset.height,
    }

def _parse_editor_story_packet(payload: dict[str, Any], job: ResearchJob) -> EditorStoryPacket:
    root_cluster_key, root_cluster_summary = _build_root_cluster(job)
    return EditorStoryPacket(
        job_id=job.job_id,
        family_key=job.family_key,
        family_label=job.family_label,
        workflow_type=job.workflow_type,
        story_type_hint=job.story_type_hint,
        editorial_priority_hint=job.editorial_priority_hint,
        story_role_hint=payload.get("story_role_hint", job.story_role_hint),
        investigation_question=job.investigation_question,
        why_flagged=job.why_flagged,
        focus_points=job.focus_points,
        primary_market=job.primary_market,
        overlap_group_key=payload.get("overlap_group_key", job.overlap_group_key),
        overlap_summary=payload.get("overlap_summary", job.overlap_summary),
        root_cluster_key=root_cluster_key,
        root_cluster_summary=root_cluster_summary,
        suggested_merge_with=tuple(str(item) for item in payload.get("suggested_merge_with", job.suggested_merge_with)),
        related_markets=job.related_markets,
        status=payload["status"],
        explanation_class=payload.get("explanation_class"),
        confidence=float(payload.get("confidence", 0.0)),
        model_name=str(payload.get("model_name") or "unknown"),
        most_plausible_explanation=str(payload.get("most_plausible_explanation", "")),
        why_market_moved=str(payload.get("why_market_moved", "")),
        price_action_summary=str(payload.get("price_action_summary", "")),
        surprise_assessment=str(payload.get("surprise_assessment", "")),
        main_narrative=str(payload.get("main_narrative", "")),
        belief_shift_drivers=tuple(str(item) for item in payload.get("belief_shift_drivers", ())),
        signal_types=tuple(str(item) for item in payload.get("signal_types", ())),
        why_now=str(payload.get("why_now", "")),
        alternative_explanations=tuple(str(item) for item in payload.get("alternative_explanations", ())),
        unresolved_points=tuple(str(item) for item in payload.get("unresolved_points", ())),
        note_to_editor=str(payload.get("note_to_editor", "")),
        draft_headline=str(payload.get("draft_headline", "")),
        draft_markdown=str(payload.get("draft_markdown", "")),
        key_evidence=tuple(_parse_evidence_item(item) for item in payload.get("key_evidence", ())),
        contradictory_evidence=tuple(
            _parse_evidence_item(item) for item in payload.get("contradictory_evidence", ())
        ),
        open_questions=tuple(str(item) for item in payload.get("open_questions", ())),
        completed_at=datetime.fromisoformat(payload["completed_at"]),
        error_message=payload.get("error_message"),
        used_cache=bool(payload.get("used_cache", False)),
    )


def _build_root_cluster(job: ResearchJob) -> tuple[str | None, str | None]:
    text = " ".join(
        item
        for item in (
            job.family_label,
            job.primary_market.question,
            job.primary_market.event_title,
            job.overlap_summary,
        )
        if item
    ).lower()
    normalized = _normalized_tokens(text)
    geography = _dominant_geography_token(normalized)
    if geography is None:
        return None, None
    category = job.primary_market.category
    theme = _root_theme(normalized)
    key = f"{category}:{geography}"
    if theme:
        summary = f"Broader weekly cluster around {geography} with a dominant {theme.replace('_', ' ')} angle."
    else:
        summary = f"Broader weekly cluster around {geography}-related developments."
    return key, summary


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-z0-9]+", text.lower()) if token)


def _dominant_geography_token(tokens: tuple[str, ...]) -> str | None:
    counts: dict[str, int] = {}
    for token in tokens:
        normalized = _GEOGRAPHY_ALIASES.get(token, token)
        if normalized in _GEOGRAPHY_TOKENS:
            counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _root_theme(tokens: tuple[str, ...]) -> str | None:
    for theme, members in _ROOT_THEMES.items():
        if any(token in members for token in tokens):
            return theme
    return None


def _parse_evidence_item(payload: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        source_type=payload["source_type"],
        url=str(payload["url"]),
        title_or_text=str(payload["title_or_text"]),
        author_or_publication=payload.get("author_or_publication"),
        published_at=(
            datetime.fromisoformat(payload["published_at"]) if payload.get("published_at") else None
        ),
        collected_at=datetime.fromisoformat(payload["collected_at"]),
        relevance_score=float(payload["relevance_score"]),
        temporal_proximity_score=float(payload["temporal_proximity_score"]),
        stance=payload["stance"],
        excerpt=str(payload["excerpt"]),
        query=payload.get("query"),
        quality_tier=str(payload.get("quality_tier", "secondary")),
    )


_GEOGRAPHY_ALIASES = {
    "us": "united_states",
    "u": "united_states",
    "s": "united_states",
    "u.s": "united_states",
    "u.s.": "united_states",
    "usa": "united_states",
    "american": "united_states",
}


_GEOGRAPHY_TOKENS = {
    "iran",
    "slovenia",
    "denmark",
    "china",
    "lebanon",
    "israel",
    "somalia",
    "lyon",
    "france",
    "italy",
    "united_states",
}


_ROOT_THEMES = {
    "diplomacy": {"meeting", "ceasefire", "diplomatic", "talks", "negotiation", "visit"},
    "military": {"forces", "strike", "strikes", "military", "offensive", "action", "war"},
    "election": {"election", "pm", "prime", "minister", "winner", "mayoral", "parliamentary"},
    "macro": {"fed", "rates", "hike", "economy", "inflation"},
}
