from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from xai_sdk import Client
from xai_sdk.chat import system, user
from xai_sdk.tools import web_search, x_search

from pmr.models import EvidenceItem, ResearchJob, ResearchQueryPlan, ResearchResult, StoryWorkflowType


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_REASONING_MODEL = "grok-4.20-reasoning-latest"
DEFAULT_MULTI_AGENT_MODEL = "grok-4.20-multi-agent-latest"
DEFAULT_XAI_MODEL = DEFAULT_REASONING_MODEL
DEFAULT_REASONING_MODEL_CANDIDATES = (
    "grok-4.20-reasoning-latest",
    "grok-4.20",
    "grok-4-1-fast-reasoning-latest",
)
DEFAULT_MULTI_AGENT_MODEL_CANDIDATES = (
    "grok-4.20-multi-agent-latest",
    "grok-4.20-multi-agent",
)


class _EvidenceRecord(BaseModel):
    source_type: Literal["x_post", "web_article", "news_article"]
    url: str
    title_or_text: str
    author_or_publication: str | None = None
    published_at: datetime | None = None
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    temporal_proximity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    stance: Literal["supporting", "contradictory", "contextual"] = "contextual"
    excerpt: str | None = None
    query: str | None = None


class _EvidenceEnvelope(BaseModel):
    evidence: list[_EvidenceRecord] = Field(default_factory=list)


class _SynthesisEnvelope(BaseModel):
    status: Literal["completed", "insufficient_evidence", "failed"] = "failed"
    explanation_class: Literal["clear", "plausible", "speculative"] | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    most_plausible_explanation: str = ""
    why_market_moved: str = ""
    price_action_summary: str = ""
    surprise_assessment: str = ""
    main_narrative: str = ""
    alternative_explanations: list[str] = Field(default_factory=list)
    note_to_editor: str = ""
    draft_headline: str = ""
    draft_markdown: str = ""
    open_questions: list[str] = Field(default_factory=list)
    error_message: str | None = None


@dataclass(slots=True)
class _XaiSdkAdapterBase:
    api_key: str
    model: str | None = None
    base_url: str = DEFAULT_XAI_BASE_URL
    timeout_seconds: int = 60
    _client: Client | None = field(default=None, init=False, repr=False)
    _available_model_names: set[str] | None = field(default=None, init=False, repr=False)
    _resolved_models: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(
                api_key=self.api_key,
                api_host=_normalize_api_host(self.base_url),
                timeout=float(self.timeout_seconds),
            )
        return self._client

    def _get_model_for_job(self, job: ResearchJob) -> str:
        if self.model:
            return self.model
        workflow_key = job.workflow_type
        if workflow_key in self._resolved_models:
            return self._resolved_models[workflow_key]
        if self._available_model_names is None:
            self._available_model_names = _list_language_model_names(self._get_client())
        resolved = _choose_best_model_name(self._available_model_names, workflow_type=job.workflow_type)
        self._resolved_models[workflow_key] = resolved
        return resolved


@dataclass(slots=True)
class XaiResearchSource(_XaiSdkAdapterBase):
    """xAI SDK-backed research retriever that asks for X-first evidence plus web corroboration."""

    max_evidence_items: int = 12

    @classmethod
    def from_env(cls) -> "XaiResearchSource":
        api_key = os.environ["XAI_API_KEY"]
        model = os.environ.get("PMR_XAI_MODEL") or None
        base_url = os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
        return cls(api_key=api_key, model=model, base_url=base_url)

    def search(self, job: ResearchJob, query_plan: ResearchQueryPlan) -> Sequence[EvidenceItem]:
        prompt = _build_source_prompt(job, query_plan, max_evidence_items=self.max_evidence_items)
        model_name = self._get_model_for_job(job)
        chat = self._get_client().chat.create(
            model=model_name,
            temperature=0.1,
            tools=[
                x_search(
                    from_date=query_plan.time_window_start,
                    to_date=query_plan.time_window_end,
                ),
                web_search(),
            ],
            tool_choice="required",
            store_messages=False,
            parallel_tool_calls=True,
            **_multi_agent_kwargs(model_name),
        )
        chat.append(system(SOURCE_SYSTEM_PROMPT))
        chat.append(user(prompt))
        _, parsed = chat.parse(_EvidenceEnvelope)
        collected_at = datetime.now(timezone.utc)
        return tuple(
            EvidenceItem(
                source_type=item.source_type,
                url=item.url.strip(),
                title_or_text=item.title_or_text.strip(),
                author_or_publication=_none_if_blank(item.author_or_publication),
                published_at=item.published_at,
                collected_at=collected_at,
                relevance_score=_clamp_float(item.relevance_score, default=0.5),
                temporal_proximity_score=_clamp_float(item.temporal_proximity_score, default=0.5),
                stance=item.stance,
                excerpt=_truncate((item.excerpt or item.title_or_text).strip()),
                query=_none_if_blank(item.query),
            )
            for item in parsed.evidence
            if item.url.strip() and item.title_or_text.strip()
        )


@dataclass(slots=True)
class XaiResearchSynthesizer(_XaiSdkAdapterBase):
    """xAI SDK-backed synthesizer that turns normalized evidence into a structured result."""

    @classmethod
    def from_env(cls) -> "XaiResearchSynthesizer":
        api_key = os.environ["XAI_API_KEY"]
        model = os.environ.get("PMR_XAI_MODEL") or None
        base_url = os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
        return cls(api_key=api_key, model=model, base_url=base_url)

    def summarize(
        self,
        job: ResearchJob,
        query_plan: ResearchQueryPlan,
        evidence: Sequence[EvidenceItem],
        *,
        cache_key: str,
        provider_name: str,
        prompt_version: str,
        generated_at: datetime,
    ) -> ResearchResult:
        model_name = self._get_model_for_job(job)
        chat = self._get_client().chat.create(
            model=model_name,
            temperature=0.1,
            store_messages=False,
            **_multi_agent_kwargs(model_name),
        )
        chat.append(system(SYNTHESIS_SYSTEM_PROMPT))
        chat.append(
            user(
                _build_synthesis_prompt(
                    job=job,
                    query_plan=query_plan,
                    evidence=evidence,
                )
            )
        )
        _, parsed = chat.parse(_SynthesisEnvelope)
        return ResearchResult(
            job_id=job.job_id,
            cache_key=cache_key,
            provider=provider_name,
            prompt_version=prompt_version,
            model_name=model_name,
            workflow_type=job.workflow_type,
            story_role_hint=job.story_role_hint,
            status=parsed.status,
            explanation_class=parsed.explanation_class,
            confidence=_clamp_float(parsed.confidence, default=0.0),
            most_plausible_explanation=parsed.most_plausible_explanation.strip(),
            why_market_moved=parsed.why_market_moved.strip(),
            price_action_summary=parsed.price_action_summary.strip(),
            surprise_assessment=parsed.surprise_assessment.strip(),
            main_narrative=parsed.main_narrative.strip(),
            alternative_explanations=tuple(item.strip() for item in parsed.alternative_explanations if item.strip()),
            note_to_editor=parsed.note_to_editor.strip(),
            draft_headline=parsed.draft_headline.strip(),
            draft_markdown=parsed.draft_markdown.strip(),
            overlap_group_key=job.overlap_group_key,
            overlap_summary=job.overlap_summary,
            suggested_merge_with=job.suggested_merge_with,
            key_evidence=tuple(evidence[:5]),
            contradictory_evidence=tuple(item for item in evidence if item.stance == "contradictory")[:3],
            open_questions=tuple(item.strip() for item in parsed.open_questions if item.strip()),
            completed_at=generated_at,
            error_message=_none_if_blank(parsed.error_message),
        )


SOURCE_SYSTEM_PROMPT = """You are a research retriever for a prediction-market story-development system.

Use x_search first and web_search second. Gather evidence about why the market repriced near the supplied move window.

Return only structured data matching the provided schema. Do not invent URLs, authors, or publication dates. If evidence is weak, return fewer items rather than guessing. Prefer evidence that is temporally close to the move timestamp and clearly relevant to the market question. Include at least one contradictory or cautionary item when the evidence is contested or incomplete.

Prefer underlying reporting, official statements, and directly relevant local/news accounts over posts that merely comment on Polymarket odds or trading flow.
"""


SYNTHESIS_SYSTEM_PROMPT = """You are a prediction-market story developer.

You will receive a market, a search plan, weekly price-action context, and normalized evidence items that already came from xAI search tools. Return only structured data matching the provided schema.

If the story is a resolution story, quantify the surprise clearly. If it is a repricing story, focus on the best narrative for why market perception shifted without claiming false certainty. Always leave a note for the editor. If the evidence is weak or conflicting, use "insufficient_evidence" or "speculative" rather than overstating certainty.

Draft requirements:
- keep the tone concise, factual, and information-dense
- lead with the trigger and the market move
- for repricing stories, distinguish what happened from why traders changed their minds
- for resolution stories, explicitly quantify the surprise
- make note_to_editor short and actionable
- if overlap metadata suggests a merge, mention that directly in note_to_editor
"""


def _build_source_prompt(job: ResearchJob, query_plan: ResearchQueryPlan, *, max_evidence_items: int) -> str:
    return (
        f"Investigate this repricing and return at most {max_evidence_items} normalized evidence items.\n\n"
        f"Story: {job.family_label}\n"
        f"Workflow type: {job.workflow_type}\n"
        f"Story type hint: {job.story_type_hint}\n"
        f"Story role hint: {job.story_role_hint}\n"
        f"Editorial priority hint: {job.editorial_priority_hint}\n"
        f"Investigation question: {job.investigation_question}\n"
        f"Why flagged: {job.why_flagged}\n"
        f"Primary market question: {job.primary_market.question}\n"
        f"Primary market category: {job.primary_market.category}\n"
        f"Primary market URL: {job.primary_market.url or 'n/a'}\n"
        f"Primary event title: {job.primary_market.event_title or 'n/a'}\n"
        f"Detection window: {job.primary_market.detection_window_start.isoformat()} to "
        f"{job.primary_market.detection_window_end.isoformat()}\n"
        f"Max move timestamp: "
        f"{job.primary_market.max_move_timestamp.isoformat() if job.primary_market.max_move_timestamp else 'n/a'}\n"
        f"Weekly price trace (8h): {_format_price_trace(job)}\n"
        f"Surprise context: {_format_surprise_context(job)}\n"
        f"Overlap context: {_format_overlap_context(job)}\n"
        f"Focus points: {', '.join(query_plan.focus_points) if query_plan.focus_points else 'none'}\n"
        f"X queries: {', '.join(query_plan.x_queries)}\n"
        f"Web queries: {', '.join(query_plan.web_queries)}\n"
        f"Related markets: {_format_related_markets(job)}\n"
        "Favor posts/articles that directly explain a change in perceived probability, not just generic background. "
        "Treat posts that only comment on Polymarket itself as weak unless they point to underlying reporting or information flow."
    )


def _build_synthesis_prompt(
    *,
    job: ResearchJob,
    query_plan: ResearchQueryPlan,
    evidence: Sequence[EvidenceItem],
) -> str:
    lines = [
        f"Story: {job.family_label}",
        f"Workflow type: {job.workflow_type}",
        f"Story type hint: {job.story_type_hint}",
        f"Story role hint: {job.story_role_hint}",
        f"Investigation question: {job.investigation_question}",
        f"Why flagged: {job.why_flagged}",
        f"Primary market: {job.primary_market.question}",
        f"Detection window: {query_plan.time_window_start.isoformat()} to {query_plan.time_window_end.isoformat()}",
        f"Focus timestamp: {query_plan.focus_timestamp.isoformat() if query_plan.focus_timestamp else 'n/a'}",
        f"Price action summary packet: {_format_price_context(job)}",
        f"Overlap context: {_format_overlap_context(job)}",
        "Evidence:",
    ]
    for index, item in enumerate(evidence, start=1):
        published_at = item.published_at.isoformat() if item.published_at else "n/a"
        lines.append(
            f"{index}. [{item.source_type}] {item.title_or_text} | url={item.url} | "
            f"author={item.author_or_publication or 'n/a'} | published_at={published_at} | "
            f"relevance={item.relevance_score:.2f} | temporal={item.temporal_proximity_score:.2f} | "
            f"stance={item.stance}"
        )
    return "\n".join(lines)


def _format_related_markets(job: ResearchJob) -> str:
    if not job.related_markets:
        return "none"
    return "; ".join(f"{item.market_id}: {item.question}" for item in job.related_markets)


def _format_price_trace(job: ResearchJob) -> str:
    if not job.primary_market.price_context.trace_points:
        return "none"
    return "; ".join(
        (
            f"{item.observed_at.isoformat()}={item.probability * 100:.1f}%"
            + (
                f" ({item.move_since_previous * 100:+.1f} pts)"
                if item.move_since_previous is not None
                else ""
            )
        )
        for item in job.primary_market.price_context.trace_points
    )


def _format_surprise_context(job: ResearchJob) -> str:
    context = job.primary_market.price_context
    if context.surprise_points is None:
        return "repricing_story or no clean resolution surprise metric"
    return (
        f"reference={context.surprise_reference_probability * 100:.1f}% "
        f"surprise={context.surprise_points:.1f} pts "
        f"label={context.surprise_label or 'n/a'}"
    )


def _format_price_context(job: ResearchJob) -> str:
    context = job.primary_market.price_context
    return (
        f"open={job.primary_market.window_open_probability * 100:.1f}% "
        f"close={job.primary_market.window_close_probability * 100:.1f}% "
        f"high={job.primary_market.window_high_probability * 100:.1f}% "
        f"low={job.primary_market.window_low_probability * 100:.1f}% "
        f"range={job.primary_market.weekly_range * 100:.1f} pts "
        f"max24h={job.primary_market.max_abs_move_24h * 100:.1f} pts "
        f"largest_window={context.largest_move_window_hours}h "
        f"from={context.largest_move_window_start.isoformat() if context.largest_move_window_start else 'n/a'} "
        f"to={context.largest_move_window_end.isoformat() if context.largest_move_window_end else 'n/a'}"
    )


def _format_overlap_context(job: ResearchJob) -> str:
    if not job.overlap_group_key:
        return "standalone"
    return (
        f"group={job.overlap_group_key} role={job.story_role_hint} "
        f"merge_with={', '.join(job.suggested_merge_with) if job.suggested_merge_with else 'none'} "
        f"summary={job.overlap_summary or 'n/a'}"
    )


def _normalize_api_host(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        if parsed.hostname:
            if parsed.port:
                return f"{parsed.hostname}:{parsed.port}"
            return parsed.hostname
        return parsed.netloc
    trimmed = value.strip().rstrip("/")
    return trimmed.split("/", 1)[0]


def _list_language_model_names(client: Client) -> set[str] | None:
    try:
        models = client.models.list_language_models()
    except Exception:
        return None
    names: set[str] = set()
    for model in models:
        name = getattr(model, "name", "")
        if name:
            names.add(name)
        aliases = getattr(model, "aliases", ())
        for alias in aliases:
            if alias:
                names.add(alias)
    return names


def _choose_best_model_name(
    available_names: set[str] | None,
    *,
    workflow_type: StoryWorkflowType = "resolution_story",
) -> str:
    if not available_names:
        return DEFAULT_MULTI_AGENT_MODEL if workflow_type == "repricing_story" else DEFAULT_REASONING_MODEL
    if workflow_type == "repricing_story":
        for candidate in DEFAULT_MULTI_AGENT_MODEL_CANDIDATES:
            if candidate in available_names:
                return candidate
        grok4_multi_agent = sorted(name for name in available_names if "multi-agent" in name and name.startswith("grok-4"))
        if grok4_multi_agent:
            return grok4_multi_agent[0]
    for candidate in DEFAULT_REASONING_MODEL_CANDIDATES:
        if candidate in available_names:
            return candidate
    grok4_reasoning = sorted(
        name
        for name in available_names
        if name.startswith("grok-4") and "reasoning" in name and "non-reasoning" not in name and "multi-agent" not in name
    )
    if grok4_reasoning:
        return grok4_reasoning[0]
    grok4_general = sorted(
        name for name in available_names if name.startswith("grok-4") and "multi-agent" not in name
    )
    if grok4_general:
        return grok4_general[0]
    return DEFAULT_MULTI_AGENT_MODEL if workflow_type == "repricing_story" else DEFAULT_REASONING_MODEL


def _multi_agent_kwargs(model_name: str) -> dict[str, int]:
    if "multi-agent" in model_name:
        return {"agent_count": 4}
    return {}


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _truncate(value: str, limit: int = 2_000) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _clamp_float(value: float, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
