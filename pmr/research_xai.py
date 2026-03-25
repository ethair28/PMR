from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
from urllib import request

from pmr.models import EvidenceItem, ResearchJob, ResearchQueryPlan, ResearchResult


DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_MODEL = "grok-3-mini"


@dataclass(slots=True)
class XaiResearchSource:
    """xAI-backed research retriever that asks for X-first evidence plus web corroboration."""

    api_key: str
    model: str = DEFAULT_XAI_MODEL
    base_url: str = DEFAULT_XAI_BASE_URL
    timeout_seconds: int = 60
    max_evidence_items: int = 12

    @classmethod
    def from_env(cls) -> "XaiResearchSource":
        api_key = os.environ["XAI_API_KEY"]
        model = os.environ.get("PMR_XAI_MODEL", DEFAULT_XAI_MODEL)
        base_url = os.environ.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
        return cls(api_key=api_key, model=model, base_url=base_url)

    def search(self, job: ResearchJob, query_plan: ResearchQueryPlan) -> Sequence[EvidenceItem]:
        prompt = _build_source_prompt(job, query_plan, max_evidence_items=self.max_evidence_items)
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SOURCE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        response_payload = _post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        content = _extract_message_content(response_payload)
        parsed = json.loads(_strip_code_fences(content))
        collected_at = datetime.now(timezone.utc)
        evidence = []
        for item in parsed.get("evidence", ()):
            source_type = item.get("source_type")
            if source_type not in {"x_post", "web_article", "news_article"}:
                continue
            url = str(item.get("url", "")).strip()
            title_or_text = str(item.get("title_or_text", "")).strip()
            if not url or not title_or_text:
                continue
            evidence.append(
                EvidenceItem(
                    source_type=source_type,
                    url=url,
                    title_or_text=title_or_text,
                    author_or_publication=_none_if_blank(item.get("author_or_publication")),
                    published_at=_parse_optional_datetime(item.get("published_at")),
                    collected_at=collected_at,
                    relevance_score=_clamp_float(item.get("relevance_score"), default=0.5),
                    temporal_proximity_score=_clamp_float(item.get("temporal_proximity_score"), default=0.5),
                    stance=_validate_stance(item.get("stance")),
                    excerpt=_truncate(str(item.get("excerpt", "")).strip() or title_or_text),
                    query=_none_if_blank(item.get("query")),
                )
            )
        return tuple(evidence)


@dataclass(slots=True)
class XaiResearchSynthesizer:
    """xAI-backed synthesizer that turns normalized evidence into a structured result."""

    api_key: str
    model: str = DEFAULT_XAI_MODEL
    base_url: str = DEFAULT_XAI_BASE_URL
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "XaiResearchSynthesizer":
        api_key = os.environ["XAI_API_KEY"]
        model = os.environ.get("PMR_XAI_MODEL", DEFAULT_XAI_MODEL)
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
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_synthesis_prompt(
                        job=job,
                        query_plan=query_plan,
                        evidence=evidence,
                    ),
                },
            ],
        }
        response_payload = _post_json(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            api_key=self.api_key,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        content = _extract_message_content(response_payload)
        parsed = json.loads(_strip_code_fences(content))
        status = parsed.get("status", "failed")
        explanation_class = parsed.get("explanation_class")
        if explanation_class not in {"clear", "plausible", "speculative"}:
            explanation_class = None
        return ResearchResult(
            job_id=job.job_id,
            cache_key=cache_key,
            provider=provider_name,
            prompt_version=prompt_version,
            status=status if status in {"completed", "insufficient_evidence", "failed"} else "failed",
            explanation_class=explanation_class,
            confidence=_clamp_float(parsed.get("confidence"), default=0.0),
            most_plausible_explanation=str(parsed.get("most_plausible_explanation", "")).strip(),
            why_market_moved=str(parsed.get("why_market_moved", "")).strip(),
            key_evidence=tuple(evidence[:5]),
            contradictory_evidence=tuple(item for item in evidence if item.stance == "contradictory")[:3],
            open_questions=tuple(str(item).strip() for item in parsed.get("open_questions", ()) if str(item).strip()),
            completed_at=generated_at,
            error_message=_none_if_blank(parsed.get("error_message")),
        )


SOURCE_SYSTEM_PROMPT = """You are a research retriever for a prediction-market reporting system.

Search X first, then use web/news as corroboration. Return only JSON with this shape:
{
  "evidence": [
    {
      "source_type": "x_post" | "web_article" | "news_article",
      "url": "...",
      "title_or_text": "...",
      "author_or_publication": "...",
      "published_at": "ISO-8601 or null",
      "relevance_score": 0.0 to 1.0,
      "temporal_proximity_score": 0.0 to 1.0,
      "stance": "supporting" | "contradictory" | "contextual",
      "excerpt": "...",
      "query": "..."
    }
  ]
}

Do not invent URLs. If evidence is weak, return fewer items rather than guessing.
"""


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesizer for a prediction-market reporting system.

Return only JSON with this shape:
{
  "status": "completed" | "insufficient_evidence" | "failed",
  "explanation_class": "clear" | "plausible" | "speculative" | null,
  "confidence": 0.0 to 1.0,
  "most_plausible_explanation": "...",
  "why_market_moved": "...",
  "open_questions": ["..."],
  "error_message": null
}

If the evidence is weak or conflicting, use "insufficient_evidence" or "speculative" rather than overstating certainty.
"""


def _build_source_prompt(job: ResearchJob, query_plan: ResearchQueryPlan, *, max_evidence_items: int) -> str:
    return json.dumps(
        {
            "job_id": job.job_id,
            "story": {
                "family_label": job.family_label,
                "story_type_hint": job.story_type_hint,
                "editorial_priority_hint": job.editorial_priority_hint,
            },
            "investigation_question": job.investigation_question,
            "why_flagged": job.why_flagged,
            "primary_market": {
                "question": job.primary_market.question,
                "category": job.primary_market.category,
                "url": job.primary_market.url,
                "event_title": job.primary_market.event_title,
                "detection_window_start": job.primary_market.detection_window_start.isoformat(),
                "detection_window_end": job.primary_market.detection_window_end.isoformat(),
                "max_move_timestamp": (
                    job.primary_market.max_move_timestamp.isoformat()
                    if job.primary_market.max_move_timestamp
                    else None
                ),
            },
            "related_markets": [
                {"market_id": item.market_id, "question": item.question}
                for item in job.related_markets
            ],
            "query_plan": {
                "x_queries": list(query_plan.x_queries),
                "web_queries": list(query_plan.web_queries),
                "time_window_start": query_plan.time_window_start.isoformat(),
                "time_window_end": query_plan.time_window_end.isoformat(),
                "focus_timestamp": (
                    query_plan.focus_timestamp.isoformat() if query_plan.focus_timestamp else None
                ),
                "focus_points": list(query_plan.focus_points),
            },
            "max_evidence_items": max_evidence_items,
        }
    )


def _build_synthesis_prompt(
    *,
    job: ResearchJob,
    query_plan: ResearchQueryPlan,
    evidence: Sequence[EvidenceItem],
) -> str:
    return json.dumps(
        {
            "job_id": job.job_id,
            "story": {
                "family_label": job.family_label,
                "story_type_hint": job.story_type_hint,
            },
            "investigation_question": job.investigation_question,
            "why_flagged": job.why_flagged,
            "query_plan": {
                "time_window_start": query_plan.time_window_start.isoformat(),
                "time_window_end": query_plan.time_window_end.isoformat(),
                "focus_timestamp": (
                    query_plan.focus_timestamp.isoformat() if query_plan.focus_timestamp else None
                ),
            },
            "evidence": [
                {
                    "source_type": item.source_type,
                    "url": item.url,
                    "title_or_text": item.title_or_text,
                    "author_or_publication": item.author_or_publication,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "relevance_score": item.relevance_score,
                    "temporal_proximity_score": item.temporal_proximity_score,
                    "stance": item.stance,
                    "excerpt": item.excerpt,
                }
                for item in evidence
            ],
        }
    )


def _post_json(*, url: str, api_key: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", ())
    if not choices:
        raise ValueError("xAI response did not include any choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("xAI response did not include textual message content.")
    return content


def _strip_code_fences(value: str) -> str:
    trimmed = value.strip()
    if trimmed.startswith("```"):
        parts = trimmed.split("\n", 1)
        trimmed = parts[1] if len(parts) > 1 else trimmed
        if trimmed.endswith("```"):
            trimmed = trimmed[:-3]
    return trimmed.strip()


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(str(value))


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_stance(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"supporting", "contradictory", "contextual"}:
        return text
    return "contextual"


def _truncate(value: str, limit: int = 2_000) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
