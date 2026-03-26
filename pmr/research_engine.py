from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Protocol, Sequence

from pmr.models import EvidenceItem, ResearchBatchResult, ResearchJob, ResearchQueryPlan, ResearchResult
from pmr.research_store import ResearchStore


class ResearchSource(Protocol):
    def search(self, job: ResearchJob, query_plan: ResearchQueryPlan) -> Sequence[EvidenceItem]:
        """Collect normalized evidence for a research job."""


class ResearchSynthesizer(Protocol):
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
        """Turn normalized evidence into one structured story-development output."""


@dataclass(slots=True)
class HeuristicResearchSynthesizer:
    """Deterministic story drafter used for tests and fallback paths."""

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
        supporting = tuple(item for item in evidence if item.stance != "contradictory")
        contradictory = tuple(item for item in evidence if item.stance == "contradictory")
        if not evidence:
            result = ResearchResult(
                job_id=job.job_id,
                cache_key=cache_key,
                provider=provider_name,
                prompt_version=prompt_version,
                model_name="heuristic",
                workflow_type=job.workflow_type,
                status="insufficient_evidence",
                explanation_class="speculative",
                confidence=0.24,
                most_plausible_explanation="The live research pass did not surface enough evidence to explain the repricing confidently.",
                why_market_moved=job.why_flagged,
                price_action_summary=_build_price_action_summary(job),
                surprise_assessment=_build_surprise_assessment(job),
                main_narrative="The story remains underdeveloped because the current source mix did not produce enough timely evidence.",
                alternative_explanations=(
                    "The move may reflect rumor propagation or low-visibility reporting not captured by the current search pass.",
                ),
                note_to_editor="Evidence is too thin for a confident story. Treat this as a watchlist item unless a stronger second pass surfaces new reporting.",
                draft_headline=job.family_label,
                draft_markdown=_build_fallback_draft(
                    job,
                    headline=job.family_label,
                    narrative="Evidence was too thin to support a publication-ready draft.",
                ),
                key_evidence=(),
                contradictory_evidence=(),
                open_questions=(
                    "No strong X or web evidence was found near the move window.",
                    "The move may reflect rumors, thin context, or information not captured by the current source mix.",
                ),
                completed_at=generated_at,
            )
            return result

        top_support = supporting[:3]
        top_contradictory = contradictory[:2]
        avg_support = (
            sum(item.relevance_score + item.temporal_proximity_score for item in top_support) / len(top_support)
            if top_support
            else 0.0
        )
        avg_contradictory = (
            sum(item.relevance_score + item.temporal_proximity_score for item in top_contradictory)
            / len(top_contradictory)
            if top_contradictory
            else 0.0
        )
        if len(top_support) >= 2 and avg_support >= 1.35 and avg_contradictory < 1.0:
            status = "completed"
            explanation_class = "clear"
            confidence = 0.74
        elif top_support:
            status = "completed"
            explanation_class = "plausible"
            confidence = 0.57 if avg_contradictory < avg_support else 0.44
        else:
            status = "insufficient_evidence"
            explanation_class = "speculative"
            confidence = 0.28

        lead = top_support[0].title_or_text if top_support else "available evidence is thin"
        explanation = (
            f"The repricing most likely reflects a shift captured in evidence such as '{lead}', "
            "with the strongest items clustered near the move window."
        )
        if top_contradictory:
            explanation += " Some contradictory signals remain, so confidence should stay bounded."
        open_questions = []
        if status != "completed":
            open_questions.append("More evidence is needed before treating this as a settled explanation.")
        if top_contradictory:
            open_questions.append("Contradictory items suggest the story may still be developing.")
        headline = _build_heuristic_headline(job)
        main_narrative = explanation
        note_to_editor = (
            "This is a fallback heuristic draft. Use it for testing and plumbing only, not as a publication-grade story."
        )
        result = ResearchResult(
            job_id=job.job_id,
            cache_key=cache_key,
            provider=provider_name,
            prompt_version=prompt_version,
            model_name="heuristic",
            workflow_type=job.workflow_type,
            status=status,
            explanation_class=explanation_class,
            confidence=confidence,
            most_plausible_explanation=explanation,
            why_market_moved=job.why_flagged,
            price_action_summary=_build_price_action_summary(job),
            surprise_assessment=_build_surprise_assessment(job),
            main_narrative=main_narrative,
            alternative_explanations=tuple(
                "Contradictory or delayed signals could still change the story."
                for _ in top_contradictory[:1]
            ),
            note_to_editor=note_to_editor,
            draft_headline=headline,
            draft_markdown=_build_fallback_draft(job, headline=headline, narrative=main_narrative),
            key_evidence=top_support,
            contradictory_evidence=top_contradictory,
            open_questions=tuple(open_questions),
            completed_at=generated_at,
        )
        return result


@dataclass(slots=True)
class ResearchEngine:
    """Coordinate query planning, evidence collection, caching, and story drafting."""

    source: ResearchSource
    synthesizer: ResearchSynthesizer
    provider_name: str
    prompt_version: str
    store: ResearchStore | None = None
    max_evidence_items: int = 50

    def run_batch(
        self,
        jobs: Sequence[ResearchJob],
        *,
        refresh: bool = False,
        max_jobs: int | None = None,
        now: datetime | None = None,
    ) -> ResearchBatchResult:
        now = now or datetime.now(timezone.utc)
        if self.store is not None:
            self.store.initialize()
            self.store.prune(now=now)
        selected_jobs = tuple(jobs[:max_jobs] if max_jobs is not None else jobs)
        results = tuple(self.investigate_job(job, refresh=refresh, now=now) for job in selected_jobs)
        if self.store is not None:
            self.store.prune(now=now)
        return ResearchBatchResult(
            provider=self.provider_name,
            prompt_version=self.prompt_version,
            generated_at=now,
            results=results,
        )

    def investigate_job(
        self,
        job: ResearchJob,
        *,
        refresh: bool = False,
        now: datetime | None = None,
    ) -> ResearchResult:
        now = now or datetime.now(timezone.utc)
        cache_key = build_research_cache_key(
            job=job,
            provider_name=self.provider_name,
            prompt_version=self.prompt_version,
        )
        if self.store is not None and not refresh:
            cached = self.store.get_cached_result(cache_key)
            if cached is not None:
                return cached

        query_plan = build_research_query_plan(job)
        try:
            evidence = self.source.search(job, query_plan)
            ranked_evidence = rank_evidence_for_job(
                job=job,
                evidence=evidence,
                max_items=self.max_evidence_items,
            )
            result = self.synthesizer.summarize(
                job,
                query_plan,
                ranked_evidence,
                cache_key=cache_key,
                provider_name=self.provider_name,
                prompt_version=self.prompt_version,
                generated_at=now,
            )
        except Exception as exc:
            result = ResearchResult(
                job_id=job.job_id,
                cache_key=cache_key,
                provider=self.provider_name,
                prompt_version=self.prompt_version,
                model_name="unavailable",
                workflow_type=job.workflow_type,
                status="failed",
                explanation_class=None,
                confidence=0.0,
                most_plausible_explanation="",
                why_market_moved=job.why_flagged,
                price_action_summary=_build_price_action_summary(job),
                surprise_assessment=_build_surprise_assessment(job),
                main_narrative="",
                alternative_explanations=(),
                note_to_editor="The story-development run failed before synthesis completed.",
                draft_headline=job.family_label,
                draft_markdown="",
                key_evidence=(),
                contradictory_evidence=(),
                open_questions=("The research run failed before completing synthesis.",),
                completed_at=now,
                error_message=str(exc),
            )
        if self.store is not None:
            self.store.upsert_result(job, result)
        return result


def build_research_query_plan(job: ResearchJob) -> ResearchQueryPlan:
    """Build a deterministic X-first search plan for one research job."""

    focus_timestamp = job.primary_market.max_move_timestamp or job.primary_market.detection_window_end
    family_term = job.family_label
    primary_question = job.primary_market.question
    related_terms = [item.question for item in job.related_markets[:2]]
    x_queries = [
        primary_question,
        family_term,
        f"{family_term} {job.primary_market.category}",
    ]
    if job.primary_market.event_title:
        x_queries.append(job.primary_market.event_title)
    x_queries.extend(related_terms)
    if job.workflow_type == "resolution_story":
        x_queries.extend(
            [
                f"{family_term} results",
                f"{family_term} exit polls",
            ]
        )
    else:
        x_queries.extend(
            [
                f"{family_term} rumor",
                f"{family_term} talks",
            ]
        )
    web_queries = [
        family_term,
        primary_question,
        f"{family_term} news",
    ]
    if job.primary_market.event_title:
        web_queries.append(f"{job.primary_market.event_title} analysis")
    if job.workflow_type == "resolution_story":
        web_queries.append(f"{family_term} results")
    else:
        web_queries.append(f"{family_term} rumors")
    time_window_start = max(
        job.primary_market.detection_window_start - timedelta(days=1),
        focus_timestamp - timedelta(days=3),
    )
    time_window_end = job.primary_market.detection_window_end + timedelta(hours=12)
    return ResearchQueryPlan(
        job_id=job.job_id,
        x_queries=tuple(dict.fromkeys(_clean_query(item) for item in x_queries if item)),
        web_queries=tuple(dict.fromkeys(_clean_query(item) for item in web_queries if item)),
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        focus_timestamp=job.primary_market.max_move_timestamp,
        focus_points=job.focus_points,
    )


def build_research_cache_key(job: ResearchJob, *, provider_name: str, prompt_version: str) -> str:
    """Build the bounded cache key for one job/provider/prompt/window tuple."""

    raw = "|".join(
        (
            job.job_id,
            job.primary_market.detection_window_end.isoformat(),
            provider_name,
            prompt_version,
        )
    )
    digest = sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{job.job_id}:{prompt_version}:{digest}"


def rank_evidence_for_job(
    *,
    job: ResearchJob,
    evidence: Sequence[EvidenceItem],
    max_items: int,
) -> tuple[EvidenceItem, ...]:
    """Dedupe and rank evidence using temporal proximity and source relevance."""

    focus_timestamp = job.primary_market.max_move_timestamp or job.primary_market.detection_window_end
    deduped: dict[str, EvidenceItem] = {}
    for item in evidence:
        normalized = _normalize_evidence_item(item, focus_timestamp)
        key = normalized.url.strip().lower().rstrip("/") or normalized.title_or_text.strip().lower()
        current = deduped.get(key)
        if current is None or _combined_evidence_score(normalized) > _combined_evidence_score(current):
            deduped[key] = normalized
    ranked = sorted(
        deduped.values(),
        key=lambda item: (
            _combined_evidence_score(item),
            item.published_at.isoformat() if item.published_at else "",
            item.url,
        ),
        reverse=True,
    )
    return tuple(ranked[:max_items])


def _normalize_evidence_item(item: EvidenceItem, focus_timestamp: datetime) -> EvidenceItem:
    proximity = item.temporal_proximity_score
    if item.published_at is not None:
        proximity = max(proximity, _temporal_proximity_score(item.published_at, focus_timestamp))
    return EvidenceItem(
        source_type=item.source_type,
        url=item.url,
        title_or_text=item.title_or_text.strip(),
        author_or_publication=item.author_or_publication.strip() if item.author_or_publication else None,
        published_at=item.published_at,
        collected_at=item.collected_at,
        relevance_score=max(0.0, min(1.0, item.relevance_score)),
        temporal_proximity_score=max(0.0, min(1.0, proximity)),
        stance=item.stance,
        excerpt=item.excerpt.strip(),
        query=item.query.strip() if item.query else None,
    )


def _combined_evidence_score(item: EvidenceItem) -> float:
    stance_bonus = 0.05 if item.stance == "supporting" else 0.0
    return item.relevance_score + item.temporal_proximity_score + stance_bonus


def _temporal_proximity_score(observed_at: datetime, focus_timestamp: datetime) -> float:
    hours = abs((observed_at - focus_timestamp).total_seconds()) / 3600
    return 1.0 / (1.0 + hours / 24.0)


def _clean_query(value: str) -> str:
    return " ".join(value.split())


def _build_price_action_summary(job: ResearchJob) -> str:
    market = job.primary_market
    return (
        f"Over the 7-day detection window, the market moved from {market.window_open_probability * 100:.1f}% "
        f"to {market.window_close_probability * 100:.1f}%, with a weekly range of {market.weekly_range * 100:.1f} "
        f"points and a max 24h move of {market.max_abs_move_24h * 100:.1f} points."
    )


def _build_surprise_assessment(job: ResearchJob) -> str:
    context = job.primary_market.price_context
    if job.workflow_type == "repricing_story":
        return (
            "This looks more like a broad repricing story than a resolved outcome. The key question is what changed "
            "market perception enough to move the odds materially without fully settling the event."
        )
    if context.surprise_points is None:
        return "The move looks resolution-driven, but the surprise level could not be quantified cleanly from the current packet."
    return (
        f"This looks like a {context.surprise_label or 'measurable'} resolution surprise. "
        f"Using the window-open probability as the pre-resolution baseline, the outcome surprised the market by "
        f"about {context.surprise_points:.1f} percentage points."
    )


def _build_heuristic_headline(job: ResearchJob) -> str:
    if job.workflow_type == "repricing_story":
        return f"{job.family_label}: Odds Repriced Sharply Inside One Week"
    return f"{job.family_label}: Market Resolution and Surprise Assessment"


def _build_fallback_draft(job: ResearchJob, *, headline: str, narrative: str) -> str:
    return "\n".join(
        (
            f"# {headline}",
            "",
            _build_price_action_summary(job),
            "",
            narrative,
            "",
            _build_surprise_assessment(job),
        )
    )
