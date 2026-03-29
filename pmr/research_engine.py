from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Protocol, Sequence

from pmr.models import (
    EvidenceItem,
    FollowUpQuery,
    InvestigationPlan,
    ResearchBatchResult,
    ResearchJob,
    ResearchQueryPlan,
    ResearchResult,
)
from pmr.research_store import ResearchStore


class ResearchSource(Protocol):
    def search(self, job: ResearchJob, query_plan: ResearchQueryPlan) -> Sequence[EvidenceItem]:
        """Collect normalized evidence for a research job."""

    def search_follow_up(
        self,
        job: ResearchJob,
        query_plan: ResearchQueryPlan,
        follow_up_queries: Sequence[FollowUpQuery],
    ) -> Sequence[EvidenceItem]:
        """Collect targeted follow-up evidence for a repricing job."""


class ResearchPlanner(Protocol):
    def plan(
        self,
        job: ResearchJob,
        query_plan: ResearchQueryPlan,
        evidence: Sequence[EvidenceItem],
        *,
        generated_at: datetime,
    ) -> InvestigationPlan:
        """Turn first-pass repricing evidence into a bounded investigation plan."""


class ResearchSynthesizer(Protocol):
    def summarize(
        self,
        job: ResearchJob,
        query_plan: ResearchQueryPlan,
        evidence: Sequence[EvidenceItem],
        *,
        investigation_plan: InvestigationPlan | None,
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
        investigation_plan: InvestigationPlan | None,
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
                story_role_hint=job.story_role_hint,
                status="insufficient_evidence",
                explanation_class="speculative",
                confidence=0.24,
                most_plausible_explanation="The live research pass did not surface enough evidence to explain the repricing confidently.",
                why_market_moved=job.why_flagged,
                price_action_summary=_build_price_action_summary(job),
                surprise_assessment=_build_surprise_assessment(job),
                main_narrative="The story remains underdeveloped because the current source mix did not produce enough timely evidence.",
                belief_shift_drivers=(),
                signal_types=(),
                why_now="The current search pass did not recover enough time-aligned evidence to explain why the move accelerated when it did.",
                alternative_explanations=(
                    "The move may reflect rumor propagation or low-visibility reporting not captured by the current search pass.",
                ),
                unresolved_points=(
                    "There is not yet enough evidence to isolate the main belief-shift driver.",
                ),
                note_to_editor="Evidence is too thin for a confident story. Treat this as a watchlist item unless a stronger second pass surfaces new reporting.",
                draft_headline=job.family_label,
                draft_markdown=_build_fallback_draft(
                    job,
                    headline=job.family_label,
                    narrative="Evidence was too thin to support a publication-ready draft.",
                ),
                overlap_group_key=job.overlap_group_key,
                overlap_summary=job.overlap_summary,
                suggested_merge_with=job.suggested_merge_with,
                key_evidence=(),
                contradictory_evidence=(),
                open_questions=(
                    "No strong X or web evidence was found near the move window.",
                    "The move may reflect rumors, thin context, or information not captured by the current source mix.",
                ),
                completed_at=generated_at,
                investigation_plan=investigation_plan,
            )
            return result

        top_support = supporting[:3]
        top_contradictory = contradictory[:2]
        avg_support = (
            sum(item.relevance_score + item.temporal_proximity_score for item in top_support) / len(top_support)
            if top_support
            else 0.0
        )
        avg_support_quality = (
            sum(_quality_confidence_weight(item.quality_tier) for item in top_support) / len(top_support)
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
        if top_support and avg_support_quality < 0.45:
            confidence = max(0.22, confidence - 0.16)
            if explanation_class == "clear":
                explanation_class = "plausible"

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
        belief_shift_drivers = _build_belief_shift_drivers(job, top_support)
        signal_types = _infer_signal_types(top_support)
        why_now = _build_why_now(job, top_support)
        unresolved_points = _build_unresolved_points(job, top_contradictory, open_questions)
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
            story_role_hint=job.story_role_hint,
            status=status,
            explanation_class=explanation_class,
            confidence=confidence,
            most_plausible_explanation=explanation,
            why_market_moved=job.why_flagged,
            price_action_summary=_build_price_action_summary(job),
            surprise_assessment=_build_surprise_assessment(job),
            main_narrative=main_narrative,
            belief_shift_drivers=belief_shift_drivers,
            signal_types=signal_types,
            why_now=why_now,
            alternative_explanations=tuple(
                "Contradictory or delayed signals could still change the story."
                for _ in top_contradictory[:1]
            ),
            unresolved_points=unresolved_points,
            note_to_editor=note_to_editor,
            draft_headline=headline,
            draft_markdown=_build_fallback_draft(job, headline=headline, narrative=main_narrative),
            overlap_group_key=job.overlap_group_key,
            overlap_summary=job.overlap_summary,
            suggested_merge_with=job.suggested_merge_with,
            key_evidence=top_support,
            contradictory_evidence=top_contradictory,
            open_questions=tuple(open_questions),
            completed_at=generated_at,
            investigation_plan=investigation_plan,
        )
        return result


@dataclass(slots=True)
class ResearchEngine:
    """Coordinate query planning, evidence collection, caching, and story drafting."""

    source: ResearchSource
    synthesizer: ResearchSynthesizer
    provider_name: str
    prompt_version: str
    planner: ResearchPlanner | None = None
    store: ResearchStore | None = None
    max_evidence_items: int = 50
    max_follow_up_queries: int = 5

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
            investigation_plan: InvestigationPlan | None = None
            if (
                job.workflow_type == "repricing_story"
                and self.planner is not None
                and ranked_evidence
            ):
                try:
                    investigation_plan = self.planner.plan(
                        job,
                        query_plan,
                        ranked_evidence,
                        generated_at=now,
                    )
                    follow_up_queries = select_follow_up_queries(
                        investigation_plan,
                        max_queries=self.max_follow_up_queries,
                    )
                    if investigation_plan.needs_more_research and follow_up_queries:
                        follow_up_evidence = self.source.search_follow_up(
                            job,
                            query_plan,
                            follow_up_queries,
                        )
                        ranked_evidence = rank_evidence_for_job(
                            job=job,
                            evidence=tuple(ranked_evidence) + tuple(follow_up_evidence),
                            max_items=self.max_evidence_items,
                        )
                except Exception:
                    investigation_plan = None
            result = self.synthesizer.summarize(
                job,
                query_plan,
                ranked_evidence,
                investigation_plan=investigation_plan,
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
                story_role_hint=job.story_role_hint,
                status="failed",
                explanation_class=None,
                confidence=0.0,
                most_plausible_explanation="",
                why_market_moved=job.why_flagged,
                price_action_summary=_build_price_action_summary(job),
                surprise_assessment=_build_surprise_assessment(job),
                main_narrative="",
                belief_shift_drivers=(),
                signal_types=(),
                why_now="The story-development run failed before the move-timing explanation could be synthesized.",
                alternative_explanations=(),
                unresolved_points=("The story-development run failed before unresolved points could be summarized.",),
                note_to_editor="The story-development run failed before synthesis completed.",
                draft_headline=job.family_label,
                draft_markdown="",
                overlap_group_key=job.overlap_group_key,
                overlap_summary=job.overlap_summary,
                suggested_merge_with=job.suggested_merge_with,
                key_evidence=(),
                contradictory_evidence=(),
                open_questions=("The research run failed before completing synthesis.",),
                completed_at=now,
                investigation_plan=None,
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


def select_follow_up_queries(plan: InvestigationPlan, *, max_queries: int) -> tuple[FollowUpQuery, ...]:
    """Bound and dedupe dynamic follow-up queries while preserving one skeptical check."""

    if max_queries <= 0:
        return ()

    selected: list[FollowUpQuery] = []
    seen: set[tuple[str, str]] = set()
    skeptical_query = plan.skeptical_query
    skeptical_key = None
    if skeptical_query is not None:
        skeptical_key = (skeptical_query.source_type, _clean_query(skeptical_query.query).lower())

    for item in plan.follow_up_queries:
        normalized_query = _clean_query(item.query)
        if not normalized_query:
            continue
        key = (item.source_type, normalized_query.lower())
        if key in seen or key == skeptical_key:
            continue
        selected.append(
            FollowUpQuery(
                query=normalized_query,
                source_type=item.source_type,
                reason=item.reason.strip(),
                skeptical=item.skeptical,
            )
        )
        seen.add(key)

    if skeptical_query is not None and skeptical_key not in seen:
        skeptical_query = FollowUpQuery(
            query=_clean_query(skeptical_query.query),
            source_type=skeptical_query.source_type,
            reason=skeptical_query.reason.strip(),
            skeptical=True,
        )

    if skeptical_query is not None:
        positive_limit = max(max_queries - 1, 0)
        trimmed = selected[:positive_limit]
        if skeptical_query.query:
            trimmed.append(skeptical_query)
        return tuple(trimmed[:max_queries])
    return tuple(selected[:max_queries])


def rank_evidence_for_job(
    *,
    job: ResearchJob,
    evidence: Sequence[EvidenceItem],
    max_items: int,
) -> tuple[EvidenceItem, ...]:
    """Dedupe and rank evidence using temporal proximity, source quality, and workflow needs."""

    focus_timestamp = job.primary_market.max_move_timestamp or job.primary_market.detection_window_end
    deduped: dict[str, EvidenceItem] = {}
    for item in evidence:
        if _is_disallowed_recursive_evidence(item):
            continue
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
    if job.workflow_type == "repricing_story":
        contradictory = [item for item in ranked if item.stance == "contradictory"]
        non_contradictory = [item for item in ranked if item.stance != "contradictory"]
        if contradictory:
            selected: list[EvidenceItem] = [non_contradictory[0]] if non_contradictory else []
            selected.append(contradictory[0])
            for item in ranked:
                if item in selected:
                    continue
                selected.append(item)
                if len(selected) >= max_items:
                    break
            return tuple(selected[:max_items])
    return tuple(ranked[:max_items])


def _normalize_evidence_item(item: EvidenceItem, focus_timestamp: datetime) -> EvidenceItem:
    proximity = item.temporal_proximity_score
    if item.published_at is not None:
        proximity = max(proximity, _temporal_proximity_score(item.published_at, focus_timestamp))
    normalized_item = EvidenceItem(
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
        quality_tier=item.quality_tier,
    )
    quality_tier = _evidence_quality_tier(normalized_item)
    return EvidenceItem(
        source_type=normalized_item.source_type,
        url=normalized_item.url,
        title_or_text=normalized_item.title_or_text,
        author_or_publication=normalized_item.author_or_publication,
        published_at=normalized_item.published_at,
        collected_at=normalized_item.collected_at,
        relevance_score=normalized_item.relevance_score,
        temporal_proximity_score=normalized_item.temporal_proximity_score,
        stance=normalized_item.stance,
        excerpt=normalized_item.excerpt,
        query=normalized_item.query,
        quality_tier=quality_tier,
    )


def _combined_evidence_score(item: EvidenceItem) -> float:
    stance_bonus = 0.05 if item.stance == "supporting" else 0.0
    return (
        item.relevance_score
        + item.temporal_proximity_score
        + stance_bonus
        + _source_quality_adjustment(item)
        + _quality_tier_adjustment(item.quality_tier)
    )


def _temporal_proximity_score(observed_at: datetime, focus_timestamp: datetime) -> float:
    hours = abs((observed_at - focus_timestamp).total_seconds()) / 3600
    return 1.0 / (1.0 + hours / 24.0)


def _clean_query(value: str) -> str:
    return " ".join(value.split())


def _source_quality_adjustment(item: EvidenceItem) -> float:
    url = item.url.lower()
    title = item.title_or_text.lower()
    author = (item.author_or_publication or "").lower()
    adjustment = 0.0
    if "wikipedia.org" in url:
        adjustment -= 0.40
    if item.source_type == "x_post":
        adjustment -= 0.08
        if any(token in author for token in _MARKET_COMMENTARY_HANDLES) or any(
            token in title for token in _MARKET_COMMENTARY_TERMS
        ):
            adjustment -= 0.18
        if any(token in author for token in _NEWSY_X_ACCOUNTS):
            adjustment += 0.06
    if any(domain in url for domain in _HIGH_SIGNAL_DOMAINS):
        adjustment += 0.12
    if "truthsocial.com" in url or "whitehouse.gov" in url:
        adjustment += 0.18
    return adjustment


def _evidence_quality_tier(item: EvidenceItem) -> str:
    url = item.url.lower()
    title = item.title_or_text.lower()
    author = (item.author_or_publication or "").lower()
    if any(domain in url for domain in _PRIMARY_SIGNAL_DOMAINS):
        return "primary"
    if "whitehouse.gov" in url or "truthsocial.com" in url:
        return "primary"
    if item.source_type == "x_post":
        if _is_recursive_model_commentary(item):
            return "weak_context"
        if any(token in author for token in _NEWSY_X_ACCOUNTS):
            return "secondary"
        if any(token in title for token in _MARKET_COMMENTARY_TERMS) or any(
            token in author for token in _MARKET_COMMENTARY_HANDLES
        ):
            return "weak_context"
        if any(token in title for token in {"rumor", "reportedly", "possible", "talks"}):
            return "secondary"
        return "secondary"
    if "wikipedia.org" in url:
        return "weak_context"
    if item.source_type in {"news_article", "web_article"}:
        if any(token in url for token in _LOW_SIGNAL_CONTEXT_DOMAINS):
            return "weak_context"
        return "secondary"
    return "secondary"


def _quality_tier_adjustment(tier: str) -> float:
    if tier == "primary":
        return 0.16
    if tier == "secondary":
        return 0.04
    return -0.16


def _quality_confidence_weight(tier: str) -> float:
    if tier == "primary":
        return 1.0
    if tier == "secondary":
        return 0.6
    return 0.2


def _is_recursive_model_commentary(item: EvidenceItem) -> bool:
    url = item.url.lower()
    author = (item.author_or_publication or "").lower()
    return "x.com/grok/" in url or author == "grok"


def _is_disallowed_recursive_evidence(item: EvidenceItem) -> bool:
    return _is_recursive_model_commentary(item)


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


def _build_belief_shift_drivers(
    job: ResearchJob,
    evidence: Sequence[EvidenceItem],
) -> tuple[str, ...]:
    if job.workflow_type != "repricing_story":
        return ()
    drivers: list[str] = []
    for item in evidence[:3]:
        text = item.title_or_text.strip()
        if text and text not in drivers:
            drivers.append(text)
    if not drivers:
        drivers.append("No single dominant driver was recoverable from the current evidence mix.")
    return tuple(drivers)


def _infer_signal_types(evidence: Sequence[EvidenceItem]) -> tuple[str, ...]:
    signal_types: list[str] = []
    for item in evidence:
        signal = _signal_type_for_evidence(item)
        if signal not in signal_types:
            signal_types.append(signal)
    return tuple(signal_types)


def _signal_type_for_evidence(item: EvidenceItem) -> str:
    url = item.url.lower()
    title = item.title_or_text.lower()
    author = (item.author_or_publication or "").lower()
    if "whitehouse.gov" in url or "truthsocial.com" in url or "official" in author:
        return "official_statement"
    if item.source_type in {"news_article", "web_article"}:
        return "reporting"
    if item.source_type == "x_post":
        if any(token in title for token in _MARKET_COMMENTARY_TERMS) or any(
            token in author for token in _MARKET_COMMENTARY_HANDLES
        ):
            return "market_commentary"
        if any(token in title for token in {"rumor", "reportedly", "possible", "talks"}):
            return "rumor_or_speculation"
        return "social_commentary"
    return "context"


def _build_why_now(job: ResearchJob, evidence: Sequence[EvidenceItem]) -> str:
    focus = job.primary_market.max_move_timestamp
    if focus is None:
        return "The move appears to have built over the week rather than around one clearly isolated timestamp."
    if not evidence:
        return (
            f"The largest move clustered around {focus.isoformat()}, but the current evidence mix does not yet "
            "identify one clean trigger for that timing."
        )
    lead = evidence[0]
    published = lead.published_at.isoformat() if lead.published_at else "the move window"
    return (
        f"The odds moved most sharply around {focus.isoformat()}, and the strongest supporting evidence surfaced "
        f"around {published}, which is why this story looks tied to that specific window rather than to a slow weekly drift."
    )


def _build_unresolved_points(
    job: ResearchJob,
    contradictory: Sequence[EvidenceItem],
    open_questions: Sequence[str],
) -> tuple[str, ...]:
    points = list(open_questions[:2])
    if job.workflow_type == "repricing_story" and not points:
        points.append("The move remains unresolved, so later information could still reverse or validate the current narrative.")
    if contradictory and not any("contradict" in point.lower() for point in points):
        points.append("Some contradictory evidence remains in the source mix, so confidence should stay bounded.")
    return tuple(points)


_MARKET_COMMENTARY_TERMS = {
    "polymarket",
    "odds",
    "volume",
    "traders",
    "cents",
    "market",
}

_MARKET_COMMENTARY_HANDLES = {
    "polynews",
    "agentonchain",
    "whalemovers",
    "polyworm",
    "0xzx",
}

_NEWSY_X_ACCOUNTS = {
    "reuters",
    "politico",
    "france 24",
    "euronews",
    "news",
    "company",
}

_HIGH_SIGNAL_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "politico.eu",
    "apnews.com",
    "upi.com",
    "euronews.com",
    "france24.com",
    "aljazeera.com",
    "news.un.org",
    "whitehouse.gov",
}

_PRIMARY_SIGNAL_DOMAINS = _HIGH_SIGNAL_DOMAINS | {
    "ft.com",
    "wsj.com",
    "nytimes.com",
    "washingtonpost.com",
}

_LOW_SIGNAL_CONTEXT_DOMAINS = {
    "wikipedia.org",
    "investopedia.com",
}
