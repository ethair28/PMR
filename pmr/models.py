from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ExplanationType = Literal["clear", "plausible", "speculative"]
HistoryMode = Literal["full_history", "short_history", "insufficient_data"]
ConfidenceLevel = Literal["high", "medium", "low"]
StoryTypeHint = Literal["live_repricing", "resolved_surprise", "late_stage_resolution"]
StoryWorkflowType = Literal["resolution_story", "repricing_story"]
StoryRoleHint = Literal["primary", "secondary", "standalone"]
EditorialPriority = Literal["high", "medium", "low"]
EvidenceSourceType = Literal["x_post", "web_article", "news_article"]
EvidenceStance = Literal["supporting", "contradictory", "contextual"]
ResearchStatus = Literal["completed", "insufficient_evidence", "failed"]
EditorDecisionAction = Literal["include", "merge", "exclude"]
EditorialDetailLevel = Literal["brief", "standard", "extended", "lead"]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """A single probability observation for a binary market."""

    observed_at: datetime
    probability: float


@dataclass(frozen=True, slots=True)
class Market:
    """Market metadata and liquidity fields used by the detector."""

    market_id: str
    question: str
    category: str
    tags: tuple[str, ...] = ()
    slug: str | None = None
    url: str | None = None
    description: str | None = None
    condition_id: str | None = None
    tracked_outcome: str | None = None
    tracked_token_id: str | None = None
    event_title: str | None = None
    volume_7d_usd: float = 0.0
    volume_24h_usd: float = 0.0
    open_interest_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketSeries:
    """A market plus its observed probability history."""

    market: Market
    snapshots: tuple[MarketSnapshot, ...]
    research_hints: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineStats:
    """Pre-window normalization statistics for a market."""

    usable_history_days: float
    preferred_window_days: int
    snapshot_count: int
    median_abs_move_6h: float | None
    median_abs_move_24h: float | None
    median_weekly_range: float | None
    sample_count_6h: int
    sample_count_24h: int
    sample_count_weekly_range: int


@dataclass(frozen=True, slots=True)
class LiquidityMetrics:
    """Liquidity data surfaced alongside anomaly features."""

    volume_7d_usd: float
    volume_24h_usd: float
    open_interest_usd: float
    liquidity_score: float


@dataclass(frozen=True, slots=True)
class RepricingEvent:
    """Explainable weekly anomaly candidate for downstream research/reporting."""

    market: Market
    detection_window_start: datetime
    detection_window_end: datetime
    history_mode: HistoryMode
    confidence_level: ConfidenceLevel
    confidence_score: float
    eligible_for_ranking: bool
    exclusion_reason: str | None
    window_open_probability: float
    window_close_probability: float
    window_high_probability: float
    window_low_probability: float
    largest_6h_move: float
    max_abs_move_6h: float
    largest_24h_move: float
    max_abs_move_24h: float
    weekly_range: float
    close_to_open_move: float
    persistence_of_largest_move: float
    jump_count_over_threshold: int
    max_move_timestamp: datetime | None
    story_group_key: str
    story_group_label: str
    story_type_hint: StoryTypeHint
    distance_from_extremes: float
    entered_extreme_zone: bool
    related_market_ids: tuple[str, ...]
    related_market_questions: tuple[str, ...]
    baseline_stats: BaselineStats
    liquidity_metrics: LiquidityMetrics
    z_6h: float
    z_24h: float
    z_range: float
    composite_score: float
    notes: tuple[str, ...]
    series: MarketSeries = field(repr=False)

    @property
    def move(self) -> float:
        return self.close_to_open_move

    @property
    def abs_move(self) -> float:
        return abs(self.close_to_open_move)

    @property
    def significance_score(self) -> float:
        return self.composite_score


@dataclass(frozen=True, slots=True)
class RelatedMarket:
    """A secondary market that provides context for a research job."""

    market_id: str
    question: str


@dataclass(frozen=True, slots=True)
class PriceTracePoint:
    """A compact price observation used by the story-development stage."""

    observed_at: datetime
    probability: float
    move_since_previous: float | None = None


@dataclass(frozen=True, slots=True)
class ResearchPriceContext:
    """A concise price-action packet for one research/story job."""

    interval_hours: int
    trace_points: tuple[PriceTracePoint, ...]
    largest_move_window_hours: int
    largest_move_window_start: datetime | None
    largest_move_window_end: datetime | None
    surprise_reference_probability: float | None = None
    surprise_points: float | None = None
    surprise_label: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchMarketContext:
    """Typed primary-market context for a research job."""

    market_id: str
    question: str
    detection_window_start: datetime
    detection_window_end: datetime
    history_mode: HistoryMode
    confidence_level: ConfidenceLevel
    confidence_score: float
    composite_score: float
    window_open_probability: float
    window_close_probability: float
    window_high_probability: float
    window_low_probability: float
    close_to_open_move: float
    max_abs_move_6h: float
    max_abs_move_24h: float
    largest_6h_move: float
    largest_24h_move: float
    weekly_range: float
    persistence_of_largest_move: float
    jump_count_over_threshold: int
    max_move_timestamp: datetime | None
    category: str
    price_context: ResearchPriceContext
    slug: str | None = None
    url: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    condition_id: str | None = None
    tracked_outcome: str | None = None
    tracked_token_id: str | None = None
    event_title: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchJob:
    """A typed research brief derived from one detected anomaly candidate."""

    job_id: str
    family_key: str
    family_label: str
    workflow_type: StoryWorkflowType
    story_type_hint: StoryTypeHint
    distance_from_extremes: float
    entered_extreme_zone: bool
    editorial_priority_hint: EditorialPriority
    story_role_hint: StoryRoleHint
    investigation_question: str
    why_flagged: str
    focus_points: tuple[str, ...]
    primary_market: ResearchMarketContext
    overlap_group_key: str | None = None
    overlap_summary: str | None = None
    suggested_merge_with: tuple[str, ...] = ()
    related_markets: tuple[RelatedMarket, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchQueryPlan:
    """Search-plan inputs for one research job."""

    job_id: str
    x_queries: tuple[str, ...]
    web_queries: tuple[str, ...]
    time_window_start: datetime
    time_window_end: datetime
    focus_timestamp: datetime | None
    focus_points: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Normalized evidence collected during the research step."""

    source_type: EvidenceSourceType
    url: str
    title_or_text: str
    author_or_publication: str | None
    published_at: datetime | None
    collected_at: datetime
    relevance_score: float
    temporal_proximity_score: float
    stance: EvidenceStance
    excerpt: str
    query: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Structured story-development output produced by the research layer."""

    job_id: str
    cache_key: str
    provider: str
    prompt_version: str
    model_name: str
    workflow_type: StoryWorkflowType
    story_role_hint: StoryRoleHint
    status: ResearchStatus
    explanation_class: ExplanationType | None
    confidence: float
    most_plausible_explanation: str
    why_market_moved: str
    price_action_summary: str
    surprise_assessment: str
    main_narrative: str
    alternative_explanations: tuple[str, ...]
    note_to_editor: str
    draft_headline: str
    draft_markdown: str
    overlap_group_key: str | None
    overlap_summary: str | None
    suggested_merge_with: tuple[str, ...]
    key_evidence: tuple[EvidenceItem, ...]
    contradictory_evidence: tuple[EvidenceItem, ...]
    open_questions: tuple[str, ...]
    completed_at: datetime
    error_message: str | None = None
    used_cache: bool = False


@dataclass(frozen=True, slots=True)
class ResearchBatchResult:
    """Batch output for a run of the research layer."""

    provider: str
    prompt_version: str
    generated_at: datetime
    results: tuple[ResearchResult, ...]

    @property
    def processed_jobs(self) -> int:
        return len(self.results)

    @property
    def cached_jobs(self) -> int:
        return sum(1 for result in self.results if result.used_cache)

    @property
    def failed_jobs(self) -> int:
        return sum(1 for result in self.results if result.status == "failed")


@dataclass(frozen=True, slots=True)
class EditorStoryPacket:
    """Merged editor-stage input with story draft plus rich market context."""

    job_id: str
    family_key: str
    family_label: str
    workflow_type: StoryWorkflowType
    story_type_hint: StoryTypeHint
    editorial_priority_hint: EditorialPriority
    story_role_hint: StoryRoleHint
    investigation_question: str
    why_flagged: str
    focus_points: tuple[str, ...]
    primary_market: ResearchMarketContext
    overlap_group_key: str | None
    overlap_summary: str | None
    root_cluster_key: str | None
    root_cluster_summary: str | None
    suggested_merge_with: tuple[str, ...]
    related_markets: tuple[RelatedMarket, ...]
    status: ResearchStatus
    explanation_class: ExplanationType | None
    confidence: float
    model_name: str
    most_plausible_explanation: str
    why_market_moved: str
    price_action_summary: str
    surprise_assessment: str
    main_narrative: str
    alternative_explanations: tuple[str, ...]
    note_to_editor: str
    draft_headline: str
    draft_markdown: str
    key_evidence: tuple[EvidenceItem, ...]
    contradictory_evidence: tuple[EvidenceItem, ...]
    open_questions: tuple[str, ...]
    completed_at: datetime
    error_message: str | None = None
    used_cache: bool = False


@dataclass(frozen=True, slots=True)
class EditorDecision:
    """One editorial keep/merge/cut decision for a story packet."""

    job_id: str
    action: EditorDecisionAction
    rationale: str
    detail_level: EditorialDetailLevel
    merge_with: tuple[str, ...] = ()
    section_headline: str | None = None
    section_rank: int | None = None


@dataclass(frozen=True, slots=True)
class ComposedSection:
    """A final weekly report section produced by the editor/composer."""

    headline: str
    dek: str
    body_markdown: str
    included_job_ids: tuple[str, ...]
    detail_level: EditorialDetailLevel


@dataclass(frozen=True, slots=True)
class WeeklyReport:
    """Structured weekly report plus editorial decisions."""

    provider: str
    prompt_version: str
    model_name: str
    generated_at: datetime
    report_title: str
    report_subtitle: str
    editorial_summary: str
    overall_note_to_reader: str
    sections: tuple[ComposedSection, ...]
    decisions: tuple[EditorDecision, ...]

    @property
    def included_story_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.action == "include")

    @property
    def merged_story_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.action == "merge")

    @property
    def excluded_story_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.action == "exclude")
