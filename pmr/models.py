from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ExplanationType = Literal["clear", "plausible", "speculative"]
HistoryMode = Literal["full_history", "short_history", "insufficient_data"]
ConfidenceLevel = Literal["high", "medium", "low"]
StoryTypeHint = Literal["live_repricing", "resolved_surprise", "late_stage_resolution"]


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
class ResearchFinding:
    """Research-layer explanation attached to a detected anomaly."""

    explanation_type: ExplanationType
    summary: str
    confidence: float
    evidence: tuple[str, ...]
    caveats: tuple[str, ...] = ()
