from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ExplanationType = Literal["clear", "plausible", "speculative"]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    observed_at: datetime
    probability: float


@dataclass(frozen=True, slots=True)
class Market:
    market_id: str
    question: str
    category: str
    tags: tuple[str, ...] = ()
    url: str | None = None
    volume_7d_usd: float = 0.0
    volume_24h_usd: float = 0.0
    open_interest_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketSeries:
    market: Market
    snapshots: tuple[MarketSnapshot, ...]
    research_hints: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class RepricingEvent:
    market: Market
    start_snapshot: MarketSnapshot
    end_snapshot: MarketSnapshot
    window_days: int
    move: float
    abs_move: float
    baseline_abs_daily_move: float
    anomaly_ratio: float
    liquidity_score: float
    significance_score: float
    series: MarketSeries = field(repr=False)


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    explanation_type: ExplanationType
    summary: str
    confidence: float
    evidence: tuple[str, ...]
    caveats: tuple[str, ...] = ()
