from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence

from pmr.models import (
    Market,
    MarketSeries,
    MarketSnapshot,
    RepricingEvent,
    ResearchFinding,
)


class MarketDataProvider(Protocol):
    def list_market_series(self) -> Sequence[MarketSeries]:
        """Return market histories for downstream filtering and analysis."""


class ResearchProvider(Protocol):
    def investigate(self, event: RepricingEvent) -> ResearchFinding:
        """Investigate a repricing event and return a synthesized finding."""


@dataclass(slots=True)
class StaticMarketDataProvider:
    markets: Sequence[MarketSeries]

    def list_market_series(self) -> Sequence[MarketSeries]:
        return self.markets


@dataclass(slots=True)
class JsonFileMarketDataProvider:
    path: Path

    def list_market_series(self) -> Sequence[MarketSeries]:
        payload = json.loads(self.path.read_text())
        return tuple(_market_series_from_dict(item) for item in payload["markets"])


@dataclass(slots=True)
class MockResearchProvider:
    """Placeholder provider until web/X-backed research is wired in."""

    def investigate(self, event: RepricingEvent) -> ResearchFinding:
        hints = event.series.research_hints
        if len(hints) >= 2:
            explanation_type = "clear"
            confidence = min(0.9, 0.62 + 0.08 * len(hints))
            summary = (
                "Multiple independent clues line up with the repricing, "
                "so the move likely reflects a real information update."
            )
        elif len(hints) == 1:
            explanation_type = "plausible"
            confidence = 0.58
            summary = (
                "There is one concrete lead, but the evidence is still too thin "
                "to call the explanation definitive."
            )
        else:
            explanation_type = "speculative"
            confidence = 0.32
            summary = (
                "No research backend is configured yet, so the cause remains "
                "a hypothesis rather than an evidenced explanation."
            )

        evidence = hints or (
            "Run a web and X research pass for this market before treating the move as explained.",
        )
        caveats = ()
        if event.anomaly_ratio < 3.0:
            caveats += ("The move is notable, but it is not far above recent realized volatility.",)
        if event.market.volume_24h_usd < event.market.volume_7d_usd / 10:
            caveats += ("Trading activity cooled after the move, which can make the signal less reliable.",)

        return ResearchFinding(
            explanation_type=explanation_type,
            summary=summary,
            confidence=confidence,
            evidence=evidence,
            caveats=caveats,
        )


def _market_series_from_dict(payload: dict) -> MarketSeries:
    market = Market(
        market_id=payload["market_id"],
        question=payload["question"],
        category=payload["category"],
        tags=tuple(payload.get("tags", ())),
        url=payload.get("url"),
        volume_7d_usd=float(payload.get("volume_7d_usd", 0.0)),
        volume_24h_usd=float(payload.get("volume_24h_usd", 0.0)),
        open_interest_usd=float(payload.get("open_interest_usd", 0.0)),
    )
    snapshots = tuple(
        MarketSnapshot(
            observed_at=datetime.fromisoformat(item["observed_at"]),
            probability=float(item["probability"]),
        )
        for item in payload["snapshots"]
    )
    return MarketSeries(
        market=market,
        snapshots=snapshots,
        research_hints=tuple(payload.get("research_hints", ())),
        notes=payload.get("notes"),
    )
