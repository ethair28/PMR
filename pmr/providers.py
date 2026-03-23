from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from pmr.models import (
    Market,
    MarketSeries,
    MarketSnapshot,
    RepricingEvent,
    ResearchFinding,
)
from pmr.polymarket import PolymarketClient
from pmr.storage import SnapshotBounds, SnapshotStore


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
class PolymarketMarketDataProvider:
    """Fetch live Polymarket markets and adapt them into PMR's market-series model."""

    client: PolymarketClient
    max_markets: int = 100
    page_size: int = 100
    max_pages: int = 20
    history_days: int = 97
    fidelity_minutes: int = 60
    target_categories: tuple[str, ...] = ("politics", "geopolitics", "economics", "macro")
    category_aliases: dict[str, tuple[str, ...]] | None = None
    max_markets_per_category: int | None = None
    refresh_mode: str = "full"
    existing_snapshot_bounds: dict[str, SnapshotBounds] | None = None
    incremental_overlap_minutes: int = 180
    backfill_chunk_days: int = 14
    current_time: datetime | None = None

    def list_market_series(self) -> Sequence[MarketSeries]:
        markets: list[MarketSeries] = []
        seen_market_ids: set[str] = set()
        per_category_counts: dict[str, int] = {}
        max_per_category = self.max_markets_per_category or max(
            1, self.max_markets // max(len(self.target_categories), 1)
        )

        for page_number in range(self.max_pages):
            offset = page_number * self.page_size
            page = self.client.list_markets(limit=self.page_size, offset=offset)
            if not page:
                break

            for payload in page:
                market_id = str(payload.get("id", ""))
                if not market_id or market_id in seen_market_ids:
                    continue
                seen_market_ids.add(market_id)
                category, matched_terms = self._infer_category(payload)
                if category is None:
                    continue
                if per_category_counts.get(category, 0) >= max_per_category:
                    continue

                try:
                    series = self._build_market_series(payload, inferred_category=category, matched_terms=matched_terms)
                except (OSError, KeyError, TypeError, ValueError):
                    continue
                if series is None:
                    continue
                markets.append(series)
                per_category_counts[category] = per_category_counts.get(category, 0) + 1
                if len(markets) >= self.max_markets:
                    return tuple(markets)

            if len(page) < self.page_size:
                break

        return tuple(markets)

    def _build_market_series(
        self,
        payload: dict[str, Any],
        *,
        inferred_category: str,
        matched_terms: tuple[str, ...],
    ) -> MarketSeries | None:
        if not payload.get("active") or payload.get("closed"):
            return None
        if payload.get("sportsMarketType"):
            return None

        outcomes = _parse_string_list(payload.get("outcomes"))
        token_ids = _parse_string_list(payload.get("clobTokenIds"))
        if len(outcomes) != 2 or len(token_ids) != 2:
            return None

        tracked_index = _select_tracked_outcome_index(outcomes)
        tracked_token_id = token_ids[tracked_index]

        end_time = self.current_time or datetime.now(timezone.utc)
        start_time, end_time = self._determine_history_window(
            market_id=str(payload["id"]),
            now=end_time,
        )
        if start_time >= end_time:
            return None
        history = self.client.get_price_history(
            token_id=tracked_token_id,
            start_ts=int(start_time.timestamp()),
            end_ts=int(end_time.timestamp()),
            fidelity_minutes=self.fidelity_minutes,
        )
        snapshots = tuple(_history_item_to_snapshot(item) for item in history if _is_history_item_usable(item))
        if len(snapshots) < 2:
            return None

        condition_id = _as_optional_str(payload.get("conditionId"))
        open_interest = self.client.get_open_interest(condition_id) if condition_id else None
        notes: list[str] = []
        if open_interest is None:
            open_interest = _to_float(payload.get("liquidityNum"), default=0.0)
            notes.append("Open interest fell back to Gamma liquidity because Data API open interest was unavailable.")

        first_event = payload.get("events", [{}])[0] if payload.get("events") else {}
        slug = _as_optional_str(payload.get("slug"))
        url = f"https://polymarket.com/event/{slug}" if slug else None
        description = _as_optional_str(payload.get("description"))
        event_title = _as_optional_str(first_event.get("title"))
        event_context = _as_optional_str(first_event.get("eventMetadata", {}).get("context_description"))
        if event_context:
            notes.append(f"Event context: {event_context}")

        market = Market(
            market_id=str(payload["id"]),
            question=_as_optional_str(payload.get("question")) or "Untitled market",
            category=inferred_category,
            tags=tuple(sorted({inferred_category, *matched_terms})),
            slug=slug,
            url=url,
            description=description,
            condition_id=condition_id,
            tracked_outcome=outcomes[tracked_index],
            tracked_token_id=tracked_token_id,
            event_title=event_title,
            volume_7d_usd=_to_float(payload.get("volume1wkClob", payload.get("volume1wk")), default=0.0),
            volume_24h_usd=_to_float(payload.get("volume24hrClob", payload.get("volume24hr")), default=0.0),
            open_interest_usd=open_interest,
        )
        return MarketSeries(
            market=market,
            snapshots=snapshots,
            notes="\n".join(notes) if notes else None,
        )

    def _infer_category(self, payload: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
        aliases = self.category_aliases or {}
        text = " ".join(
            filter(
                None,
                (
                    _as_optional_str(payload.get("question")),
                    _as_optional_str(payload.get("description")),
                    _as_optional_str(payload.get("slug")),
                    _as_optional_str(payload.get("events", [{}])[0].get("title")) if payload.get("events") else None,
                    _as_optional_str(payload.get("events", [{}])[0].get("eventMetadata", {}).get("context_description"))
                    if payload.get("events")
                    else None,
                ),
            )
        ).lower()

        scores: dict[str, list[str]] = {}
        for category in self.target_categories:
            terms = aliases.get(category, (category,))
            matched = [term for term in terms if _text_contains_term(text, term)]
            if matched:
                scores[category] = matched

        if not scores:
            return None, ()

        best_category = max(
            scores.items(),
            key=lambda item: (len(item[1]), max((len(term) for term in item[1]), default=0)),
        )[0]
        return best_category, tuple(scores[best_category])

    def _determine_history_window(
        self,
        *,
        market_id: str,
        now: datetime,
    ) -> tuple[datetime, datetime]:
        full_start = now - timedelta(days=self.history_days)
        if self.refresh_mode == "full" or not self.existing_snapshot_bounds:
            return full_start, now

        bounds = self.existing_snapshot_bounds.get(market_id)
        if bounds is None:
            return full_start, now

        overlap = timedelta(minutes=self.incremental_overlap_minutes)
        if self.refresh_mode == "incremental":
            return max(full_start, bounds.latest_observed_at - overlap), now

        if self.refresh_mode == "backfill":
            if bounds.earliest_observed_at <= full_start:
                return max(full_start, bounds.latest_observed_at - overlap), now
            backfill_start = max(full_start, bounds.earliest_observed_at - timedelta(days=self.backfill_chunk_days))
            backfill_end = min(now, bounds.earliest_observed_at + overlap)
            if backfill_start < backfill_end:
                return backfill_start, backfill_end
            return max(full_start, bounds.latest_observed_at - overlap), now

        return full_start, now


@dataclass(slots=True)
class StoredMarketDataProvider:
    """Load recent market series from the local SQLite snapshot store."""

    store: SnapshotStore
    history_days: int
    staleness_hours: int
    max_markets: int
    max_markets_per_category: int

    def list_market_series(self) -> Sequence[MarketSeries]:
        return self.store.load_market_series(
            history_days=self.history_days,
            staleness_hours=self.staleness_hours,
            max_markets=self.max_markets,
            max_markets_per_category=self.max_markets_per_category,
        )


@dataclass(slots=True)
class MockResearchProvider:
    """Placeholder provider until web/X-backed research is wired in."""

    def investigate(self, event: RepricingEvent) -> ResearchFinding:
        hints = event.series.research_hints
        signal_cap = 0.45 + 0.5 * event.confidence_score
        if len(hints) >= 2:
            explanation_type = "clear"
            confidence = min(signal_cap, min(0.9, 0.62 + 0.08 * len(hints)))
            summary = (
                "Multiple independent clues line up with the repricing, "
                "so the move likely reflects a real information update."
            )
        elif len(hints) == 1:
            explanation_type = "plausible"
            confidence = min(signal_cap, 0.58)
            summary = (
                "There is one concrete lead, but the evidence is still too thin "
                "to call the explanation definitive."
            )
        else:
            explanation_type = "speculative"
            confidence = min(signal_cap, 0.32)
            summary = (
                "No research backend is configured yet, so the cause remains "
                "a hypothesis rather than an evidenced explanation."
            )

        evidence = hints or (
            "Run a web and X research pass for this market before treating the move as explained.",
        )
        caveats = ()
        if event.history_mode != "full_history":
            caveats += ("The detector is operating without a deep baseline, so signal confidence is lower.",)
        if event.persistence_of_largest_move < 0.35 and event.weekly_range >= event.max_abs_move_24h:
            caveats += ("A large part of the move retraced before the close, which makes causal attribution noisier.",)
        if event.jump_count_over_threshold > 1:
            caveats += ("There were multiple large jump episodes during the week, not just a single clean repricing.",)
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
        slug=payload.get("slug"),
        url=payload.get("url"),
        description=payload.get("description"),
        condition_id=payload.get("condition_id"),
        tracked_outcome=payload.get("tracked_outcome"),
        tracked_token_id=payload.get("tracked_token_id"),
        event_title=payload.get("event_title"),
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


def _parse_string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ()
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed)
    return ()


def _select_tracked_outcome_index(outcomes: Sequence[str]) -> int:
    for index, outcome in enumerate(outcomes):
        if outcome.strip().lower() == "yes":
            return index
    return 0


def _history_item_to_snapshot(item: dict[str, Any]) -> MarketSnapshot:
    return MarketSnapshot(
        observed_at=datetime.fromtimestamp(float(item["t"]), tz=timezone.utc),
        probability=float(item["p"]),
    )


def _is_history_item_usable(item: dict[str, Any]) -> bool:
    try:
        float(item["t"])
        float(item["p"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text) is not None
