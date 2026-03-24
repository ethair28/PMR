from __future__ import annotations

from collections import Counter
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
from pmr.market_filters import classify_universe_market_exclusion
from pmr.polymarket import PolymarketClient
from pmr.storage import SnapshotBounds, SnapshotStore
from pmr.universe_selection import (
    UniverseCandidate,
    build_universe_group_key,
    market_priority_sort_key,
    prioritize_grouped_universe_candidates,
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


@dataclass(frozen=True, slots=True)
class PolymarketRejectedMarket:
    """A sampled market-level rejection emitted during a live Polymarket scan."""

    market_id: str | None
    question: str | None
    reason: str
    category: str | None = None
    matched_terms: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PolymarketScanDiagnostics:
    """Explainable scan summary for a live Polymarket universe fetch."""

    pages_scanned: int
    payloads_seen: int
    unique_markets_seen: int
    topic_matches: int
    selected_markets: int
    history_requests: int
    open_interest_fallbacks: int
    per_category_selected: dict[str, int]
    rejection_counts: dict[str, int]
    sampled_rejections: tuple[PolymarketRejectedMarket, ...]


@dataclass(frozen=True, slots=True)
class PolymarketScanResult:
    """Live provider result containing both selected series and scan diagnostics."""

    markets: tuple[MarketSeries, ...]
    diagnostics: PolymarketScanDiagnostics


@dataclass(frozen=True, slots=True)
class _BuildMarketSeriesResult:
    series: MarketSeries | None
    rejection_reason: str | None = None
    detail: str | None = None
    history_requested: bool = False
    used_open_interest_fallback: bool = False


@dataclass(frozen=True, slots=True)
class _TopicalPayloadCandidate:
    payload: dict[str, Any]
    category: str
    matched_terms: tuple[str, ...]
    event_group_key: str
    sort_key: tuple[float, float, float, str]


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
    min_markets_per_category: int = 5
    max_category_share_of_universe: float | None = 0.6
    max_markets_per_category: int | None = None
    max_contracts_per_event: int = 2
    refresh_mode: str = "full"
    existing_snapshot_bounds: dict[str, SnapshotBounds] | None = None
    incremental_overlap_minutes: int = 180
    backfill_chunk_days: int = 14
    current_time: datetime | None = None
    diagnostic_sample_size: int = 25

    def list_market_series(self) -> Sequence[MarketSeries]:
        return self.scan_market_series().markets

    def scan_market_series(self) -> PolymarketScanResult:
        markets: list[MarketSeries] = []
        seen_market_ids: set[str] = set()
        per_category_counts: dict[str, int] = {}
        per_event_counts: dict[str, int] = {}
        rejection_counts: Counter[str] = Counter()
        sampled_rejections: list[PolymarketRejectedMarket] = []
        topical_candidates: list[_TopicalPayloadCandidate] = []
        pages_scanned = 0
        payloads_seen = 0
        unique_markets_seen = 0
        topic_matches = 0
        history_requests = 0
        open_interest_fallbacks = 0

        def record_rejection(
            *,
            payload: dict[str, Any],
            reason: str,
            category: str | None = None,
            matched_terms: tuple[str, ...] = (),
            detail: str | None = None,
        ) -> None:
            rejection_counts[reason] += 1
            if len(sampled_rejections) >= self.diagnostic_sample_size:
                return
            sampled_rejections.append(
                PolymarketRejectedMarket(
                    market_id=_as_optional_str(payload.get("id")),
                    question=_as_optional_str(payload.get("question")),
                    reason=reason,
                    category=category,
                    matched_terms=matched_terms,
                    detail=detail,
                )
            )

        def selection_rejection_reason() -> tuple[str, str | None]:
            if self.max_markets_per_category is not None:
                return "category_cap", "Candidate was trimmed by the configured hard per-category cap."
            return "selection_overflow", "Candidate fell below the soft floor + global overflow universe cutoff."

        def build_result() -> PolymarketScanResult:
            return PolymarketScanResult(
                markets=tuple(markets),
                diagnostics=PolymarketScanDiagnostics(
                    pages_scanned=pages_scanned,
                    payloads_seen=payloads_seen,
                    unique_markets_seen=unique_markets_seen,
                    topic_matches=topic_matches,
                    selected_markets=len(markets),
                    history_requests=history_requests,
                    open_interest_fallbacks=open_interest_fallbacks,
                    per_category_selected=dict(sorted(per_category_counts.items())),
                    rejection_counts=dict(sorted(rejection_counts.items())),
                    sampled_rejections=tuple(sampled_rejections),
                ),
            )

        for page_number in range(self.max_pages):
            offset = page_number * self.page_size
            page = self.client.list_markets(limit=self.page_size, offset=offset)
            if not page:
                break
            pages_scanned += 1
            payloads_seen += len(page)

            for payload in page:
                market_id = str(payload.get("id", ""))
                if not market_id:
                    record_rejection(payload=payload, reason="missing_market_id")
                    continue
                if market_id in seen_market_ids:
                    record_rejection(payload=payload, reason="duplicate_market_id")
                    continue
                seen_market_ids.add(market_id)
                unique_markets_seen += 1
                if not payload.get("active") or payload.get("closed"):
                    record_rejection(payload=payload, reason="inactive_or_closed")
                    continue
                if payload.get("sportsMarketType"):
                    record_rejection(payload=payload, reason="sports_market")
                    continue
                exclusion_reason = classify_universe_market_exclusion(
                    question=_as_optional_str(payload.get("question")),
                    description=_as_optional_str(payload.get("description")),
                    slug=_as_optional_str(payload.get("slug")),
                    event_title=_as_optional_str(payload.get("events", [{}])[0].get("title")) if payload.get("events") else None,
                )
                if exclusion_reason is not None:
                    record_rejection(payload=payload, reason=exclusion_reason)
                    continue
                category, matched_terms = self._infer_category(payload)
                if category is None:
                    record_rejection(payload=payload, reason="off_topic")
                    continue
                topic_matches += 1
                if not _has_binary_clob_shape(payload):
                    record_rejection(
                        payload=payload,
                        reason="non_binary_market",
                        category=category,
                        matched_terms=matched_terms,
                    )
                    continue
                topical_candidates.append(
                    _TopicalPayloadCandidate(
                        payload=payload,
                        category=category,
                        matched_terms=matched_terms,
                        event_group_key=_payload_event_group_key(payload),
                        sort_key=_payload_sort_key(payload),
                    )
                )

            if len(page) < self.page_size:
                break

        group_metrics = _aggregate_topical_group_metrics(topical_candidates)
        prioritized_candidates = prioritize_grouped_universe_candidates(
            [
                UniverseCandidate(
                    item=candidate,
                    category=candidate.category,
                    sort_key=_payload_sort_key(candidate.payload),
                    group_key=candidate.event_group_key,
                    group_sort_key=market_priority_sort_key(
                        volume_7d=group_metrics[candidate.event_group_key]["volume_7d"],
                        volume_24h=group_metrics[candidate.event_group_key]["volume_24h"],
                        depth=group_metrics[candidate.event_group_key]["depth"],
                        market_id=candidate.event_group_key,
                    ),
                )
                for candidate in topical_candidates
            ],
            target_categories=self.target_categories,
            max_items=self.max_markets,
            min_items_per_category=self.min_markets_per_category,
            max_category_share=self.max_category_share_of_universe,
            max_items_per_group=self.max_contracts_per_event,
        )

        for candidate_wrapper in prioritized_candidates:
            candidate = candidate_wrapper.item
            if len(markets) >= self.max_markets:
                reason, detail = selection_rejection_reason()
                record_rejection(
                    payload=candidate.payload,
                    reason=reason,
                    category=candidate.category,
                    matched_terms=candidate.matched_terms,
                    detail=detail,
                )
                continue

            if self.max_markets_per_category is not None and (
                per_category_counts.get(candidate.category, 0) >= self.max_markets_per_category
            ):
                reason, detail = selection_rejection_reason()
                record_rejection(
                    payload=candidate.payload,
                    reason=reason,
                    category=candidate.category,
                    matched_terms=candidate.matched_terms,
                    detail=detail,
                )
                continue
            if per_event_counts.get(candidate.event_group_key, 0) >= self.max_contracts_per_event:
                record_rejection(
                    payload=candidate.payload,
                    reason="event_contract_cap",
                    category=candidate.category,
                    matched_terms=candidate.matched_terms,
                    detail="Candidate was trimmed because too many contracts from the same event were already selected.",
                )
                continue

            build = self._build_market_series(
                candidate.payload,
                inferred_category=candidate.category,
                matched_terms=candidate.matched_terms,
            )
            if build.history_requested:
                history_requests += 1
            if build.used_open_interest_fallback:
                open_interest_fallbacks += 1
            if build.series is None:
                record_rejection(
                    payload=candidate.payload,
                    reason=build.rejection_reason or "build_error",
                    category=candidate.category,
                    matched_terms=candidate.matched_terms,
                    detail=build.detail,
                )
                continue
            markets.append(build.series)
            per_category_counts[candidate.category] = per_category_counts.get(candidate.category, 0) + 1
            per_event_counts[candidate.event_group_key] = per_event_counts.get(candidate.event_group_key, 0) + 1

        return build_result()

    def _build_market_series(
        self,
        payload: dict[str, Any],
        *,
        inferred_category: str,
        matched_terms: tuple[str, ...],
    ) -> _BuildMarketSeriesResult:
        if not payload.get("active") or payload.get("closed"):
            return _BuildMarketSeriesResult(series=None, rejection_reason="inactive_or_closed")
        if payload.get("sportsMarketType"):
            return _BuildMarketSeriesResult(series=None, rejection_reason="sports_market")

        outcomes = _parse_string_list(payload.get("outcomes"))
        token_ids = _parse_string_list(payload.get("clobTokenIds"))
        if len(outcomes) != 2 or len(token_ids) != 2:
            return _BuildMarketSeriesResult(series=None, rejection_reason="non_binary_market")

        tracked_index = _select_tracked_outcome_index(outcomes)
        tracked_token_id = token_ids[tracked_index]

        history_requested = False
        try:
            end_time = self.current_time or datetime.now(timezone.utc)
            start_time, end_time = self._determine_history_window(
                market_id=str(payload["id"]),
                now=end_time,
            )
            if start_time >= end_time:
                return _BuildMarketSeriesResult(series=None, rejection_reason="invalid_history_window")
            history_requested = True
            history = self.client.get_price_history(
                token_id=tracked_token_id,
                start_ts=int(start_time.timestamp()),
                end_ts=int(end_time.timestamp()),
                fidelity_minutes=self.fidelity_minutes,
            )
            snapshots = tuple(_history_item_to_snapshot(item) for item in history if _is_history_item_usable(item))
            if len(snapshots) < 2:
                return _BuildMarketSeriesResult(
                    series=None,
                    rejection_reason="insufficient_history",
                    history_requested=True,
                )
            condition_id = _as_optional_str(payload.get("conditionId"))
            open_interest = self.client.get_open_interest(condition_id) if condition_id else None
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return _BuildMarketSeriesResult(
                series=None,
                rejection_reason="build_error",
                detail=str(exc),
                history_requested=history_requested,
            )

        notes: list[str] = []
        used_open_interest_fallback = False
        if open_interest is None:
            open_interest = _to_float(payload.get("liquidityNum"), default=0.0)
            notes.append("Open interest fell back to Gamma liquidity because Data API open interest was unavailable.")
            used_open_interest_fallback = True

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
        return _BuildMarketSeriesResult(
            series=MarketSeries(
                market=market,
                snapshots=snapshots,
                notes="\n".join(notes) if notes else None,
            ),
            history_requested=True,
            used_open_interest_fallback=used_open_interest_fallback,
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
    target_categories: tuple[str, ...]
    history_days: int
    staleness_hours: int
    max_markets: int
    min_markets_per_category: int
    max_category_share_of_universe: float | None
    max_markets_per_category: int | None = None
    max_contracts_per_event: int = 2

    def list_market_series(self) -> Sequence[MarketSeries]:
        return self.store.load_market_series(
            target_categories=self.target_categories,
            history_days=self.history_days,
            staleness_hours=self.staleness_hours,
            max_markets=self.max_markets,
            min_markets_per_category=self.min_markets_per_category,
            max_category_share_of_universe=self.max_category_share_of_universe,
            max_markets_per_category=self.max_markets_per_category,
            max_contracts_per_event=self.max_contracts_per_event,
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


def _has_binary_clob_shape(payload: dict[str, Any]) -> bool:
    outcomes = _parse_string_list(payload.get("outcomes"))
    token_ids = _parse_string_list(payload.get("clobTokenIds"))
    return len(outcomes) == 2 and len(token_ids) == 2


def _payload_sort_key(payload: dict[str, Any]) -> tuple[float, float, float, str]:
    return market_priority_sort_key(
        volume_7d=_payload_volume_7d(payload),
        volume_24h=_payload_volume_24h(payload),
        depth=_payload_depth(payload),
        market_id=_as_optional_str(payload.get("id")) or "",
    )


def _payload_event_group_key(payload: dict[str, Any]) -> str:
    first_event = payload.get("events", [{}])[0] if payload.get("events") else {}
    return build_universe_group_key(
        event_title=_as_optional_str(first_event.get("title")),
        slug=_as_optional_str(payload.get("slug")),
        question=_as_optional_str(payload.get("question")),
        market_id=_as_optional_str(payload.get("id")) or "",
    )


def _aggregate_topical_group_metrics(
    candidates: Sequence[_TopicalPayloadCandidate],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        group = metrics.setdefault(
            candidate.event_group_key,
            {"volume_7d": 0.0, "volume_24h": 0.0, "depth": 0.0},
        )
        group["volume_7d"] += _payload_volume_7d(candidate.payload)
        group["volume_24h"] += _payload_volume_24h(candidate.payload)
        group["depth"] += _payload_depth(candidate.payload)
    return metrics


def _payload_volume_7d(payload: dict[str, Any]) -> float:
    return _to_float(payload.get("volume1wkClob", payload.get("volume1wk")), default=0.0)


def _payload_volume_24h(payload: dict[str, Any]) -> float:
    return _to_float(payload.get("volume24hrClob", payload.get("volume24hr")), default=0.0)


def _payload_depth(payload: dict[str, Any]) -> float:
    return _to_float(payload.get("liquidityClob", payload.get("liquidityNum")), default=0.0)


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
