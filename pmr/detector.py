from __future__ import annotations

from datetime import timedelta
from math import log10
from statistics import median
from typing import Iterable, Sequence

from pmr.config import MonitoringConfig
from pmr.models import MarketSeries, MarketSnapshot, RepricingEvent


def detect_significant_moves(
    markets: Sequence[MarketSeries],
    config: MonitoringConfig,
) -> list[RepricingEvent]:
    events: list[RepricingEvent] = []
    for series in markets:
        if not _category_matches(series, config):
            continue
        if not _passes_liquidity(series, config):
            continue
        event = _detect_event(series, config)
        if event is None:
            continue
        if event.abs_move < config.min_abs_move:
            continue
        if event.anomaly_ratio < config.min_anomaly_ratio:
            continue
        events.append(event)

    return sorted(events, key=lambda item: item.significance_score, reverse=True)[: config.max_events]


def _detect_event(series: MarketSeries, config: MonitoringConfig) -> RepricingEvent | None:
    snapshots = tuple(sorted(series.snapshots, key=lambda item: item.observed_at))
    if len(snapshots) < 2:
        return None

    end_snapshot = snapshots[-1]
    window_start = end_snapshot.observed_at - timedelta(days=config.lookback_days)
    start_snapshot = _select_start_snapshot(snapshots, window_start)
    if start_snapshot is None or start_snapshot.observed_at >= end_snapshot.observed_at:
        return None

    move = end_snapshot.probability - start_snapshot.probability
    abs_move = abs(move)

    baseline_moves = tuple(
        abs(current.probability - previous.probability)
        for previous, current in _pairwise(snapshots)
        if current.observed_at <= window_start
    )
    baseline_abs_daily_move = median(baseline_moves) if baseline_moves else 0.02
    anomaly_ratio = abs_move / max(baseline_abs_daily_move, 0.01)
    liquidity_score = _liquidity_score(
        volume_7d_usd=series.market.volume_7d_usd,
        open_interest_usd=series.market.open_interest_usd,
    )
    significance_score = abs_move * anomaly_ratio * liquidity_score

    return RepricingEvent(
        market=series.market,
        start_snapshot=start_snapshot,
        end_snapshot=end_snapshot,
        window_days=config.lookback_days,
        move=move,
        abs_move=abs_move,
        baseline_abs_daily_move=baseline_abs_daily_move,
        anomaly_ratio=anomaly_ratio,
        liquidity_score=liquidity_score,
        significance_score=significance_score,
        series=series,
    )


def _select_start_snapshot(
    snapshots: Sequence[MarketSnapshot],
    window_start,
) -> MarketSnapshot | None:
    candidates_before = [item for item in snapshots if item.observed_at <= window_start]
    if candidates_before:
        return candidates_before[-1]

    candidates_after = [item for item in snapshots if item.observed_at >= window_start]
    if candidates_after:
        return candidates_after[0]
    return None


def _category_matches(series: MarketSeries, config: MonitoringConfig) -> bool:
    haystack = " ".join((series.market.category, *series.market.tags)).lower()
    accepted_terms = {
        alias.lower()
        for category in config.target_categories
        for alias in config.category_aliases.get(category, (category,))
    }
    return any(term in haystack for term in accepted_terms)


def _passes_liquidity(series: MarketSeries, config: MonitoringConfig) -> bool:
    market = series.market
    return (
        market.volume_7d_usd >= config.min_volume_7d_usd
        and market.open_interest_usd >= config.min_open_interest_usd
    )


def _liquidity_score(volume_7d_usd: float, open_interest_usd: float) -> float:
    raw = log10(1.0 + max(volume_7d_usd, 0.0) + max(open_interest_usd, 0.0))
    return max(1.0, raw / 4.0)


def _pairwise(items: Sequence[MarketSnapshot]) -> Iterable[tuple[MarketSnapshot, MarketSnapshot]]:
    for index in range(1, len(items)):
        yield items[index - 1], items[index]
