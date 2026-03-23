from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from statistics import median
from typing import Sequence

from pmr.config import MonitoringConfig
from pmr.models import (
    BaselineStats,
    ConfidenceLevel,
    HistoryMode,
    LiquidityMetrics,
    MarketSeries,
    MarketSnapshot,
    RepricingEvent,
)
from pmr.story_groups import build_story_family_key


@dataclass(frozen=True, slots=True)
class _HorizonMove:
    start_snapshot: MarketSnapshot
    end_snapshot: MarketSnapshot
    move: float
    abs_move: float


def detect_significant_moves(
    markets: Sequence[MarketSeries],
    config: MonitoringConfig,
) -> list[RepricingEvent]:
    """Return ranked weekly anomalies for markets that pass all filters."""

    events: list[RepricingEvent] = []
    for series in markets:
        candidate = evaluate_market_event(series, config)
        if candidate is None or not candidate.eligible_for_ranking:
            continue
        events.append(candidate)

    ranked = sorted(events, key=lambda item: item.composite_score, reverse=True)
    return _dedupe_story_families(ranked, config=config)


def evaluate_market_event(
    series: MarketSeries,
    config: MonitoringConfig,
) -> RepricingEvent | None:
    """Evaluate one market and return a weekly anomaly candidate."""

    if not _category_matches(series, config):
        return None

    snapshots = tuple(sorted(series.snapshots, key=lambda item: item.observed_at))
    if len(snapshots) < 2:
        return None

    detection_window_end = snapshots[-1].observed_at
    detection_window_start = detection_window_end - timedelta(days=config.detection_window_days)
    detection_window_snapshots = _select_detection_window_snapshots(snapshots, detection_window_start)
    if len(detection_window_snapshots) < 2:
        return None

    total_history_days = _days_between(snapshots[0].observed_at, detection_window_end)
    history_mode = _determine_history_mode(total_history_days, config)

    baseline_window_start = detection_window_start - timedelta(days=config.preferred_baseline_window_days)
    baseline_stats = _build_baseline_stats(
        snapshots=snapshots,
        baseline_window_start=baseline_window_start,
        detection_window_start=detection_window_start,
        config=config,
    )

    detection_endpoints = tuple(
        item for item in snapshots if item.observed_at >= detection_window_start
    )
    moves_6h = _collect_horizon_moves(snapshots, detection_endpoints, timedelta(hours=6))
    moves_24h = _collect_horizon_moves(snapshots, detection_endpoints, timedelta(hours=24))
    largest_6h = max(moves_6h, key=lambda item: item.abs_move, default=None)
    largest_24h = max(moves_24h, key=lambda item: item.abs_move, default=None)
    reference_move = largest_24h or largest_6h

    window_open_probability = detection_window_snapshots[0].probability
    window_close_probability = detection_window_snapshots[-1].probability
    window_high_probability = max(item.probability for item in detection_window_snapshots)
    window_low_probability = min(item.probability for item in detection_window_snapshots)
    weekly_range = window_high_probability - window_low_probability
    close_to_open_move = window_close_probability - window_open_probability
    persistence = _persistence_of_largest_move(reference_move, window_close_probability)
    jump_count = _count_jump_episodes(moves_24h, config.min_abs_move_24h)
    max_move_timestamp = reference_move.end_snapshot.observed_at if reference_move else None

    max_abs_move_6h = largest_6h.abs_move if largest_6h else 0.0
    max_abs_move_24h = largest_24h.abs_move if largest_24h else 0.0
    largest_6h_move = largest_6h.move if largest_6h else 0.0
    largest_24h_move = largest_24h.move if largest_24h else 0.0

    z_6h = _normalized_score(
        value=max_abs_move_6h,
        baseline_value=baseline_stats.median_abs_move_6h,
        floor=max(config.min_abs_move_24h / 2.0, 0.02),
    )
    z_24h = _normalized_score(
        value=max_abs_move_24h,
        baseline_value=baseline_stats.median_abs_move_24h,
        floor=max(config.min_abs_move_24h / 2.0, 0.03),
    )
    z_range = _normalized_score(
        value=weekly_range,
        baseline_value=baseline_stats.median_weekly_range,
        floor=max(config.min_weekly_range / 2.0, 0.05),
    )

    liquidity_metrics = _build_liquidity_metrics(series, config)
    passes_liquidity = _passes_liquidity(series, config)
    confidence_score = _build_confidence_score(
        history_mode=history_mode,
        baseline_stats=baseline_stats,
        detection_snapshot_count=len(detection_window_snapshots),
        passes_liquidity=passes_liquidity,
        config=config,
    )
    confidence_level = _confidence_level(confidence_score)
    composite_score = _composite_score(
        history_mode=history_mode,
        confidence_score=confidence_score,
        z_6h=z_6h,
        z_24h=z_24h,
        z_range=z_range,
        max_abs_move_24h=max_abs_move_24h,
        weekly_range=weekly_range,
        close_to_open_move=close_to_open_move,
        persistence=persistence,
        jump_count=jump_count,
        liquidity_score=liquidity_metrics.liquidity_score,
        config=config,
    )

    notes: list[str] = []
    eligible_for_ranking = True
    exclusion_reason: str | None = None

    if max_abs_move_24h >= config.min_abs_move_24h:
        notes.append(
            f"Max absolute 24h move reached {max_abs_move_24h * 100:.1f} percentage points "
            f"at {max_move_timestamp.isoformat() if max_move_timestamp else 'an unknown time'}."
        )
    if weekly_range >= config.min_weekly_range:
        notes.append(f"Weekly range reached {weekly_range * 100:.1f} percentage points inside the detection window.")
    if abs(close_to_open_move) < config.min_abs_move_24h and weekly_range >= config.min_weekly_range:
        notes.append("The move retraced before the close, but the within-window repricing still looks material.")
    if history_mode == "short_history":
        notes.append(
            f"Short-history mode: only {total_history_days:.1f} days of total market history are available."
        )
    if history_mode == "insufficient_data":
        eligible_for_ranking = False
        exclusion_reason = "insufficient_data"
        notes.append(
            f"Excluded from ranking because only {total_history_days:.1f} days of market history are available."
        )
    if not passes_liquidity:
        eligible_for_ranking = False
        exclusion_reason = "liquidity"
        notes.append("Excluded from ranking because 7d volume or open interest is below the configured minimums.")
    if max_abs_move_24h < config.min_abs_move_24h and weekly_range < config.min_weekly_range:
        eligible_for_ranking = False
        exclusion_reason = exclusion_reason or "below_threshold"
        notes.append("Excluded from ranking because the market did not produce a large enough 24h move or weekly range.")

    if baseline_stats.sample_count_24h == 0:
        notes.append("24h normalization used a fallback floor because the pre-window history is too thin.")
    if baseline_stats.sample_count_weekly_range == 0:
        notes.append("Weekly-range normalization used a fallback because no prior rolling weekly windows were available.")
    notes.append(f"Normalized scores: z_6h={z_6h:.1f}, z_24h={z_24h:.1f}, z_range={z_range:.1f}.")

    return RepricingEvent(
        market=series.market,
        detection_window_start=detection_window_start,
        detection_window_end=detection_window_end,
        history_mode=history_mode,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        eligible_for_ranking=eligible_for_ranking,
        exclusion_reason=exclusion_reason,
        window_open_probability=window_open_probability,
        window_close_probability=window_close_probability,
        window_high_probability=window_high_probability,
        window_low_probability=window_low_probability,
        largest_6h_move=largest_6h_move,
        max_abs_move_6h=max_abs_move_6h,
        largest_24h_move=largest_24h_move,
        max_abs_move_24h=max_abs_move_24h,
        weekly_range=weekly_range,
        close_to_open_move=close_to_open_move,
        persistence_of_largest_move=persistence,
        jump_count_over_threshold=jump_count,
        max_move_timestamp=max_move_timestamp,
        baseline_stats=baseline_stats,
        liquidity_metrics=liquidity_metrics,
        z_6h=z_6h,
        z_24h=z_24h,
        z_range=z_range,
        composite_score=composite_score,
        notes=tuple(notes),
        series=series,
    )


def _dedupe_story_families(
    events: Sequence[RepricingEvent],
    *,
    config: MonitoringConfig,
) -> list[RepricingEvent]:
    limit_per_family = max(config.max_events_per_story_family, 1)
    family_counts: dict[str, int] = {}
    selected: list[RepricingEvent] = []

    for event in events:
        family_key = build_story_family_key(event.market)
        if family_counts.get(family_key, 0) >= limit_per_family:
            continue
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        selected.append(event)
        if len(selected) >= config.max_events:
            break

    return selected


def _select_detection_window_snapshots(
    snapshots: Sequence[MarketSnapshot],
    detection_window_start: datetime,
) -> tuple[MarketSnapshot, ...]:
    anchor = _snapshot_at_or_before(snapshots, detection_window_start)
    in_window = tuple(item for item in snapshots if item.observed_at >= detection_window_start)
    if anchor is None:
        return in_window
    if not in_window:
        return (anchor,)
    if anchor.observed_at == in_window[0].observed_at:
        return in_window
    return (anchor, *in_window)


def _build_baseline_stats(
    snapshots: Sequence[MarketSnapshot],
    baseline_window_start: datetime,
    detection_window_start: datetime,
    config: MonitoringConfig,
) -> BaselineStats:
    pre_window_snapshots = tuple(item for item in snapshots if item.observed_at < detection_window_start)
    baseline_endpoints = tuple(
        item
        for item in pre_window_snapshots
        if item.observed_at >= baseline_window_start
    )
    move_6h = _collect_horizon_moves(pre_window_snapshots, baseline_endpoints, timedelta(hours=6))
    move_24h = _collect_horizon_moves(pre_window_snapshots, baseline_endpoints, timedelta(hours=24))
    weekly_ranges = _collect_rolling_ranges(
        snapshots=pre_window_snapshots,
        period_start=baseline_window_start,
        period_end=detection_window_start,
        window=timedelta(days=config.detection_window_days),
    )
    baseline_start = max(baseline_window_start, snapshots[0].observed_at)
    usable_history_days = max(0.0, _days_between(baseline_start, detection_window_start))
    return BaselineStats(
        usable_history_days=usable_history_days,
        preferred_window_days=config.preferred_baseline_window_days,
        snapshot_count=len(baseline_endpoints),
        median_abs_move_6h=median(item.abs_move for item in move_6h) if move_6h else None,
        median_abs_move_24h=median(item.abs_move for item in move_24h) if move_24h else None,
        median_weekly_range=median(weekly_ranges) if weekly_ranges else None,
        sample_count_6h=len(move_6h),
        sample_count_24h=len(move_24h),
        sample_count_weekly_range=len(weekly_ranges),
    )


def _collect_horizon_moves(
    snapshots: Sequence[MarketSnapshot],
    endpoints: Sequence[MarketSnapshot],
    horizon: timedelta,
) -> tuple[_HorizonMove, ...]:
    timestamps = [item.observed_at for item in snapshots]
    moves: list[_HorizonMove] = []
    for current in endpoints:
        target = current.observed_at - horizon
        index = bisect_right(timestamps, target) - 1
        if index < 0:
            continue
        start_snapshot = snapshots[index]
        if start_snapshot.observed_at >= current.observed_at:
            continue
        move = current.probability - start_snapshot.probability
        moves.append(
            _HorizonMove(
                start_snapshot=start_snapshot,
                end_snapshot=current,
                move=move,
                abs_move=abs(move),
            )
        )
    return tuple(moves)


def _collect_rolling_ranges(
    snapshots: Sequence[MarketSnapshot],
    period_start: datetime,
    period_end: datetime,
    window: timedelta,
) -> tuple[float, ...]:
    timestamps = [item.observed_at for item in snapshots]
    min_queue: deque[int] = deque()
    max_queue: deque[int] = deque()
    start_index = 0
    ranges: list[float] = []

    for index, snapshot in enumerate(snapshots):
        if snapshot.observed_at >= period_end:
            break

        window_start = snapshot.observed_at - window
        while start_index <= index and snapshots[start_index].observed_at < window_start:
            start_index += 1

        while min_queue and min_queue[0] < start_index:
            min_queue.popleft()
        while max_queue and max_queue[0] < start_index:
            max_queue.popleft()

        while min_queue and snapshots[min_queue[-1]].probability >= snapshot.probability:
            min_queue.pop()
        while max_queue and snapshots[max_queue[-1]].probability <= snapshot.probability:
            max_queue.pop()
        min_queue.append(index)
        max_queue.append(index)

        if snapshot.observed_at < period_start:
            continue
        if bisect_right(timestamps, window_start) == 0:
            continue
        if not min_queue or not max_queue:
            continue

        rolling_range = snapshots[max_queue[0]].probability - snapshots[min_queue[0]].probability
        ranges.append(rolling_range)

    return tuple(ranges)


def _snapshot_at_or_before(
    snapshots: Sequence[MarketSnapshot],
    target: datetime,
) -> MarketSnapshot | None:
    timestamps = [item.observed_at for item in snapshots]
    index = bisect_right(timestamps, target) - 1
    if index >= 0:
        return snapshots[index]
    return snapshots[0] if snapshots else None


def _determine_history_mode(
    total_history_days: float,
    config: MonitoringConfig,
) -> HistoryMode:
    if total_history_days >= config.min_history_days_for_full_scoring:
        return "full_history"
    if total_history_days >= config.min_history_days_for_short_scoring:
        return "short_history"
    return "insufficient_data"


def _build_confidence_score(
    history_mode: HistoryMode,
    baseline_stats: BaselineStats,
    detection_snapshot_count: int,
    passes_liquidity: bool,
    config: MonitoringConfig,
) -> float:
    if history_mode == "full_history":
        score = 0.86
    elif history_mode == "short_history":
        score = 0.64
    else:
        score = 0.34

    if baseline_stats.usable_history_days < config.min_history_days_for_short_scoring:
        score -= 0.10
    if baseline_stats.sample_count_24h == 0:
        score -= 0.10
    if baseline_stats.sample_count_weekly_range == 0:
        score -= 0.10
    if detection_snapshot_count < 3:
        score -= 0.10
    if not passes_liquidity:
        score -= 0.10

    return max(0.05, min(0.95, score))


def _confidence_level(score: float) -> ConfidenceLevel:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _normalized_score(value: float, baseline_value: float | None, floor: float) -> float:
    if value <= 0:
        return 0.0
    denominator = max(baseline_value or 0.0, floor)
    return value / denominator


def _build_liquidity_metrics(
    series: MarketSeries,
    config: MonitoringConfig,
) -> LiquidityMetrics:
    market = series.market
    volume_ratio = min(3.0, market.volume_7d_usd / max(config.min_volume_7d_usd, 1.0))
    open_interest_ratio = min(3.0, market.open_interest_usd / max(config.min_open_interest_usd, 1.0))
    activity_floor = max(config.min_volume_7d_usd / max(config.detection_window_days, 1), 1.0)
    activity_ratio = min(3.0, market.volume_24h_usd / activity_floor)
    liquidity_score = 0.25 * volume_ratio + 0.25 * open_interest_ratio + 0.10 * activity_ratio
    return LiquidityMetrics(
        volume_7d_usd=market.volume_7d_usd,
        volume_24h_usd=market.volume_24h_usd,
        open_interest_usd=market.open_interest_usd,
        liquidity_score=liquidity_score,
    )


def _composite_score(
    history_mode: HistoryMode,
    confidence_score: float,
    z_6h: float,
    z_24h: float,
    z_range: float,
    max_abs_move_24h: float,
    weekly_range: float,
    close_to_open_move: float,
    persistence: float,
    jump_count: int,
    liquidity_score: float,
    config: MonitoringConfig,
) -> float:
    normalized_raw = (
        0.20 * min(z_6h, 8.0)
        + 0.45 * min(z_24h, 8.0)
        + 0.35 * min(z_range, 8.0)
    )
    absolute_raw = (
        0.55 * (max_abs_move_24h / max(config.min_abs_move_24h, 0.01))
        + 0.30 * (weekly_range / max(config.min_weekly_range, 0.01))
        + 0.15 * (abs(close_to_open_move) / max(config.min_abs_move_24h, 0.01))
    )

    if history_mode == "full_history":
        normalized_weight = 0.65
        absolute_weight = 0.35
    else:
        normalized_weight = 0.40
        absolute_weight = 0.60

    score = (
        normalized_weight * normalized_raw
        + absolute_weight * absolute_raw
        + 0.35 * persistence
        + 0.15 * min(jump_count, 4)
        + liquidity_score
    )

    if history_mode == "short_history":
        score -= config.short_history_penalty
    if confidence_score < 0.50:
        score -= config.low_confidence_penalty

    return max(0.0, score)


def _persistence_of_largest_move(
    move: _HorizonMove | None,
    close_probability: float,
) -> float:
    if move is None or move.move == 0:
        return 0.0
    retained_move = close_probability - move.start_snapshot.probability
    persistence = retained_move / move.move
    return max(0.0, min(1.0, persistence))


def _count_jump_episodes(
    moves: Sequence[_HorizonMove],
    threshold: float,
) -> int:
    episodes = 0
    in_jump = False
    for move in moves:
        is_jump = move.abs_move >= threshold
        if is_jump and not in_jump:
            episodes += 1
        in_jump = is_jump
    return episodes


def _days_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 86_400.0)


def _category_matches(series: MarketSeries, config: MonitoringConfig) -> bool:
    haystack = " ".join((series.market.category, *series.market.tags)).lower()
    accepted_terms = {
        alias.lower()
        for category in config.target_categories
        for alias in config.category_aliases.get(category, (category,))
    }
    return any(_text_contains_term(haystack, term) for term in accepted_terms)


def _passes_liquidity(series: MarketSeries, config: MonitoringConfig) -> bool:
    market = series.market
    return (
        market.volume_7d_usd >= config.min_volume_7d_usd
        and market.open_interest_usd >= config.min_open_interest_usd
    )


def _text_contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text) is not None
