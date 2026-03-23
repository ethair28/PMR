from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    """One candidate item competing for a limited market-universe slot."""

    item: Any
    category: str
    sort_key: tuple[float, float, float, str]


def prioritize_universe_candidates(
    candidates: Sequence[UniverseCandidate],
    *,
    target_categories: Sequence[str],
    max_items: int,
    min_items_per_category: int,
    max_category_share: float | None,
) -> tuple[UniverseCandidate, ...]:
    """
    Return candidates in selection priority order.

    The ordering is:
    1. A small category floor for each configured topic.
    2. Global overflow ranked by market quality.
    3. Candidates from categories that exceeded the soft share brake, deferred but not dropped.
    """

    if not candidates:
        return ()

    min_items_per_category = max(0, min_items_per_category)
    categories = tuple(dict.fromkeys(target_categories))
    buckets: dict[str, list[UniverseCandidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.category].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.sort_key)

    ordered: list[UniverseCandidate] = []
    selected_ids: set[int] = set()
    per_category_counts: dict[str, int] = defaultdict(int)

    def add_candidate(candidate: UniverseCandidate) -> None:
        candidate_id = id(candidate)
        if candidate_id in selected_ids:
            return
        ordered.append(candidate)
        selected_ids.add(candidate_id)
        per_category_counts[candidate.category] += 1

    for category in categories:
        for candidate in buckets.get(category, ())[:min_items_per_category]:
            add_candidate(candidate)

    remaining = sorted(
        (candidate for candidate in candidates if id(candidate) not in selected_ids),
        key=lambda item: item.sort_key,
    )
    soft_share_limit = _soft_category_limit(
        max_items=max_items,
        min_items_per_category=min_items_per_category,
        max_category_share=max_category_share,
    )

    deferred: list[UniverseCandidate] = []
    for candidate in remaining:
        if soft_share_limit is not None and per_category_counts[candidate.category] >= soft_share_limit:
            deferred.append(candidate)
            continue
        add_candidate(candidate)

    for candidate in deferred:
        add_candidate(candidate)

    return tuple(ordered)


def market_priority_sort_key(
    *,
    volume_7d: float,
    volume_24h: float,
    depth: float,
    market_id: str,
) -> tuple[float, float, float, str]:
    """Return a sort key that prefers larger, more liquid markets."""

    return (-volume_7d, -volume_24h, -depth, market_id)


def _soft_category_limit(
    *,
    max_items: int,
    min_items_per_category: int,
    max_category_share: float | None,
) -> int | None:
    if max_category_share is None:
        return None
    bounded_share = min(max(max_category_share, 0.0), 1.0)
    return max(min_items_per_category, ceil(max_items * bounded_share))
