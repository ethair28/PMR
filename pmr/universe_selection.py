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
    group_key: str | None = None
    group_sort_key: tuple[Any, ...] | None = None


@dataclass(frozen=True, slots=True)
class _UniverseCandidateGroup:
    key: str
    category: str
    sort_key: tuple[Any, ...]
    primary_candidates: tuple[UniverseCandidate, ...]
    overflow_candidates: tuple[UniverseCandidate, ...]


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


def prioritize_grouped_universe_candidates(
    candidates: Sequence[UniverseCandidate],
    *,
    target_categories: Sequence[str],
    max_items: int,
    min_items_per_category: int,
    max_category_share: float | None,
    max_items_per_group: int,
) -> tuple[UniverseCandidate, ...]:
    """Return candidates ordered by grouped story/event priority, then child priority within each group."""

    if not candidates:
        return ()

    max_items_per_group = max(1, max_items_per_group)
    groups = _build_candidate_groups(candidates, max_items_per_group=max_items_per_group)
    if not groups:
        return ()

    min_items_per_category = max(0, min_items_per_category)
    categories = tuple(dict.fromkeys(target_categories))
    buckets: dict[str, list[_UniverseCandidateGroup]] = defaultdict(list)
    for group in groups:
        buckets[group.category].append(group)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.sort_key)

    ordered_groups: list[_UniverseCandidateGroup] = []
    selected_group_keys: set[str] = set()
    per_category_counts: dict[str, int] = defaultdict(int)

    def add_group(group: _UniverseCandidateGroup) -> None:
        if group.key in selected_group_keys:
            return
        ordered_groups.append(group)
        selected_group_keys.add(group.key)
        per_category_counts[group.category] += len(group.primary_candidates)

    for category in categories:
        for group in buckets.get(category, ()):
            if per_category_counts[category] >= min_items_per_category:
                break
            add_group(group)

    remaining = sorted(
        (group for group in groups if group.key not in selected_group_keys),
        key=lambda item: item.sort_key,
    )
    soft_share_limit = _soft_category_limit(
        max_items=max_items,
        min_items_per_category=min_items_per_category,
        max_category_share=max_category_share,
    )

    deferred: list[_UniverseCandidateGroup] = []
    for group in remaining:
        group_size = len(group.primary_candidates)
        if soft_share_limit is not None and per_category_counts[group.category] + group_size > soft_share_limit:
            deferred.append(group)
            continue
        add_group(group)

    for group in deferred:
        add_group(group)

    ordered_candidates: list[UniverseCandidate] = []
    for group in ordered_groups:
        ordered_candidates.extend(group.primary_candidates)
    for group in ordered_groups:
        ordered_candidates.extend(group.overflow_candidates)
    return tuple(ordered_candidates)


def market_priority_sort_key(
    *,
    volume_7d: float,
    volume_24h: float,
    depth: float,
    market_id: str,
) -> tuple[float, float, float, str]:
    """Return a sort key that prefers larger, more liquid markets."""

    return (-volume_7d, -volume_24h, -depth, market_id)


def build_universe_group_key(
    *,
    event_title: str | None,
    slug: str | None,
    question: str | None,
    market_id: str,
) -> str:
    """Return a stable event-ish key used to aggregate sibling contracts during selection."""

    if event_title:
        return f"event:{_normalize_group_text(event_title)}"
    if slug:
        return f"slug:{_normalize_group_text(slug)}"
    if question:
        return f"question:{_normalize_group_text(question)}"
    return f"market:{market_id}"


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


def _build_candidate_groups(
    candidates: Sequence[UniverseCandidate],
    *,
    max_items_per_group: int,
) -> tuple[_UniverseCandidateGroup, ...]:
    buckets: dict[str, list[UniverseCandidate]] = defaultdict(list)
    for candidate in candidates:
        group_key = candidate.group_key or f"candidate:{id(candidate)}"
        buckets[group_key].append(candidate)

    groups: list[_UniverseCandidateGroup] = []
    for group_key, group_candidates in buckets.items():
        ordered_candidates = tuple(sorted(group_candidates, key=lambda item: item.sort_key))
        representative = ordered_candidates[0]
        groups.append(
            _UniverseCandidateGroup(
                key=group_key,
                category=representative.category,
                sort_key=representative.group_sort_key or representative.sort_key,
                primary_candidates=ordered_candidates[:max_items_per_group],
                overflow_candidates=ordered_candidates[max_items_per_group:],
            )
        )
    return tuple(groups)


def _normalize_group_text(value: str) -> str:
    return " ".join(value.strip().lower().split())
