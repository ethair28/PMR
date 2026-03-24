from __future__ import annotations

import unittest

from pmr.universe_selection import (
    UniverseCandidate,
    prioritize_grouped_universe_candidates,
    prioritize_universe_candidates,
)


class UniverseSelectionTests(unittest.TestCase):
    def test_selection_uses_category_floors_then_global_overflow(self) -> None:
        candidates = (
            UniverseCandidate(item="pol-1", category="politics", sort_key=(-100.0, -50.0, -20.0, "pol-1")),
            UniverseCandidate(item="pol-2", category="politics", sort_key=(-95.0, -45.0, -20.0, "pol-2")),
            UniverseCandidate(item="pol-3", category="politics", sort_key=(-90.0, -40.0, -20.0, "pol-3")),
            UniverseCandidate(item="geo-1", category="geopolitics", sort_key=(-80.0, -30.0, -20.0, "geo-1")),
            UniverseCandidate(item="macro-1", category="macro", sort_key=(-85.0, -35.0, -20.0, "macro-1")),
        )

        ordered = prioritize_universe_candidates(
            candidates,
            target_categories=("politics", "geopolitics", "economics", "macro"),
            max_items=4,
            min_items_per_category=1,
            max_category_share=0.75,
        )

        self.assertEqual(
            [candidate.item for candidate in ordered[:4]],
            ["pol-1", "geo-1", "macro-1", "pol-2"],
        )

    def test_selection_defers_dominant_category_but_does_not_drop_it(self) -> None:
        candidates = (
            UniverseCandidate(item="pol-1", category="politics", sort_key=(-100.0, -50.0, -20.0, "pol-1")),
            UniverseCandidate(item="pol-2", category="politics", sort_key=(-99.0, -49.0, -20.0, "pol-2")),
            UniverseCandidate(item="pol-3", category="politics", sort_key=(-98.0, -48.0, -20.0, "pol-3")),
            UniverseCandidate(item="pol-4", category="politics", sort_key=(-97.0, -47.0, -20.0, "pol-4")),
            UniverseCandidate(item="macro-1", category="macro", sort_key=(-60.0, -30.0, -20.0, "macro-1")),
            UniverseCandidate(item="geo-1", category="geopolitics", sort_key=(-59.0, -29.0, -20.0, "geo-1")),
        )

        ordered = prioritize_universe_candidates(
            candidates,
            target_categories=("politics", "geopolitics", "economics", "macro"),
            max_items=4,
            min_items_per_category=1,
            max_category_share=0.5,
        )

        self.assertEqual(
            [candidate.item for candidate in ordered],
            ["pol-1", "geo-1", "macro-1", "pol-2", "pol-3", "pol-4"],
        )

    def test_grouped_selection_uses_group_priority_then_caps_children(self) -> None:
        candidates = (
            UniverseCandidate(
                item="slovenia-a",
                category="politics",
                sort_key=(-310.0, -35.0, -20.0, "slovenia-a"),
                group_key="event:slovenia-pm",
                group_sort_key=(-763.0, -68.0, -50.0, "event:slovenia-pm"),
            ),
            UniverseCandidate(
                item="slovenia-b",
                category="politics",
                sort_key=(-286.0, -18.0, -20.0, "slovenia-b"),
                group_key="event:slovenia-pm",
                group_sort_key=(-763.0, -68.0, -50.0, "event:slovenia-pm"),
            ),
            UniverseCandidate(
                item="slovenia-c",
                category="politics",
                sort_key=(-166.0, -14.0, -20.0, "slovenia-c"),
                group_key="event:slovenia-pm",
                group_sort_key=(-763.0, -68.0, -50.0, "event:slovenia-pm"),
            ),
            UniverseCandidate(
                item="other-politics",
                category="politics",
                sort_key=(-400.0, -60.0, -20.0, "other-politics"),
                group_key="event:other-politics",
                group_sort_key=(-400.0, -60.0, -20.0, "event:other-politics"),
            ),
            UniverseCandidate(
                item="geo-1",
                category="geopolitics",
                sort_key=(-340.0, -50.0, -20.0, "geo-1"),
                group_key="event:geo-1",
                group_sort_key=(-340.0, -50.0, -20.0, "event:geo-1"),
            ),
            UniverseCandidate(
                item="macro-1",
                category="macro",
                sort_key=(-330.0, -45.0, -20.0, "macro-1"),
                group_key="event:macro-1",
                group_sort_key=(-330.0, -45.0, -20.0, "event:macro-1"),
            ),
        )

        ordered = prioritize_grouped_universe_candidates(
            candidates,
            target_categories=("politics", "geopolitics", "economics", "macro"),
            max_items=4,
            min_items_per_category=1,
            max_category_share=0.75,
            max_items_per_group=2,
        )

        self.assertEqual(
            [candidate.item for candidate in ordered[:4]],
            ["slovenia-a", "slovenia-b", "geo-1", "macro-1"],
        )


if __name__ == "__main__":
    unittest.main()
