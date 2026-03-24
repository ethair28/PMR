from dataclasses import dataclass, field


@dataclass(slots=True)
class MonitoringConfig:
    """Runtime knobs for weekly event detection and ranking."""

    detection_window_days: int = 7
    preferred_baseline_window_days: int = 90
    snapshot_retention_days: int = 120
    stored_market_staleness_hours: int = 24
    min_history_days_for_full_scoring: int = 30
    min_history_days_for_short_scoring: int = 10
    feature_interval_minutes: int = 60
    min_abs_move_24h: float = 0.05
    min_weekly_range: float = 0.08
    min_volume_7d_usd: float = 50_000.0
    min_open_interest_usd: float = 10_000.0
    short_history_penalty: float = 0.35
    low_confidence_penalty: float = 0.25
    max_events: int = 12
    min_markets_per_category: int = 5
    max_category_share_of_universe: float | None = 0.6
    max_markets_per_category: int | None = None
    max_contracts_per_event: int = 2
    max_events_per_story_family: int = 1
    max_related_markets_per_story: int = 3
    target_categories: tuple[str, ...] = (
        "politics",
        "geopolitics",
        "economics",
        "macro",
    )
    category_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "politics": (
                "politics",
                "political",
                "elections",
                "election",
                "government",
                "president",
                "senate",
                "house",
                "congress",
                "prime minister",
                "minister",
                "cabinet",
                "referendum",
                "approval",
            ),
            "geopolitics": (
                "geopolitics",
                "geopolitical",
                "foreign policy",
                "war",
                "conflict",
                "international",
                "ceasefire",
                "military",
                "sanctions",
                "ukraine",
                "russia",
                "china",
                "taiwan",
                "israel",
                "gaza",
                "iran",
                "nato",
            ),
            "economics": (
                "economics",
                "economic",
                "economy",
                "gdp",
                "jobs",
                "payrolls",
                "unemployment",
                "tariffs",
                "trade",
                "recession",
                "growth",
            ),
            "macro": (
                "macro",
                "macroeconomics",
                "rates",
                "rate cut",
                "interest rate",
                "inflation",
                "cpi",
                "pce",
                "fed",
                "fomc",
                "yield",
                "treasury",
                "central bank",
            ),
        }
    )
