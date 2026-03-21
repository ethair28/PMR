from dataclasses import dataclass, field


@dataclass(slots=True)
class MonitoringConfig:
    lookback_days: int = 7
    min_volume_7d_usd: float = 50_000.0
    min_open_interest_usd: float = 10_000.0
    min_abs_move: float = 0.08
    min_anomaly_ratio: float = 2.5
    max_events: int = 8
    target_categories: tuple[str, ...] = (
        "politics",
        "geopolitics",
        "economics",
        "macro",
    )
    category_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "politics": ("politics", "political", "elections", "government"),
            "geopolitics": (
                "geopolitics",
                "geopolitical",
                "foreign policy",
                "war",
                "conflict",
                "international",
            ),
            "economics": ("economics", "economic", "economy"),
            "macro": ("macro", "macroeconomics", "rates", "inflation", "fed"),
        }
    )
