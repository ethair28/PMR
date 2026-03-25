from __future__ import annotations

from dataclasses import dataclass

from pmr.config import MonitoringConfig
from pmr.detector import detect_significant_moves
from pmr.models import RepricingEvent
from pmr.providers import MarketDataProvider


@dataclass(frozen=True, slots=True)
class DetectionPipelineResult:
    events: tuple[RepricingEvent, ...]


def run_detection_pipeline(
    market_data_provider: MarketDataProvider,
    config: MonitoringConfig | None = None,
) -> DetectionPipelineResult:
    config = config or MonitoringConfig()
    markets = tuple(market_data_provider.list_market_series())
    events = detect_significant_moves(markets, config)
    return DetectionPipelineResult(events=events)
