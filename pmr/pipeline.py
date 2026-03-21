from __future__ import annotations

from dataclasses import dataclass

from pmr.config import MonitoringConfig
from pmr.detector import detect_significant_moves
from pmr.providers import MarketDataProvider, ResearchProvider
from pmr.reporting import EventReport, build_markdown_report


@dataclass(frozen=True, slots=True)
class PipelineResult:
    markdown_report: str
    event_reports: tuple[EventReport, ...]


def run_pipeline(
    market_data_provider: MarketDataProvider,
    research_provider: ResearchProvider,
    config: MonitoringConfig | None = None,
) -> PipelineResult:
    config = config or MonitoringConfig()
    markets = tuple(market_data_provider.list_market_series())
    events = detect_significant_moves(markets, config)
    event_reports = tuple(
        EventReport(event=event, research=research_provider.investigate(event))
        for event in events
    )
    markdown_report = build_markdown_report(event_reports=event_reports, config=config)
    return PipelineResult(markdown_report=markdown_report, event_reports=event_reports)
