from __future__ import annotations

import argparse
from pathlib import Path

from pmr.config import MonitoringConfig
from pmr.pipeline import run_pipeline
from pmr.providers import JsonFileMarketDataProvider, MockResearchProvider, StaticMarketDataProvider
from pmr.sample_data import build_sample_market_series


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect significant prediction-market repricing and draft a Markdown report."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional path to a JSON file containing markets and snapshots.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Window size for move detection.",
    )
    parser.add_argument(
        "--min-volume-7d",
        type=float,
        default=50_000.0,
        help="Minimum 7d volume in USD.",
    )
    parser.add_argument(
        "--min-open-interest",
        type=float,
        default=10_000.0,
        help="Minimum open interest in USD.",
    )
    args = parser.parse_args()

    config = MonitoringConfig(
        lookback_days=args.lookback_days,
        min_volume_7d_usd=args.min_volume_7d,
        min_open_interest_usd=args.min_open_interest,
    )

    if args.input_json:
        market_provider = JsonFileMarketDataProvider(args.input_json)
    else:
        market_provider = StaticMarketDataProvider(build_sample_market_series())

    result = run_pipeline(
        market_data_provider=market_provider,
        research_provider=MockResearchProvider(),
        config=config,
    )
    print(result.markdown_report)
    return 0
