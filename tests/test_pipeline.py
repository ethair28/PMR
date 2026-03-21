from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pmr.config import MonitoringConfig
from pmr.pipeline import run_pipeline
from pmr.providers import JsonFileMarketDataProvider, MockResearchProvider, StaticMarketDataProvider
from pmr.sample_data import build_sample_market_series


class PipelineTests(unittest.TestCase):
    def test_sample_pipeline_finds_expected_markets(self) -> None:
        result = run_pipeline(
            market_data_provider=StaticMarketDataProvider(build_sample_market_series()),
            research_provider=MockResearchProvider(),
            config=MonitoringConfig(),
        )

        event_ids = [item.event.market.market_id for item in result.event_reports]
        self.assertEqual(
            event_ids,
            [
                "regional-offensive-2026",
                "primary-candidate-2028",
                "fed-cut-june-2026",
            ],
        )
        self.assertIn("# PMR Repricing Report", result.markdown_report)

    def test_json_provider_loads_market_file(self) -> None:
        payload = {
            "markets": [
                {
                    "market_id": "test-market",
                    "question": "Will inflation print below 2% this year?",
                    "category": "economics",
                    "tags": ["inflation"],
                    "volume_7d_usd": 100000,
                    "volume_24h_usd": 25000,
                    "open_interest_usd": 50000,
                    "snapshots": [
                        {"observed_at": "2026-03-10T00:00:00+00:00", "probability": 0.2},
                        {"observed_at": "2026-03-21T00:00:00+00:00", "probability": 0.4},
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "markets.json"
            path.write_text(json.dumps(payload))
            provider = JsonFileMarketDataProvider(path)

            markets = provider.list_market_series()

        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].market.market_id, "test-market")


if __name__ == "__main__":
    unittest.main()
