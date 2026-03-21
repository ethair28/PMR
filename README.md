# PMR

PMR is an early-stage personal news-intelligence pipeline built around one idea:

use significant repricing in liquid prediction markets as the relevance filter, then investigate what most likely caused the move.

The current repository provides a working first pass of that flow:

1. Filter markets by category and liquidity.
2. Detect 7-day repricing events that are large relative to each market's own baseline behavior.
3. Run a research step through a provider interface.
4. Draft a Markdown report suitable for a newsletter or analyst workflow.

## Current Scope

What is implemented now:

- A domain model for markets, snapshots, detected repricing events, and research findings.
- A detection engine that scores moves over the last 7 days.
- Category and liquidity filters focused on politics, geopolitics, economics, and macro.
- A Markdown report generator.
- A sample-data runner so the pipeline works locally without external credentials.
- Interfaces for swapping in real Polymarket ingestion and web/X research providers.

What is not implemented yet:

- Live Polymarket ingestion.
- Real-time or scheduled jobs.
- Web and X research.
- Email delivery.
- Persistent storage and historical backfills.

## Run

```bash
python3 main.py
```

Or, if you want to use the package entrypoint:

```bash
python3 -m pmr.cli
```

You can also point the CLI at a local JSON dataset:

```bash
python3 main.py --input-json data/markets.json
```

## JSON Input Shape

The JSON provider expects a file in this shape:

```json
{
  "markets": [
    {
      "market_id": "fed-cut-june-2026",
      "question": "Will the Fed cut rates by June 2026?",
      "category": "macro",
      "tags": ["rates", "fed"],
      "url": "https://example.com/market",
      "volume_7d_usd": 1450000,
      "volume_24h_usd": 280000,
      "open_interest_usd": 410000,
      "research_hints": [
        "Softer CPI data may have shifted expectations."
      ],
      "snapshots": [
        {
          "observed_at": "2026-03-14T00:00:00+00:00",
          "probability": 0.38
        },
        {
          "observed_at": "2026-03-21T00:00:00+00:00",
          "probability": 0.47
        }
      ]
    }
  ]
}
```

## Detection Logic

Each market is screened with these rules:

- The market category or tags must match the target coverage areas.
- The market must exceed minimum 7-day volume and open-interest thresholds.
- The net move over the configured lookback window must exceed a minimum absolute threshold.
- The move must also be large relative to the market's own historical daily behavior.

The anomaly ratio is currently:

```text
abs(window_move) / median(abs(prior_daily_moves))
```

That is intentionally simple. It is good enough for a first pass, but the likely next upgrade is a more robust event score that handles:

- intraday moves,
- mean reversion,
- volatility regimes,
- and market-specific calibration.

## Architecture

Core modules:

- `pmr/config.py`: runtime thresholds and category rules.
- `pmr/models.py`: shared dataclasses for the pipeline.
- `pmr/detector.py`: significance detection and filtering.
- `pmr/providers.py`: provider interfaces plus local mock/json implementations.
- `pmr/pipeline.py`: orchestration.
- `pmr/reporting.py`: Markdown report rendering.
- `pmr/sample_data.py`: deterministic local demo dataset.

## Planned Next Steps

The highest-value next milestones are:

1. Add a Polymarket adapter that fetches active markets plus recent price history.
2. Add a research provider that queries both the web and X, then produces evidence-ranked explanations.
3. Persist snapshots so anomaly detection can use a longer and cleaner baseline.
4. Add scheduling plus email distribution once the report quality is reliable.

## Testing

```bash
python3 -m unittest discover -s tests
```
