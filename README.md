# PMR

PMR is an early-stage personal news-intelligence pipeline built around one idea:

use significant repricing in liquid prediction markets as the relevance filter, then investigate what most likely caused the move.

The current repository provides a working first pass of that flow:

1. Filter markets by category and liquidity.
2. Detect weekly repricing events that happened anywhere inside the last 7 days.
3. Run a research step through a provider interface.
4. Draft a Markdown report suitable for a newsletter or analyst workflow.

## Current Scope

What is implemented now:

- A domain model for markets, snapshots, detected repricing events, and research findings.
- A detection engine that scores moves over the last 7 days.
- A live Polymarket provider that fetches active markets from Gamma, price history from the public CLOB API, and open interest from the public Data API.
- A local SQLite snapshot store with retention pruning for live Polymarket runs.
- Category and liquidity filters focused on politics, geopolitics, economics, and macro.
- A Markdown report generator.
- A research-input JSON exporter for downstream agent workflows.
- A sample-data runner so the pipeline works locally without external credentials.
- Interfaces for swapping in real Polymarket ingestion and web/X research providers.

What is not implemented yet:

- Real-time or scheduled jobs.
- Web and X research.
- Email delivery.
- Historical calibration and analyst-review tooling.

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
python3 main.py --source json --input-json data/markets.json
```

## Run Against Live Polymarket Data

To fetch real active Polymarket markets, run the weekly detector, and write the anomaly payloads that a research agent can consume:

```bash
python3 main.py \
  --source polymarket \
  --polymarket-max-markets 100 \
  --polymarket-max-pages 20 \
  --output-markdown out/report.md \
  --output-research-json out/research-inputs.json
```

If you want a reproducible local snapshot store as you evaluate live runs:

```bash
python3 main.py \
  --source polymarket \
  --db-path data/pmr.sqlite3 \
  --snapshot-retention-days 120 \
  --max-markets-per-category 25 \
  --polymarket-refresh-mode incremental \
  --output-research-json out/research-inputs.json
```

Then you can rerun reports from the stored dataset without hitting the live APIs:

```bash
python3 main.py \
  --source stored \
  --db-path data/pmr.sqlite3 \
  --output-markdown out/stored-report.md \
  --output-research-json out/stored-research-inputs.json
```

The live Polymarket flow now supports three refresh modes:

- `incremental`: refresh only the recent tail of already tracked markets, with a small overlap to avoid edge gaps.
- `backfill`: fetch an older chunk for markets that already have recent data but do not yet have a deep retained baseline.
- `full`: refetch the full retained history window for each selected market.

Examples:

```bash
python3 main.py \
  --source polymarket \
  --db-path data/pmr.sqlite3 \
  --polymarket-refresh-mode incremental \
  --incremental-overlap-minutes 180
```

```bash
python3 main.py \
  --source polymarket \
  --db-path data/pmr.sqlite3 \
  --polymarket-refresh-mode backfill \
  --backfill-chunk-days 14
```

This live path currently uses:

- Gamma API for active market discovery
- CLOB `prices-history` for token probability history
- Data API `/oi` for open interest

The Polymarket provider currently favors transparency over sophistication:

- it scans active markets ordered by recent volume,
- infers target categories from market text,
- fetches the tracked outcome history for binary markets,
- caps the number of selected markets per category so politics does not crowd out macro or geopolitics,
- and adapts those into `MarketSeries` objects for the detector.

The local dataset is kept bounded by:

- storing only markets that passed the topical filter,
- keeping one resampled history stream per tracked outcome,
- pruning snapshots older than `snapshot_retention_days`,
- and dropping stale markets that have not been refreshed recently.

When you run `--source polymarket`, PMR first fetches the live delta or backfill chunk, writes it into SQLite, prunes old data, and then runs detection on the merged stored dataset. That keeps reports stable while avoiding repeated full-history pulls.

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

The detector is now structured as a weekly event detector rather than a simple 7-day net-change check.

For each market:

- The detection window is the last 7 days.
- The baseline window is a longer pre-window history, with a preferred length of 90 days.
- The detector looks for repricing that happened anywhere inside the detection window, including spikes that later retrace.
- Markets are scored differently depending on how much history exists.

The current feature set is intentionally simple and explainable:

- `max_abs_move_6h`
- `max_abs_move_24h`
- `weekly_range`
- `close_to_open_move`
- `persistence_of_largest_move`
- `jump_count_over_threshold`

When enough pre-window history exists, those features are normalized against baseline medians such as:

- median absolute 6-hour move
- median absolute 24-hour move
- median rolling weekly range

The detector then combines:

- normalized move magnitude,
- absolute move magnitude,
- persistence,
- liquidity confirmation,
- and history-depth penalties

into a transparent composite score.

History handling is explicit:

- `full_history`: 30+ days of total market history
- `short_history`: 10 to 29 days of total market history
- `insufficient_data`: less than 10 days, excluded from the main ranked anomaly list

This keeps the system usable for new Polymarket markets without pretending short histories are as trustworthy as older ones.

## Architecture

Core modules:

- `pmr/config.py`: runtime thresholds and category rules.
- `pmr/models.py`: shared dataclasses for the pipeline.
- `pmr/detector.py`: weekly event extraction, baseline normalization, and ranking.
- `pmr/polymarket.py`: public Polymarket HTTP client.
- `pmr/providers.py`: provider interfaces plus local mock/json implementations.
- `pmr/research_payloads.py`: JSON serialization for downstream research-agent jobs.
- `pmr/storage.py`: SQLite snapshot persistence, bounded retention, and stored-data loading.
- `pmr/pipeline.py`: orchestration.
- `pmr/reporting.py`: Markdown report rendering.
- `pmr/sample_data.py`: deterministic local demo dataset.

## Planned Next Steps

The highest-value next milestones are:

1. Tighten topical filtering and market selection so live runs focus more aggressively on politics, geopolitics, and macro.
2. Add a historical evaluation loop so thresholds can be calibrated against real weeks of Polymarket behavior.
3. Add a research provider that queries both the web and X, then produces evidence-ranked explanations.
4. Add scheduling and delivery once the signal quality is stable.

## Testing

```bash
python3 -m unittest discover -s tests
```
