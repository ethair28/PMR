# PMR

PMR is an early-stage personal news-intelligence pipeline built around one idea:

use significant repricing in liquid prediction markets as the relevance filter, then investigate what most likely caused the move.

The current repository provides a working first pass of that flow:

1. Filter markets by category and liquidity.
2. Detect weekly repricing events that happened anywhere inside the last 7 days.
3. Export typed research jobs for a second-stage research pipeline.
4. Run an X-first research pass and persist structured research results.

## Current Scope

What is implemented now:

- A domain model for markets, snapshots, detected repricing events, research jobs, normalized evidence, and structured research results.
- A detection engine that scores moves over the last 7 days.
- A live Polymarket provider that fetches active markets from Gamma, price history from the public CLOB API, and open interest from the public Data API.
- A local SQLite snapshot store with retention pruning for live Polymarket runs.
- Category and liquidity filters focused on politics, geopolitics, economics, and macro.
- Soft universe selection with category floors plus global overflow instead of rigid equal caps.
- Event-aware universe selection, so multi-contract stories compete on aggregated event liquidity rather than per-contract liquidity alone.
- Explicit exclusion of public-market-proxy contracts such as gold, oil, and Bitcoin price-threshold markets.
- Explicit exclusion of social-activity tracker markets such as tweet-count or follower-count contracts.
- Story-family deduping so related contract variants collapse into one final research candidate.
- Editorial hints on each anomaly, including whether the move looks more like a live repricing, a resolved surprise, or a late-stage resolution.
- An optional detector-only Markdown report generator for debugging and tuning.
- A research-input JSON exporter that now emits story-oriented research jobs alongside the raw anomaly rows.
- A separate research runner that consumes those jobs, caches normalized evidence in bounded SQLite storage, and writes structured research results JSON.
- A sample-data runner so the pipeline works locally without external credentials.
- An xAI SDK-backed research source and synthesizer abstraction for X-first live research plus web corroboration.

What is not implemented yet:

- Real-time or scheduled jobs.
- Email delivery.
- Historical calibration and analyst-review tooling.
- Final multi-story newsletter composition.

## Run

These examples assume you are using the uv-managed environment. If your shell is not already inside `.venv`, prefix commands with `uv run`.

```bash
uv run python main.py
```

Or, if you want to use the package entrypoint:

```bash
uv run python -m pmr.cli
```

You can also point the CLI at a local JSON dataset:

```bash
uv run python main.py --source json --input-json data/markets.json
```

## Run Against Live Polymarket Data

To fetch real active Polymarket markets, run the weekly detector, and write the anomaly payloads that the research stage consumes:

```bash
uv run python main.py \
  --source polymarket \
  --polymarket-max-markets 125 \
  --polymarket-max-pages 20 \
  --output-research-json out/research-inputs.json \
  --output-scan-json out/scan-diagnostics.json
```

If you want the human-readable detector report for debugging, add either:

- `--output-markdown out/report.md`
- `--print-markdown`

If you want a reproducible local snapshot store as you evaluate live runs:

```bash
uv run python main.py \
  --source polymarket \
  --db-path data/pmr.sqlite3 \
  --snapshot-retention-days 120 \
  --min-markets-per-category 5 \
  --max-category-share 0.6 \
  --polymarket-refresh-mode incremental \
  --output-research-json out/research-inputs.json
```

Then you can rerun reports from the stored dataset without hitting the live APIs:

```bash
uv run python main.py \
  --source stored \
  --db-path data/pmr.sqlite3 \
  --output-research-json out/stored-research-inputs.json
```

The live Polymarket flow now supports three refresh modes:

- `incremental`: refresh only the recent tail of already tracked markets, with a small overlap to avoid edge gaps.
- `backfill`: fetch an older chunk for markets that already have recent data but do not yet have a deep retained baseline.
- `full`: refetch the full retained history window for each selected market.

Examples:

```bash
uv run python main.py \
  --source polymarket \
  --db-path data/pmr.sqlite3 \
  --polymarket-refresh-mode incremental \
  --incremental-overlap-minutes 180
```

```bash
uv run python main.py \
  --source polymarket \
  --db-path data/pmr.sqlite3 \
  --polymarket-refresh-mode backfill \
  --backfill-chunk-days 14
```

This live path currently uses:

- Gamma API for active market discovery
- CLOB `prices-history` for token probability history, fetched in bounded chunks so longer retained windows do not trip the API span limit
- Data API `/oi` for open interest

The Polymarket provider currently favors transparency over sophistication:

- it scans active markets ordered by recent volume,
- infers target categories from market text,
- excludes asset-price and futures-style contracts that mostly mirror public market data,
- excludes social-activity tracker contracts that are outside the report's value proposition,
- aggregates sibling contracts at the event level during universe selection,
- lets each winning event contribute only a small number of child contracts into the scoring universe,
- fetches the tracked outcome history for binary markets,
- guarantees small category coverage floors, then fills the rest of the universe by the strongest markets overall,
- applies a soft category-share brake instead of forcing equal caps,
- and adapts those into `MarketSeries` objects for the detector.

Topic inference intentionally uses the market's own question/description/slug/event title, but not Polymarket's free-form event-context summary. That context can still be useful as a note for research, but it is too noisy to drive inclusion decisions on its own.

The local dataset is kept bounded by:

- storing only markets that passed the topical filter,
- keeping one resampled history stream per tracked outcome,
- pruning snapshots older than `snapshot_retention_days`,
- and dropping stale markets that have not been refreshed recently.

When you run `--source polymarket`, PMR first fetches the live delta or backfill chunk, writes it into SQLite, prunes old data, and then runs detection on the merged stored dataset. That keeps reports stable while avoiding repeated full-history pulls.

The current default operating point is a `125`-market scoring universe and up to `12` final anomaly candidates per run. Those numbers are intended to keep the queue broad enough to catch more medium-liquidity stories without overwhelming the downstream research layer.

When you request the optional detector Markdown report, PMR appends a diagnostics section that explains:

- how many pages and payloads were scanned,
- how many markets matched the topical filter,
- how many histories were requested,
- how many markets survived selection into storage,
- and the main rejection reasons for excluded markets.

One important consequence of the event-aware selector is that multi-outcome stories such as election winners or next-prime-minister markets no longer need one single child contract to rank highly on its own. The event can win a universe slot on aggregated liquidity, then PMR keeps only a small number of sibling contracts from that event in the scoring universe.

If you want the same information as structured JSON, use `--output-scan-json`.

The detector also dedupes related market variants before the final ranked output:

- same-story election winner contracts,
- threshold ladders such as multiple gold or oil strike markets,
- and other markets that share the same Polymarket event title.

This keeps the research queue closer to one item per story instead of one item per contract.

Each final anomaly now also carries lightweight editorial annotations:

- `story_type_hint`: `live_repricing`, `resolved_surprise`, or `late_stage_resolution`
- `distance_from_extremes`: how close the closing probability is to 0% or 100%
- `entered_extreme_zone`: whether the market touched an extreme zone during the detection window
- `related_market_ids` / `related_market_questions`: loose same-story context for the downstream composer or research agent

The research JSON export uses those hints to produce one research job per final candidate, with:

- a primary market,
- related market variants for context,
- a short investigation brief,
- and the raw detector features that explain why the market was flagged.

## Run The Research Layer

The detector stage now stops at `research-inputs.json`. The research stage is a separate CLI that consumes those jobs and writes the canonical structured results:

```bash
uv run python -m pmr.research_cli \
  --input-json out/research-inputs.json \
  --db-path data/research.sqlite3 \
  --output-results-json out/research-results.json
```

Or with the package script:

```bash
uv run pmr-research \
  --input-json out/research-inputs.json \
  --output-results-json out/research-results.json
```

The xAI SDK-backed research runner expects:

- `XAI_API_KEY`
- optional `PMR_XAI_MODEL` (defaults to `grok-4.20-reasoning-latest`)
- optional `XAI_BASE_URL`

PMR now assumes the Grok ecosystem for the research stage. The default adapter uses the official `xai-sdk`, selects the best available reasoning model from a short Grok-first preference list, and uses built-in `x_search` plus `web_search` for retrieval.

The research cache is explicitly bounded:

- only normalized evidence metadata plus short excerpts are stored
- raw evidence expires after 30 days by default
- synthesized results expire after 90 days by default
- only the two most recent cached versions per job/provider are retained
- oldest cached batches are pruned if the SQLite file exceeds its size cap

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

On top of the anomaly score, PMR now adds lightweight editorial metadata. These hints are not used to suppress markets yet; they exist to help the downstream research/editor layer prioritize:

- `live_repricing`: material move, but the market still looks unresolved
- `resolved_surprise`: the market moved into an extreme zone from a meaningfully uncertain starting point
- `late_stage_resolution`: the market mostly confirmed an outcome that was already leaning strongly one way

## Architecture

Core modules:

- `pmr/config.py`: runtime thresholds and category rules.
- `pmr/market_filters.py`: explicit universe-level exclusion rules.
- `pmr/models.py`: shared dataclasses for the pipeline.
- `pmr/detector.py`: weekly event extraction, baseline normalization, and ranking.
- `pmr/polymarket.py`: public Polymarket HTTP client.
- `pmr/providers.py`: data-provider interfaces plus JSON, static, and Polymarket implementations.
- `pmr/research_payloads.py`: JSON serialization for research jobs and research results.
- `pmr/research_engine.py`: query planning, evidence ranking, caching, and synthesis orchestration.
- `pmr/research_store.py`: bounded SQLite cache for normalized evidence and structured research results.
- `pmr/research_xai.py`: xAI SDK-backed X-first research source and synthesizer.
- `pmr/story_groups.py`: story-family keys used for deduping related markets.
- `pmr/storage.py`: SQLite snapshot persistence, bounded retention, and stored-data loading.
- `pmr/universe_selection.py`: soft category-floor and overflow-based universe selection.
- `pmr/pipeline.py`: detection-stage orchestration.
- `pmr/reporting.py`: detector-only Markdown report rendering.
- `pmr/sample_data.py`: deterministic local demo dataset.

## Planned Next Steps

The highest-value next milestones are:

1. Validate the xAI SDK-backed research runner on real jobs and tune the evidence prompts around X-vs-web balance.
2. Add a historical evaluation loop so thresholds and exclusion rules can be calibrated against real weeks of Polymarket behavior.
3. Refine story-family clustering so closely related geopolitical and election markets can be grouped more intelligently before final composition.
4. Add scheduling, delivery, and final newsletter/editor composition once the research output is stable.

## Testing

```bash
uv run python -m unittest discover -s tests
```
