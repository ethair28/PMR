from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pmr.config import MonitoringConfig
from pmr.pipeline import run_pipeline
from pmr.polymarket import HttpPolymarketClient
from pmr.providers import (
    JsonFileMarketDataProvider,
    MockResearchProvider,
    PolymarketMarketDataProvider,
    PolymarketScanDiagnostics,
    StoredMarketDataProvider,
    StaticMarketDataProvider,
)
from pmr.research_payloads import build_research_input_payload
from pmr.sample_data import build_sample_market_series
from pmr.storage import SnapshotStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect significant prediction-market repricing and draft a Markdown report."
    )
    parser.add_argument(
        "--source",
        choices=("sample", "json", "polymarket", "stored"),
        default="sample",
        help="Data source for the run.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional path to a JSON file containing markets and snapshots.",
    )
    parser.add_argument(
        "--detection-window-days",
        "--lookback-days",
        type=int,
        default=7,
        help="Detection window size for event extraction.",
    )
    parser.add_argument(
        "--preferred-baseline-window-days",
        type=int,
        default=90,
        help="Preferred amount of pre-window history used for normalization.",
    )
    parser.add_argument(
        "--min-abs-move-24h",
        type=float,
        default=0.05,
        help="Minimum 24h move in probability points.",
    )
    parser.add_argument(
        "--min-weekly-range",
        type=float,
        default=0.08,
        help="Minimum detection-window range in probability points.",
    )
    parser.add_argument(
        "--feature-interval-minutes",
        type=int,
        default=60,
        help="Snapshot interval used for feature extraction and Polymarket history fidelity.",
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
    parser.add_argument(
        "--polymarket-max-markets",
        type=int,
        default=100,
        help="Maximum number of topic-relevant market series to analyze.",
    )
    parser.add_argument(
        "--polymarket-page-size",
        type=int,
        default=100,
        help="Page size when scanning live Polymarket markets.",
    )
    parser.add_argument(
        "--polymarket-max-pages",
        type=int,
        default=20,
        help="Maximum number of Polymarket pages to scan.",
    )
    parser.add_argument(
        "--min-markets-per-category",
        type=int,
        default=5,
        help="Minimum coverage floor per target category before global overflow allocation.",
    )
    parser.add_argument(
        "--max-category-share",
        type=float,
        default=0.6,
        help="Soft dominance brake for one category as a share of the tracked universe.",
    )
    parser.add_argument(
        "--max-markets-per-category",
        type=int,
        help="Optional legacy hard cap per category. Defaults to disabled.",
    )
    parser.add_argument(
        "--polymarket-refresh-mode",
        choices=("full", "incremental", "backfill"),
        default="incremental",
        help="How live Polymarket fetches should use the existing snapshot store.",
    )
    parser.add_argument(
        "--incremental-overlap-minutes",
        type=int,
        default=180,
        help="Overlap used when refreshing the recent tail of a stored market.",
    )
    parser.add_argument(
        "--backfill-chunk-days",
        type=int,
        default=14,
        help="Older history chunk size to fetch per run in backfill mode.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/pmr.sqlite3"),
        help="SQLite path for persisted live snapshots.",
    )
    parser.add_argument(
        "--snapshot-retention-days",
        type=int,
        default=120,
        help="How long to keep persisted snapshots before pruning.",
    )
    parser.add_argument(
        "--stored-market-staleness-hours",
        type=int,
        default=24,
        help="How recent a stored market must be to be included in reports.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to write the Markdown report.",
    )
    parser.add_argument(
        "--output-research-json",
        type=Path,
        help="Optional path to write anomaly payloads for a downstream research agent.",
    )
    parser.add_argument(
        "--output-scan-json",
        type=Path,
        help="Optional path to write live Polymarket scan diagnostics as JSON.",
    )
    args = parser.parse_args()

    config = MonitoringConfig(
        detection_window_days=args.detection_window_days,
        preferred_baseline_window_days=args.preferred_baseline_window_days,
        snapshot_retention_days=args.snapshot_retention_days,
        stored_market_staleness_hours=args.stored_market_staleness_hours,
        feature_interval_minutes=args.feature_interval_minutes,
        min_abs_move_24h=args.min_abs_move_24h,
        min_weekly_range=args.min_weekly_range,
        min_volume_7d_usd=args.min_volume_7d,
        min_open_interest_usd=args.min_open_interest,
        min_markets_per_category=args.min_markets_per_category,
        max_category_share_of_universe=args.max_category_share,
        max_markets_per_category=args.max_markets_per_category,
    )

    history_days = max(
        config.snapshot_retention_days,
        config.detection_window_days + config.preferred_baseline_window_days + 7,
    )
    store = SnapshotStore(args.db_path)
    source_name = args.source
    polymarket_scan_diagnostics: PolymarketScanDiagnostics | None = None
    stored_market_count_after_refresh: int | None = None

    if args.source == "json":
        if args.input_json is None:
            parser.error("--input-json is required when --source json is used.")
        markets = tuple(JsonFileMarketDataProvider(args.input_json).list_market_series())
    elif args.source == "polymarket":
        store.initialize()
        existing_market_ids = store.list_market_ids()
        existing_bounds = store.get_snapshot_bounds(existing_market_ids)
        live_provider = PolymarketMarketDataProvider(
            client=HttpPolymarketClient(),
            max_markets=args.polymarket_max_markets,
            page_size=args.polymarket_page_size,
            max_pages=args.polymarket_max_pages,
            history_days=history_days,
            fidelity_minutes=config.feature_interval_minutes,
            target_categories=config.target_categories,
            category_aliases=config.category_aliases,
            min_markets_per_category=config.min_markets_per_category,
            max_category_share_of_universe=config.max_category_share_of_universe,
            max_markets_per_category=config.max_markets_per_category,
            refresh_mode=args.polymarket_refresh_mode,
            existing_snapshot_bounds=existing_bounds,
            incremental_overlap_minutes=args.incremental_overlap_minutes,
            backfill_chunk_days=args.backfill_chunk_days,
        )
        scan_result = live_provider.scan_market_series()
        polymarket_scan_diagnostics = scan_result.diagnostics
        fetched_markets = scan_result.markets
        store.upsert_market_series(fetched_markets, fetched_at=datetime.now(timezone.utc))
        store.prune(retention_days=config.snapshot_retention_days)
        markets = tuple(
            StoredMarketDataProvider(
                store=store,
                target_categories=config.target_categories,
                history_days=history_days,
                staleness_hours=config.stored_market_staleness_hours,
                max_markets=args.polymarket_max_markets,
                min_markets_per_category=config.min_markets_per_category,
                max_category_share_of_universe=config.max_category_share_of_universe,
                max_markets_per_category=config.max_markets_per_category,
            ).list_market_series()
        )
        stored_market_count_after_refresh = len(store.list_market_ids())
    elif args.source == "stored":
        store.initialize()
        markets = tuple(
            StoredMarketDataProvider(
                store=store,
                target_categories=config.target_categories,
                history_days=history_days,
                staleness_hours=config.stored_market_staleness_hours,
                max_markets=args.polymarket_max_markets,
                min_markets_per_category=config.min_markets_per_category,
                max_category_share_of_universe=config.max_category_share_of_universe,
                max_markets_per_category=config.max_markets_per_category,
            ).list_market_series()
        )
    else:
        markets = tuple(build_sample_market_series())

    result = run_pipeline(
        market_data_provider=StaticMarketDataProvider(markets),
        research_provider=MockResearchProvider(),
        config=config,
    )

    markdown_report = result.markdown_report
    if polymarket_scan_diagnostics is not None:
        markdown_report = _append_polymarket_scan_summary(
            markdown_report,
            diagnostics=polymarket_scan_diagnostics,
            stored_market_count_after_refresh=stored_market_count_after_refresh,
        )

    if args.output_markdown:
        _write_text(args.output_markdown, markdown_report)
    if args.output_research_json:
        research_payload = build_research_input_payload(
            events=tuple(item.event for item in result.event_reports),
            config=config,
            source_name=source_name,
        )
        _write_text(args.output_research_json, json.dumps(research_payload, indent=2))
    if args.output_scan_json and polymarket_scan_diagnostics is not None:
        _write_text(
            args.output_scan_json,
            json.dumps(
                _build_polymarket_scan_payload(
                    diagnostics=polymarket_scan_diagnostics,
                    stored_market_count_after_refresh=stored_market_count_after_refresh,
                ),
                indent=2,
            ),
        )

    print(markdown_report)
    return 0


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _append_polymarket_scan_summary(
    markdown_report: str,
    *,
    diagnostics: PolymarketScanDiagnostics,
    stored_market_count_after_refresh: int | None,
) -> str:
    lines = [
        markdown_report,
        "",
        "## Live Scan Diagnostics",
        "",
        f"- Pages scanned: {diagnostics.pages_scanned}",
        f"- Payloads seen: {diagnostics.payloads_seen}",
        f"- Unique markets seen: {diagnostics.unique_markets_seen}",
        f"- Topic matches: {diagnostics.topic_matches}",
        f"- Selected markets before storage: {diagnostics.selected_markets}",
        f"- History requests made: {diagnostics.history_requests}",
        f"- Open-interest fallbacks used: {diagnostics.open_interest_fallbacks}",
    ]
    if stored_market_count_after_refresh is not None:
        lines.append(f"- Stored markets after refresh/prune: {stored_market_count_after_refresh}")
    if diagnostics.per_category_selected:
        lines.append(
            "- Selected by category: "
            + ", ".join(f"{category}={count}" for category, count in diagnostics.per_category_selected.items())
        )
    if diagnostics.rejection_counts:
        lines.extend(["", "### Rejection Counts", ""])
        lines.extend(
            f"- {reason}: {count}"
            for reason, count in diagnostics.rejection_counts.items()
        )
    if diagnostics.sampled_rejections:
        lines.extend(["", "### Sampled Rejections", ""])
        for item in diagnostics.sampled_rejections:
            label = item.question or item.market_id or "Unknown market"
            detail = f" ({item.detail})" if item.detail else ""
            category = f", category={item.category}" if item.category else ""
            matched_terms = f", matched_terms={', '.join(item.matched_terms)}" if item.matched_terms else ""
            lines.append(f"- {label}: {item.reason}{category}{matched_terms}{detail}")
    return "\n".join(lines)


def _build_polymarket_scan_payload(
    *,
    diagnostics: PolymarketScanDiagnostics,
    stored_market_count_after_refresh: int | None,
) -> dict[str, object]:
    return {
        "summary": {
            "pages_scanned": diagnostics.pages_scanned,
            "payloads_seen": diagnostics.payloads_seen,
            "unique_markets_seen": diagnostics.unique_markets_seen,
            "topic_matches": diagnostics.topic_matches,
            "selected_markets": diagnostics.selected_markets,
            "history_requests": diagnostics.history_requests,
            "open_interest_fallbacks": diagnostics.open_interest_fallbacks,
            "stored_markets_after_refresh": stored_market_count_after_refresh,
            "per_category_selected": diagnostics.per_category_selected,
            "rejection_counts": diagnostics.rejection_counts,
        },
        "sampled_rejections": [
            {
                "market_id": item.market_id,
                "question": item.question,
                "reason": item.reason,
                "category": item.category,
                "matched_terms": list(item.matched_terms),
                "detail": item.detail,
            }
            for item in diagnostics.sampled_rejections
        ],
    }
