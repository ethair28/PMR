from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pmr.research_engine import ResearchEngine
from pmr.research_payloads import (
    build_research_results_payload,
    load_research_jobs_from_file,
)
from pmr.research_store import ResearchCacheConfig, ResearchStore
from pmr.research_xai import XaiResearchSource, XaiResearchSynthesizer


PROMPT_VERSION = "pmr_research_v2_xai_sdk"
PROVIDER_NAME = "xai_sdk_x_first"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the PMR research layer on a previously exported research-jobs JSON file."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("out/research-inputs.json"),
        help="Path to the research-input payload exported by the detection stage.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/research.sqlite3"),
        help="SQLite path for cached evidence and synthesized research results.",
    )
    parser.add_argument(
        "--output-results-json",
        type=Path,
        default=Path("out/research-results.json"),
        help="Path to write the structured research results JSON.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Optional cap on how many jobs from the input file to process.",
    )
    parser.add_argument(
        "--job-ids",
        nargs="+",
        help="Optional explicit list of job IDs to process.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached entries and rerun the selected jobs live.",
    )
    parser.add_argument(
        "--cache-max-versions",
        type=int,
        default=2,
        help="Maximum number of cached versions to retain per job/provider pair.",
    )
    parser.add_argument(
        "--evidence-retention-days",
        type=int,
        default=30,
        help="How long to retain raw normalized evidence in the research cache.",
    )
    parser.add_argument(
        "--result-retention-days",
        type=int,
        default=90,
        help="How long to retain synthesized research results in the cache.",
    )
    parser.add_argument(
        "--max-evidence-items-per-job",
        type=int,
        default=50,
        help="Maximum number of normalized evidence items to persist per completed job.",
    )
    parser.add_argument(
        "--max-excerpt-chars",
        type=int,
        default=2_000,
        help="Maximum excerpt length to store per evidence item.",
    )
    parser.add_argument(
        "--cache-max-megabytes",
        type=int,
        default=512,
        help="Maximum allowed SQLite cache size before oldest batches are pruned.",
    )
    args = parser.parse_args()
    _load_dotenv_if_present(Path(".env"))

    jobs = load_research_jobs_from_file(args.input_json)
    if args.job_ids:
        allowed_ids = set(args.job_ids)
        jobs = tuple(job for job in jobs if job.job_id in allowed_ids)

    cache_config = ResearchCacheConfig(
        max_versions_per_job_provider=args.cache_max_versions,
        evidence_retention_days=args.evidence_retention_days,
        result_retention_days=args.result_retention_days,
        max_evidence_items_per_job=args.max_evidence_items_per_job,
        max_excerpt_chars=args.max_excerpt_chars,
        max_database_megabytes=args.cache_max_megabytes,
    )
    store = ResearchStore(path=args.db_path, cache_config=cache_config)
    engine = ResearchEngine(
        source=XaiResearchSource.from_env(),
        synthesizer=XaiResearchSynthesizer.from_env(),
        provider_name=PROVIDER_NAME,
        prompt_version=PROMPT_VERSION,
        store=store,
        max_evidence_items=cache_config.max_evidence_items_per_job,
    )
    batch = engine.run_batch(jobs, refresh=args.refresh, max_jobs=args.max_jobs)

    _write_text(args.output_results_json, json.dumps(build_research_results_payload(batch), indent=2))

    print(
        "PMR research run complete: "
        f"{batch.processed_jobs} processed, "
        f"{batch.cached_jobs} cached, "
        f"{batch.failed_jobs} failed."
    )
    return 0


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _load_dotenv_if_present(path: Path) -> None:
    """Load a simple repo-local .env file into the process environment.

    This loader is intentionally small and permissive enough for local development:
    - ignores blank lines and comments
    - trims surrounding whitespace around keys and values
    - strips matching single/double quotes around values
    - does not override already-exported environment variables
    """

    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
