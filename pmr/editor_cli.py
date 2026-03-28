from __future__ import annotations

import argparse
import json
from pathlib import Path

from pmr.editor_engine import EditorEngine, HeuristicEditorComposer
from pmr.editor_payloads import build_weekly_report_payload, load_editor_story_packets_from_files
from pmr.editor_reporting import render_editor_decisions_markdown, render_weekly_report_markdown
from pmr.editor_xai import XaiEditorComposer
from pmr.research_cli import _load_dotenv_if_present


PROMPT_VERSION = "pmr_editor_v1"
PROVIDER_NAME = "xai_sdk_editor_multi_agent"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the PMR editor/composer layer on story-development outputs."
    )
    parser.add_argument(
        "--input-results-json",
        type=Path,
        default=Path("out/research-results.json"),
        help="Path to the story-development outputs JSON.",
    )
    parser.add_argument(
        "--input-research-json",
        type=Path,
        default=Path("out/research-inputs.json"),
        help="Path to the richer research-input payload used to supply market context to the editor.",
    )
    parser.add_argument(
        "--output-report-json",
        type=Path,
        default=Path("out/weekly-report.json"),
        help="Path to write the canonical structured weekly report JSON.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("out/final-report.md"),
        help="Path to write the reader-facing final report markdown.",
    )
    parser.add_argument(
        "--output-decisions-markdown",
        type=Path,
        default=Path("out/editor-decisions.md"),
        help="Path to write the separate editor decision log markdown.",
    )
    parser.add_argument(
        "--provider",
        choices=("xai", "heuristic"),
        default="xai",
        help="Composer backend to use. Heuristic is intended for tests and local fallback.",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print the final report markdown to stdout after rendering.",
    )
    args = parser.parse_args()
    _load_dotenv_if_present(Path(".env"))

    stories = load_editor_story_packets_from_files(
        research_results_path=args.input_results_json,
        research_inputs_path=args.input_research_json,
    )
    composer = _build_composer(args.provider)
    provider_name = PROVIDER_NAME if args.provider == "xai" else "heuristic_editor"
    engine = EditorEngine(
        composer=composer,
        provider_name=provider_name,
        prompt_version=PROMPT_VERSION,
    )
    report = engine.run(stories)

    report_json = json.dumps(build_weekly_report_payload(report), indent=2)
    report_markdown = render_weekly_report_markdown(report)
    decisions_markdown = render_editor_decisions_markdown(report)

    _write_text(args.output_report_json, report_json)
    _write_text(args.output_markdown, report_markdown)
    _write_text(args.output_decisions_markdown, decisions_markdown)

    if args.print_markdown:
        print(report_markdown)

    print(
        "PMR editor run complete: "
        f"{report.included_story_count} included, "
        f"{report.merged_story_count} merged, "
        f"{report.excluded_story_count} excluded."
    )
    return 0


def _build_composer(provider: str):
    if provider == "heuristic":
        return HeuristicEditorComposer()
    return XaiEditorComposer.from_env()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


if __name__ == "__main__":
    raise SystemExit(main())
