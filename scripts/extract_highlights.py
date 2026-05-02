from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from time_dialogue.config import load_model_configs, load_run_plan
from time_dialogue.extractor import HighlightExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract highlight fragments from raw dialogue records.")
    parser.add_argument("records", nargs="*", help="Raw record JSON files. Defaults to records/raw/*.json.")
    parser.add_argument("--models", default="configs/models.json")
    parser.add_argument("--run-plan", default="configs/run_plan.json")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-llm-editor", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not call remote editor API.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    record_paths = [Path(item) for item in args.records]
    if not record_paths:
        record_paths = sorted((ROOT / "records" / "raw").glob("*.json"))
    record_paths = [path if path.is_absolute() else ROOT / path for path in record_paths]

    models = load_model_configs(ROOT / args.models)
    run_plan = load_run_plan(ROOT / args.run_plan)
    extractor = HighlightExtractor(ROOT, models, run_plan, dry_run=args.dry_run)
    for record_path in record_paths:
        output_path = await extractor.extract(
            record_path,
            use_llm_editor=args.use_llm_editor,
            top_k=args.top_k,
        )
        print(output_path)


if __name__ == "__main__":
    asyncio.run(main())
