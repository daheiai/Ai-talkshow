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

from time_dialogue.config import load_dialogue_cards, load_model_configs, load_run_plan
from time_dialogue.extractor import HighlightExtractor
from time_dialogue.runner import DialogueRunner
from time_dialogue.script_builder import ScriptBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dialogue -> highlights -> script in one command.")
    parser.add_argument("--models", default="configs/models.json")
    parser.add_argument("--cards", "--topics", dest="cards", default="configs/cards.json")
    parser.add_argument("--run-plan", default="configs/run_plan.json")
    parser.add_argument("--old", action="append")
    parser.add_argument("--new", action="append")
    parser.add_argument("--card", "--topic", dest="card", action="append")
    parser.add_argument("--limit-cards", "--limit-topics", dest="limit_cards", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-llm-editor", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    models = load_model_configs(ROOT / args.models)
    cards = load_dialogue_cards(ROOT / args.cards)
    run_plan = load_run_plan(ROOT / args.run_plan)

    runner = DialogueRunner(ROOT, models, cards, run_plan, dry_run=args.dry_run)
    raw_paths = await runner.run_all(
        old_model_ids=args.old,
        new_model_ids=args.new,
        card_ids=args.card,
        limit_cards=args.limit_cards,
        repeat_override=args.repeat,
    )

    extractor = HighlightExtractor(ROOT, models, run_plan, dry_run=args.dry_run)
    builder = ScriptBuilder(ROOT, run_plan)
    for raw_path in raw_paths:
        highlight_path = await extractor.extract(
            raw_path,
            use_llm_editor=args.use_llm_editor,
            top_k=args.top_k,
        )
        script_path = builder.build(highlight_path)
        print(f"raw: {raw_path}")
        print(f"highlights: {highlight_path}")
        print(f"script: {script_path}")


if __name__ == "__main__":
    asyncio.run(main())
