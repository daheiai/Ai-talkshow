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
from time_dialogue.runner import DialogueRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run old/new AI time-gap dialogues.")
    parser.add_argument("--models", default="configs/models.json")
    parser.add_argument("--cards", "--topics", dest="cards", default="configs/cards.json")
    parser.add_argument("--run-plan", default="configs/run_plan.json")
    parser.add_argument("--old", action="append", help="Old model id. Can be repeated.")
    parser.add_argument("--new", action="append", help="New model id. Can be repeated.")
    parser.add_argument("--card", "--topic", dest="card", action="append", help="Card id. Can be repeated.")
    parser.add_argument("--limit-cards", "--limit-topics", dest="limit_cards", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None, help="Override repeat count for each card.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call remote APIs.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    models = load_model_configs(ROOT / args.models)
    cards = load_dialogue_cards(ROOT / args.cards)
    run_plan = load_run_plan(ROOT / args.run_plan)
    runner = DialogueRunner(ROOT, models, cards, run_plan, dry_run=args.dry_run)
    output_paths = await runner.run_all(
        old_model_ids=args.old,
        new_model_ids=args.new,
        card_ids=args.card,
        limit_cards=args.limit_cards,
        repeat_override=args.repeat,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    asyncio.run(main())
