from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from time_dialogue.director_console import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local human director console.")
    parser.add_argument("--models", default="configs/models.json")
    parser.add_argument("--cards", default="configs/cards.json")
    parser.add_argument("--run-plan", default="configs/run_plan.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true", help="Do not call remote model APIs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    import uvicorn

    app = create_app(
        ROOT,
        models_path=args.models,
        cards_path=args.cards,
        run_plan_path=args.run_plan,
        dry_run=args.dry_run,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
