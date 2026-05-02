from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from time_dialogue.config import load_run_plan
from time_dialogue.script_builder import ScriptBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build script drafts from selected highlights.")
    parser.add_argument("highlights", nargs="*", help="Highlight JSON files. Defaults to records/selected/*.json.")
    parser.add_argument("--run-plan", default="configs/run_plan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.highlights]
    if not paths:
        paths = sorted((ROOT / "records" / "selected").glob("*.json"))
    paths = [path if path.is_absolute() else ROOT / path for path in paths]

    run_plan = load_run_plan(ROOT / args.run_plan)
    builder = ScriptBuilder(ROOT, run_plan)
    for path in paths:
        output_path = builder.build(path)
        print(output_path)


if __name__ == "__main__":
    main()
