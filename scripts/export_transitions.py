from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_economy.learning_trace import extract_attempt_transitions_from_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-attempt transitions from a run ledger for RL/offline learning."
    )
    parser.add_argument("run_dir", type=Path, help="run directory containing ledger.jsonl")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="emit one JSON object per line (default: pretty JSON array)",
    )
    args = parser.parse_args()

    transitions = extract_attempt_transitions_from_run(run_dir=args.run_dir)
    if args.jsonl:
        for row in transitions:
            print(json.dumps(row, ensure_ascii=False))
        return

    print(json.dumps(transitions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
