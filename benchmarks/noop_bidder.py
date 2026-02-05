from __future__ import annotations

import json
import sys


def main() -> int:
    _ = sys.stdin.read()  # ignore task list; never bid
    sys.stdout.write(json.dumps({"bids": []}) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
