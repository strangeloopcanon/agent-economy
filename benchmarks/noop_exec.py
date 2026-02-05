from __future__ import annotations


def main() -> int:
    # No-op worker used for smoke tests (should not win any tasks).
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
