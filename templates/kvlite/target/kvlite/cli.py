from __future__ import annotations

import argparse
from pathlib import Path

from kvlite.store import KVStore


def _load_or_new(db_path: Path) -> KVStore:
    if db_path.exists():
        return KVStore.load(db_path)
    return KVStore()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvlite")
    sub = parser.add_subparsers(dest="cmd", required=True)

    set_p = sub.add_parser("set")
    set_p.add_argument("key")
    set_p.add_argument("value")
    set_p.add_argument("--ttl", type=float, default=None)
    set_p.add_argument("--db", type=Path, required=True)

    get_p = sub.add_parser("get")
    get_p.add_argument("key")
    get_p.add_argument("--db", type=Path, required=True)

    keys_p = sub.add_parser("keys")
    keys_p.add_argument("--prefix", default=None)
    keys_p.add_argument("--db", type=Path, required=True)

    del_p = sub.add_parser("delete")
    del_p.add_argument("key")
    del_p.add_argument("--db", type=Path, required=True)

    args = parser.parse_args(argv)

    store = _load_or_new(args.db)

    if args.cmd == "set":
        store.set(args.key, args.value, ttl=args.ttl)
        store.dump(args.db)
        return 0

    if args.cmd == "get":
        value = store.get(args.key)
        if value is None:
            return 1
        print(value)
        return 0

    if args.cmd == "keys":
        for key in store.keys(prefix=args.prefix):
            print(key)
        return 0

    if args.cmd == "delete":
        deleted = store.delete(args.key)
        store.dump(args.db)
        return 0 if deleted else 1

    raise AssertionError(f"unhandled cmd: {args.cmd}")
