#!/usr/bin/env python3
"""Export db/gemm_y.db to JSONL (one line per measurement, run metadata joined).

The DB is the source of truth; JSONL is a regenerable view for human
inspection / diffs. Output defaults to stdout; use -o to write to a file
(e.g. db/dump.jsonl, which is gitignored).

Usage:
    python scripts/dump_db.py
    python scripts/dump_db.py -o db/dump.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import db


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output file (default: stdout)",
    )
    args = ap.parse_args()

    conn = db.connect()
    rows = db.fetch_measurements(conn)

    if args.out is None:
        for r in rows:
            sys.stdout.write(json.dumps(r) + "\n")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
