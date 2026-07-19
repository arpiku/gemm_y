#!/usr/bin/env python3
"""Ingest a benchmark CSV + .meta sidecar into db/gemm_y.db.

Usage:
    python scripts/ingest.py results/bench_sm_120_bf16.csv [--label "..."]
    python scripts/ingest.py results/bench_sm_120_bf16.csv --force

The .meta sidecar is auto-discovered by replacing the .csv extension with
.meta. Refuses to re-ingest the same (source_csv, source_meta, git_sha)
tuple unless --force is given.

Schema and query layer live in db.py. The DB is the source of truth for
the dashboard; CSVs are regenerable and stay gitignored.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import db


def iso8601_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_short_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_meta(meta_path: Path) -> dict[str, object]:
    """Parse the key=value .meta sidecar.

    Multiple `kernel=` lines are collected into a list under key 'kernels',
    each as a (name, desc) tuple split on the first '|'.
    """
    meta: dict[str, object] = {}
    kernels: list[tuple[str, str]] = []
    with meta_path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "kernel":
                name, sep, desc = val.partition("|")
                kernels.append((name.strip(), desc.strip() if sep else ""))
            else:
                meta[key] = val
    meta["kernels"] = kernels
    return meta


def _to_int(v: object) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(str(v))
    except ValueError:
        return None


def _to_float(v: object) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(str(v))
    except ValueError:
        return None


def ingest(csv_path: Path, label: Optional[str], force: bool) -> int:
    meta_path = csv_path.with_suffix(".meta")
    if not csv_path.exists():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 2
    if not meta_path.exists():
        print(f"error: .meta sidecar not found: {meta_path}", file=sys.stderr)
        return 2

    meta = parse_meta(meta_path)
    arch = str(meta.get("arch", ""))
    dtype = str(meta.get("dtype", ""))
    warmup = _to_int(meta.get("warmup_iters"))
    timed = _to_int(meta.get("timed_iters"))
    tol = _to_float(meta.get("tol"))
    sweep_sizes = str(meta.get("sweep_sizes", "")) or None

    if not arch or not dtype:
        print(
            f"error: .meta missing arch/dtype (arch={arch!r}, dtype={dtype!r})",
            file=sys.stderr,
        )
        return 2

    sha = git_short_sha()
    source_csv = str(csv_path)
    source_meta = str(meta_path)

    conn = db.connect()
    if not force and db.run_exists(
        conn, source_csv=source_csv, source_meta=source_meta, git_sha=sha
    ):
        print(
            f"skip: already ingested (csv={source_csv}, meta={source_meta}, "
            f"sha={sha}). Use --force to re-ingest.",
            file=sys.stderr,
        )
        return 1

    run_id = db.insert_run(
        conn,
        ingested_at=iso8601_now(),
        git_sha=sha,
        label=label,
        arch=arch,
        dtype=dtype,
        source_csv=source_csv,
        source_meta=source_meta,
        warmup_iters=warmup,
        timed_iters=timed,
        tol=tol,
        sweep_sizes=sweep_sizes,
    )

    # CSV schema:
    # arch,dtype,N,kernel_name,kernel_desc,h2d_ns,kernel_min_ns,
    # kernel_median_ns,d2h_ns,ref_kernel_min_ns,ref_kernel_median_ns,
    # max_abs_err,max_rel_err
    expected = [
        "arch", "dtype", "N", "kernel_name", "kernel_desc",
        "h2d_ns", "kernel_min_ns", "kernel_median_ns", "d2h_ns",
        "ref_kernel_min_ns", "ref_kernel_median_ns",
        "max_abs_err", "max_rel_err",
    ]
    n_rows = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in expected if c not in (reader.fieldnames or [])]
        if missing:
            print(
                f"error: CSV missing columns: {missing}", file=sys.stderr
            )
            return 2
        for row in reader:
            db.insert_measurement(
                conn,
                run_id,
                n=int(row["N"]),
                kernel_name=row["kernel_name"],
                kernel_desc=row["kernel_desc"],
                h2d_ns=_to_float(row["h2d_ns"]),
                kernel_min_ns=_to_float(row["kernel_min_ns"]),
                kernel_median_ns=_to_float(row["kernel_median_ns"]),
                d2h_ns=_to_float(row["d2h_ns"]),
                ref_kernel_min_ns=_to_float(row["ref_kernel_min_ns"]),
                ref_kernel_median_ns=_to_float(row["ref_kernel_median_ns"]),
                max_abs_err=_to_float(row["max_abs_err"]),
                max_rel_err=_to_float(row["max_rel_err"]),
            )
            n_rows += 1

    print(
        f"ingested run id={run_id} arch={arch} dtype={dtype} "
        f"sha={sha} rows={n_rows} label={label!r}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="path to bench_*.csv")
    ap.add_argument("--label", default=None, help="optional run label")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-ingest even if (csv, meta, sha) already present",
    )
    args = ap.parse_args()
    return ingest(args.csv, args.label, args.force)


if __name__ == "__main__":
    sys.exit(main())
