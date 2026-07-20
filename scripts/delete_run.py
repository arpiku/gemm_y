#!/usr/bin/env python3
"""Manage runs in db/gemm_y.db.

Subcommands:
    python scripts/delete_run.py list
        Print all runs (id, ingested_at, git_sha, label, arch, dtype,
        measurement count). Same columns as the dashboard's Run History
        tab, plus the count.

    python scripts/delete_run.py delete <id> [<id> ...]
        Delete the named run(s) and their measurements (cascading delete
        via FOREIGN KEY ... REFERENCES runs(id)). Confirm prompt before
        delete unless --force.

        Refuses to delete a run if it is the only one for its (arch, dtype)
        pair unless --force (guard against wiping the last baseline).

The DB is the source of truth for the dashboard; CSVs in results/ are
regenerable and stay gitignored. Deleted runs cannot be recovered from
CSVs alone — re-ingest with `python scripts/ingest.py <csv> --force`.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

import db


def _format_perf_pct(val) -> str:
    if val is None:
        return ""
    try:
        return f"{float(val):+.1f}%"
    except (TypeError, ValueError):
        return ""


def cmd_list() -> int:
    conn = db.connect()
    try:
        runs = db.list_runs(conn)
    finally:
        conn.close()
    if not runs:
        print("(no runs in db)")
        return 0
    # Column widths chosen for readability; arch/dtype are short.
    header = (
        f"{'id':>4}  {'ingested_at':<20}  {'git_sha':<10}  "
        f"{'label':<24}  {'arch':<8}  {'dtype':<6}  {'meas':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r['id']:>4}  {r['ingested_at']:<20}  "
            f"{(r['git_sha'] or ''):<10}  "
            f"{(r['label'] or '')[:24]:<24}  "
            f"{r['arch']:<8}  {r['dtype']:<6}  "
            f"{r['kernel_count']:>5}"
        )
    print(f"\n{len(runs)} run(s).")
    return 0


def _is_last_for_arch_dtype(conn: sqlite3.Connection, run_id: int) -> bool:
    """True if the run is the only one for its (arch, dtype) pair."""
    with db.cursor(conn) as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM runs
             WHERE arch = (SELECT arch FROM runs WHERE id = ?)
               AND dtype = (SELECT dtype FROM runs WHERE id = ?)
            """,
            (run_id, run_id),
        )
        return cur.fetchone()[0] == 1


def _run_summary(conn: sqlite3.Connection, run_id: int) -> dict | None:
    with db.cursor(conn) as cur:
        cur.execute(
            """
            SELECT id, ingested_at, git_sha, label, arch, dtype,
                   (SELECT COUNT(*) FROM measurements m WHERE m.run_id = r.id)
                       AS kernel_count
              FROM runs r
             WHERE id = ?
            """,
            (run_id,),
        )
        row = cur.fetchone()
        return dict(row) if row is not None else None


def _delete_one(conn: sqlite3.Connection, run_id: int) -> int:
    """Delete a single run + its measurements. Returns measurement rows
    deleted (for the success message). Caller commits."""
    with db.cursor(conn) as cur:
        cur.execute(
            "DELETE FROM measurements WHERE run_id = ?", (run_id,)
        )
        meas_deleted = cur.rowcount
        cur.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return meas_deleted


def cmd_delete(ids: list[int], force: bool) -> int:
    conn = db.connect()
    try:
        # Pre-flight: resolve all runs, surface unknown ids up front.
        summaries: list[dict] = []
        unknown: list[int] = []
        last_run_warnings: list[str] = []
        for rid in ids:
            s = _run_summary(conn, rid)
            if s is None:
                unknown.append(rid)
                continue
            summaries.append(s)
            if _is_last_for_arch_dtype(conn, rid):
                last_run_warnings.append(
                    f"  id={rid} arch={s['arch']} dtype={s['dtype']} "
                    f"(last run for this arch/dtype)"
                )

        if unknown:
            print(
                f"error: no such run id(s): {unknown}. "
                "Run `delete_run.py list` to see valid ids.",
                file=sys.stderr,
            )
            return 2

        if last_run_warnings:
            print("Refusing to delete the last run for an (arch, dtype) pair:")
            for w in last_run_warnings:
                print(w)
            if not force:
                print(
                    "Re-run with --force to delete anyway. Aborting.",
                    file=sys.stderr,
                )
                return 1
            print("--force given; proceeding.")

        if not force:
            print("About to delete:")
            for s in summaries:
                print(
                    f"  id={s['id']} ingested_at={s['ingested_at']} "
                    f"arch={s['arch']} dtype={s['dtype']} "
                    f"label={s['label']!r} measurements={s['kernel_count']}"
                )
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1

        total_meas = 0
        for s in summaries:
            total_meas += _delete_one(conn, s["id"])
        conn.commit()

        # Remaining run count for the success message.
        with db.cursor(conn) as cur:
            cur.execute("SELECT COUNT(*) FROM runs")
            remaining = cur.fetchone()[0]

        print(
            f"deleted {len(summaries)} run(s), {total_meas} measurement(s); "
            f"{remaining} run(s) remaining."
        )
        return 0
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all runs")

    d = sub.add_parser("delete", help="delete one or more runs")
    d.add_argument("ids", type=int, nargs="+", help="run id(s) to delete")
    d.add_argument(
        "--force",
        action="store_true",
        help="skip the confirm prompt and the last-run-for-arch-dtype guard",
    )

    args = ap.parse_args()
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "delete":
        return cmd_delete(args.ids, args.force)
    return 2


if __name__ == "__main__":
    sys.exit(main())
