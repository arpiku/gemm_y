# SQLite schema + query layer for the gemm_y benchmark dashboard.
# Pure functions, no Dash dependency. The DB lives at db/gemm_y.db
# (tracked in git, declared binary in .gitattributes). CSVs in results/
# stay gitignored (regenerable); the DB is the source of truth.
#
# Every measurement in the DB passed accuracy validation by construction:
# failed kernels (rel_err > tol) are skipped at the Profiler level before
# CSV write, so they never reach ingest. There is no `pass` column.
# `is_cublas` is derived in Python at query time (kernel_name == 'cublas').

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "gemm_y.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY,
    ingested_at  TEXT NOT NULL,
    git_sha      TEXT,
    label        TEXT,
    arch         TEXT NOT NULL,
    dtype        TEXT NOT NULL,
    source_csv   TEXT NOT NULL,
    source_meta  TEXT NOT NULL,
    warmup_iters INTEGER,
    timed_iters  INTEGER,
    tol          REAL,
    sweep_sizes  TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
    run_id               INTEGER NOT NULL,
    n                    INTEGER NOT NULL,
    kernel_name          TEXT NOT NULL,
    kernel_desc          TEXT NOT NULL,
    h2d_ns               REAL,
    kernel_min_ns        REAL,
    kernel_median_ns     REAL,
    d2h_ns               REAL,
    ref_kernel_min_ns    REAL,
    ref_kernel_median_ns REAL,
    max_abs_err          REAL,
    max_rel_err          REAL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_meas_run    ON measurements(run_id);
CREATE INDEX IF NOT EXISTS idx_meas_kernel ON measurements(kernel_name);
CREATE INDEX IF NOT EXISTS idx_runs_arch_dtype ON runs(arch, dtype);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection and ensure the schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def cursor(conn: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent schema creation."""
    conn.executescript(SCHEMA)
    conn.commit()


def insert_run(
    conn: sqlite3.Connection,
    *,
    ingested_at: str,
    git_sha: Optional[str],
    label: Optional[str],
    arch: str,
    dtype: str,
    source_csv: str,
    source_meta: str,
    warmup_iters: Optional[int],
    timed_iters: Optional[int],
    tol: Optional[float],
    sweep_sizes: Optional[str],
) -> int:
    """Insert a run row and return its id."""
    with cursor(conn) as cur:
        cur.execute(
            """
            INSERT INTO runs (
                ingested_at, git_sha, label, arch, dtype,
                source_csv, source_meta,
                warmup_iters, timed_iters, tol, sweep_sizes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ingested_at, git_sha, label, arch, dtype,
                source_csv, source_meta,
                warmup_iters, timed_iters, tol, sweep_sizes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_measurement(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    n: int,
    kernel_name: str,
    kernel_desc: str,
    h2d_ns: Optional[float],
    kernel_min_ns: Optional[float],
    kernel_median_ns: Optional[float],
    d2h_ns: Optional[float],
    ref_kernel_min_ns: Optional[float],
    ref_kernel_median_ns: Optional[float],
    max_abs_err: Optional[float],
    max_rel_err: Optional[float],
) -> None:
    with cursor(conn) as cur:
        cur.execute(
            """
            INSERT INTO measurements (
                run_id, n, kernel_name, kernel_desc,
                h2d_ns, kernel_min_ns, kernel_median_ns, d2h_ns,
                ref_kernel_min_ns, ref_kernel_median_ns,
                max_abs_err, max_rel_err
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, n, kernel_name, kernel_desc,
                h2d_ns, kernel_min_ns, kernel_median_ns, d2h_ns,
                ref_kernel_min_ns, ref_kernel_median_ns,
                max_abs_err, max_rel_err,
            ),
        )
    conn.commit()


def run_exists(
    conn: sqlite3.Connection,
    *,
    source_csv: str,
    source_meta: str,
    git_sha: Optional[str],
) -> bool:
    """Idempotency guard: True if this exact (csv, meta, sha) was ingested."""
    with cursor(conn) as cur:
        cur.execute(
            """
            SELECT 1 FROM runs
             WHERE source_csv = ? AND source_meta = ? AND git_sha IS ?
            LIMIT 1
            """,
            (source_csv, source_meta, git_sha),
        )
        return cur.fetchone() is not None


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All runs, newest first, with kernel count per run."""
    with cursor(conn) as cur:
        cur.execute(
            """
            SELECT r.id, r.ingested_at, r.git_sha, r.label,
                   r.arch, r.dtype, r.source_csv, r.source_meta,
                   r.warmup_iters, r.timed_iters, r.tol, r.sweep_sizes,
                   (SELECT COUNT(*) FROM measurements m WHERE m.run_id = r.id)
                       AS kernel_count
              FROM runs r
             ORDER BY r.ingested_at DESC, r.id DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


def fetch_measurements(
    conn: sqlite3.Connection,
    *,
    run_ids: Optional[list[int]] = None,
    archs: Optional[list[str]] = None,
    dtypes: Optional[list[str]] = None,
    kernel_classes: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Filtered measurement join. `kernel_classes` is a subset of
    {'cublas', 'custom'}; derived from kernel_name at query time."""
    query = """
        SELECT m.run_id, m.n, m.kernel_name, m.kernel_desc,
               m.h2d_ns, m.kernel_min_ns, m.kernel_median_ns, m.d2h_ns,
               m.ref_kernel_min_ns, m.ref_kernel_median_ns,
               m.max_abs_err, m.max_rel_err,
               r.ingested_at, r.git_sha, r.label, r.arch, r.dtype,
               r.tol, r.warmup_iters, r.timed_iters
          FROM measurements m
          JOIN runs r ON r.id = m.run_id
         WHERE 1=1
    """
    args: list[Any] = []
    if run_ids:
        query += " AND m.run_id IN (%s)" % ",".join("?" * len(run_ids))
        args.extend(run_ids)
    if archs:
        query += " AND r.arch IN (%s)" % ",".join("?" * len(archs))
        args.extend(archs)
    if dtypes:
        query += " AND r.dtype IN (%s)" % ",".join("?" * len(dtypes))
        args.extend(dtypes)
    if kernel_classes:
        # kernel_name == 'cublas' -> cublas; everything else -> custom
        clauses = []
        if "cublas" in kernel_classes:
            clauses.append("m.kernel_name = 'cublas'")
        if "custom" in kernel_classes:
            clauses.append("m.kernel_name != 'cublas'")
        query += " AND (%s)" % " OR ".join(clauses)
    query += " ORDER BY r.ingested_at, r.id, m.n, m.kernel_name"
    with cursor(conn) as cur:
        cur.execute(query, args)
        return [dict(row) for row in cur.fetchall()]


def distinct(
    conn: sqlite3.Connection, column: str, table: str = "runs"
) -> list[str]:
    """Distinct values of a column, sorted."""
    # column/table are internal, not user input — safe to interpolate.
    with cursor(conn) as cur:
        cur.execute(
            f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}"
        )
        return [row[0] for row in cur.fetchall()]
