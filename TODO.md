# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
 decision rationale in `ARD.md`.

## Phase 2B.1 — Dashboard bug fix + polish

Goal: close the Dash 4.x callback bug (every interactive action returns
HTTP 500) and apply the aesthetic/functional polish surfaced during the
Phase 2B review. No C++ changes; Python only.

**Context:** Phase 2B shipped with `dash==4.4.0` installed, but the
callback in `scripts/server.py` uses the list-wrapped `Input` signature
(`[Input(...), Input(...), ...]`), which Dash 4.x interprets as a
wildcard multi-output. The callback returns a single component
(`dcc.Graph` / `dt.DataTable`), so Dash raises
`InvalidCallbackReturnValue` on every tab switch / filter change. The
initial `GET /` returns 200 (static layout), which masked the bug during
the agent's validation. The figure builders themselves are correct
(verified in isolation); only the callback wiring is broken.

### Bug fix (blocking)

- [x] **2B.1.1** `scripts/server.py`: change the `@app.callback`
  signature from list-wrapped to flat:
  ```python
  # Before (broken on Dash 4.x — interpreted as wildcard multi-output):
  @app.callback(
      Output("tab-content", "children"),
      [Input("tabs", "value"),
       Input("filter-arch", "value"),
       ...],
  )
  # After (flat — the Dash 2.x/3.x/4.x preferred form):
  @app.callback(
      Output("tab-content", "children"),
      Input("tabs", "value"),
      Input("filter-arch", "value"),
      Input("filter-dtype", "value"),
      Input("filter-class", "value"),
      Input("filter-runs", "value"),
      Input("filter-scale", "value"),
  )
  ```
  One-line change (drop the `[...]` around the `Input` args). See
  AGENTS.md "Python / Dash" convention for the rationale.

### Functional improvements

- [x] **2B.1.2** `scripts/server.py` `_accuracy_figure`: when
  `max_rel_err == 0.0` for all points (current state — deterministic
  fill produces bit-identical output), the log-log plot is degenerate
  (`log(0) = -inf`). Clamp the displayed y-values to a small epsilon
  (e.g. `1e-15`) for display only — do not mutate the underlying data.
  Alternatively, default the accuracy tab to `linear` scale. Pick one.
- [x] **2B.1.3** `scripts/server.py` `_runs_table`: drop the unused
  `rows` parameter (the function queries the DB directly via
  `db.list_runs(db.connect())`). Also close the connection
  (`with db.connect() as conn:` or explicit `conn.close()`).
- [x] **2B.1.4** `scripts/server.py` `render_tab`: the callback opens a
  new `db.connect()` per invocation. Acceptable for a local dev tool,
  but the sidebar's run dropdown is populated once at `build_app()` time
  and goes stale if you ingest new data while the server is running.
  Either (a) document "restart server after ingesting new runs" in the
  `--help` text, or (b) add a "Refresh runs" button that re-queries.
  Pick (a) for now — simpler.

### Aesthetic improvements

- [x] **2B.1.5** `scripts/server.py` run dropdown label: drop the
  timestamp (it's in the Run History tab). New format:
  `f"#{r['id']} {r['arch']}/{r['dtype']}" + (f" [{r['label']}]" if r["label"] else "")`.
  Example: `#4 sm_120/bf16 [phase2c-bf16]`.
- [x] **2B.1.6** `scripts/server.py` timing hover: add thousands
  separator to the ns values for readability. Change
  `median=%{y:.0f} ns` to `median=%{y:,.0f} ns` and similarly for
  `ref_median`. (Plotly uses d3-format; `,` is the thousands separator.)
- [x] **2B.1.7** `scripts/server.py` legend: switch from vertical
  (right) to horizontal (below plot) to free up horizontal space.
  `legend=dict(orientation="h", y=-0.2, x=0, xanchor="left")`.
  Apply to both timing and accuracy figures.
- [x] **2B.1.8** `scripts/server.py` cuBLAS traces: with 3 cuBLAS lines
  (one per dtype) they overlap visually. Make cuBLAS lines
  semi-transparent (`opacity=0.6`) so custom kernel lines underneath
  remain visible. Apply to both timing and accuracy figures.

### Validation

- [x] **2B.1.9** Fire a real callback POST (not just `GET /`) to verify
  the fix. Either:
  - Browser: open `localhost:8050`, switch tabs, toggle filters, confirm
    no 500 in the browser console or server log.
  - CLI: `curl -X POST localhost:8050/_dash-update-component -H
    'Content-Type: application/json' -d '{...}'` and check for HTTP 200
    + JSON response containing `"Scatter"` traces.
  The Phase 2B validation only checked `GET /` (which serves the static
  layout and always returns 200); the callback POST was never tested.
- [ ] **2B.1.10** Manual UI review by user. Run the full pipeline:
  ```sh
  ./build/gemm_y
  source pyenv/bin/activate
  python scripts/ingest.py results/bench_sm_120_bf16.csv --label "..." --force
  python scripts/ingest.py results/bench_sm_120_fp16.csv --label "..." --force
  python scripts/ingest.py results/bench_sm_120_tf32.csv --label "..." --force
  python scripts/server.py
  # → open http://localhost:8050
  ```
  Verify: timing tab shows 6 lines (3 cuBLAS + 3 naive), hover shows
  all 7 fields, log/linear toggle works, accuracy tab shows tol lines,
  run history tab lists all ingested runs.

### Feedback (2026-07-19)

All 9 implemented items (2B.1.1–2B.1.9) are done; 2B.1.10 (manual UI
review) is left for the user. Summary:

- **2B.1.1 (callback fix):** changed `@app.callback` from list-wrapped
  `[Input(...), ...]` to flat `Input(...), Input(...), ...` form. Verified
  via `curl -X POST /_dash-update-component` with the correct Dash 4.x
  request body (`outputs` as a dict, not a list, for single-output
  callbacks) — returns HTTP 200 with 6 Scatter traces (3 cuBLAS + 3
  naive). The initial 500s during testing were a curl body-format error
  (sending `outputs` as a list instead of a dict); the Dash frontend sends
  the correct format automatically.
- **2B.1.2 (accuracy zero-clamp):** `_accuracy_figure` clamps displayed
  y-values to `1e-15` for display only (underlying data untouched).
  Verified: accuracy tab returns 6 traces with `y_min=1e-15` (all
  max_rel_err are 0 from deterministic fill).
- **2B.1.3 (`_runs_table` cleanup):** dropped the unused `rows` parameter;
  connection now closed via `try/finally` (sqlite3.Connection's `with`
  only commits/rolls back, does not close).
- **2B.1.4 (restart-server note):** added to the module docstring so it
  appears in `--help` output.
- **2B.1.5 (dropdown label):** dropped the timestamp; new format
  `#<id> <arch>/<dtype> [<label>]`.
- **2B.1.6 (thousands separator):** `median=%{y:,.0f} ns` and
  `ref_median=%{customdata[4]:,.0f} ns` in the timing hovertemplate.
- **2B.1.7 (horizontal legend):** `legend=dict(orientation="h", y=-0.2,
  x=0, xanchor="left")` on both figures; bottom margin bumped to 80 to
  fit the legend below the plot.
- **2B.1.8 (semi-transparent cuBLAS):** `opacity=0.6` on cuBLAS traces in
  both timing and accuracy figures; custom traces stay at 1.0.
- **2B.1.9 (callback POST validation):** fired real POSTs for all three
  tabs (timing/accuracy/runs) and the linear-scale toggle — all return
  HTTP 200 with the expected payload (6 Scatter traces for timing/accuracy,
  DataTable with 9 rows for runs, `linear` axis type for the scale toggle).
- **Connection hygiene:** `build_app()` and `render_tab` now also close
  their `db.connect()` connections via `try/finally` (beyond what 2B.1.3
  required) — prevents connection leaks across callback invocations.

**Files changed:** `scripts/server.py` only. No C++ changes; no DB schema
changes. The DB was re-ingested (runs 7–9) to reflect the Phase 2C
partial CSVs (28 rows each, up from 14 for fp16/tf32).

---

## Phase 2C — Tiled tensor-core kernels (owner: user)

Goal: first tensor-core kernel for bf16 that beats cuBLAS at large N,
then replicate to fp16 / tf32. The naive kernel (Phase 2C partial,
already landed) gives every dtype a custom kernel to compare against
cuBLAS, but it's ~10–40× slower than cuBLAS at N≥512 — the tiled TC
kernel is the real work.

**Owner:** user. The agent's first attempt (128×128 CTA, 8 warps,
`nvcuda::wmma` 16×16×16) failed accuracy and was reverted; the failure
analysis is in commit `73ef7fb`. Re-attempt from scratch — the user
will drive the debugging approach.

- [ ] **2C.1** `src/sm120/gemm_bf16_tiled_128.cu` (+ `.cuh`) — 128×128
  tile, 8 warps/CTA, `wmma`/`mma.sync` for bf16 on sm_120.
- [ ] **2C.2** `src/sm90/gemm_bf16_tiled_128.cu` (+ `.cuh`) — same
  algorithm, sm_90 `wmma` API.
- [ ] **2C.3** Register in `main.cpp` alongside `NaiveGemm<bf16>`. Run
  sweep, ingest to DB, view in dashboard, compare vs cuBLAS line.
  Iterate (one variable per commit per AGENTS.md experiment discipline).
- [ ] **2C.4** Once bf16 tiled kernel is competitive, replicate ideas to
  fp16 (Path 1 sibling). Expect near-identical perf on tensor cores.
- [ ] **2C.5** Replicate to tfloat (tf32 path) — same TC MMA, different
  dtype config.

---

## Phase 2 prep (deferred)

- `Space::HostPinned` + `Buffer<T, HostPinned>` via `cudaHostAlloc`.
- Bench runner host buffers → pinned.
- Async `cudaMemcpyAsync` on explicit stream (when pipelining lands).
- Debug-build assert or `static_assert` on pinned-ness at `copy_*` call
  sites when passing non-null stream (catches silent staging).

---

## Out of scope (unchanged)

- `cublasLtMatmul` (Phase 3+).
- Batched, transposed, epilogue-fused, non-square variants (AGENTS.md non-goals).
- nsys / ncu profiling integration.
- Multi-GPU / multi-node.
- fp32 pedantic (CUDA cores) — dropped entirely; only tf32 path for
  32-bit float storage (see ARD §9).
