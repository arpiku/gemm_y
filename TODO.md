# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

---

## Phase 2B.3 — Dashboard polish round 2 (post-review fixes)

Goal: fix the functional hover bug surfaced during the 2B.2 review and apply
the three UI polish items the user reported. Python only; no C++ changes.

**Context:** Phase 2B.2 (commit `3281428`) shipped the `% perf vs cuBLAS`
metric, Comparison tab, and `delete_run.py` CLI. The user's manual UI review
(2B.2.7) found: (a) a hovertemplate `customdata` index bug showing the wrong
values, (b) sparse/faint grid, (c) low contrast between plot area and page
background + small font, (d) overlapping tol annotations in the accuracy
tab's top-right corner. Round 2 (this phase) addresses the round-1
follow-up: grid too thick on the accuracy chart, chart too short, hover
fractions too verbose.

### Bug fix (functional) — done in round 1

- [x] **2B.3.1** `scripts/server.py` `_timing_figure`: fix the
  `customdata` index bug in both the custom and cuBLAS hovertemplates.
  Current `customdata` layout (L90–102):
  ```
  [0] arch   [1] dtype   [2] class   [3] kernel_desc
  [4] kernel_median_ns   [5] ref_kernel_median_ns
  [6] speedup            [7] perf_pct
  ```
  Current hovertemplate reads `ref_median` from `[4]` (wrong — that's
  `kernel_median_ns`) and `speedup` from `[5]` (wrong — that's
  `ref_kernel_median_ns`). Fix to:
  ```
  ref_median=%{customdata[5]:,.0f} ns
  speedup=%{customdata[6]:.3f}x
  perf=%{customdata[7]:+.1f}% vs cuBLAS (+ = faster)
  ```
  Apply to both `hovertemplate` (custom) and `cublas_hovertemplate` (cuBLAS).
  Verify in the browser: hover a custom point, confirm `ref_median` shows
  the cuBLAS median at that N (not the kernel's own median) and `speedup`
  shows the ratio (not the raw ns).

### UI polish round 1 — done

- [x] **2B.3.2** `scripts/server.py`: denser + bolder grid on all three
  figure builders (`_timing_figure`, `_accuracy_figure`,
  `_comparison_figure`). Per-axis (x and y):
  - `showgrid=True`, `gridwidth=2`, `gridcolor="rgba(0,0,0,0.35)"`.
  - `ticks="outside"`, `tickwidth=2`, `ticklen=6`.
  - For log-x (N sweep): set `tickvals` to the 14 sweep sizes
    `[32,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096]` so
    every data point has a tick. For linear-x, leave Plotly auto.
  - For log-y (timing/accuracy): `dtick="D1"` (every decade) or
    `tick0=10, dtick=10` — pick what reads best for the 10ns–10us range.
  - `zeroline=True, zerolinewidth=2` on the Comparison tab's y-axis
    (parity line at 0 should be visually distinct from the grid).
  - Consider a shared `_axis_layout(log_x, log_y)` helper to avoid
    repeating the dict across three builders.

- [x] **2B.3.3** `scripts/server.py`: higher contrast plot area vs page
  background + larger font.
  - Per-figure `update_layout`: `plot_bgcolor="#f0f0f0"` (light gray
    plot area), `paper_bgcolor="#ffffff"` (white margin/page).
    Current default is white-on-white — no visible plot boundary.
    If `#f0f0f0` is too light, try `#e8e8e8`; the user wants clear
    contrast with the white page.
  - Per-figure `update_layout`: `font=dict(size=14)` (default is 12).
    Apply to title, axis titles, tick labels, legend, hover. If 14 is
    too large for the legend, set `font=dict(size=14)` globally and
    `legend=dict(font=dict(size=12))` to keep the legend compact.
  - App-level: `app.layout`'s root `html.Div` `style` — bump
    `fontSize` from the browser default (16px) to 15–16px for sidebar
    labels, or set explicitly via `html.Label(style={"fontSize": 14})`.
    Current sidebar has no font-size set; it inherits the browser
    default which varies.

- [x] **2B.3.4** `scripts/server.py` `_accuracy_figure`: fix the
  overlapping tol annotations in the top-right corner. Root cause:
  `add_hline(..., annotation_position="top right")` is called once per
  distinct `(run_id, tol)` — with 11 runs and 2–3 distinct tols (bf16=1e-2,
  fp16=1e-3, tf32=1e-3), the annotations stack at the same corner.
  Pick one approach:
  - **(a) Deduplicate by tol value** (recommended): one `add_hline` per
    distinct tol, combined label `tol=1e-2 (bf16) / 1e-3 (fp16, tf32)`.
    Requires grouping runs by tol and building a combined annotation
    string. Single annotation per tol, no stacking.
  - **(b) Stagger positions**: rotate `annotation_position` across
    `["top left", "top center", "top right"]` per distinct tol. Quick
    but fragile if there are >3 distinct tols.
  - **(c) Drop annotations, use legend**: add the tol as a legend entry
    (invisible trace) or into the hover. Cleanest but loses the
    at-a-glance y-value reference.
  Implement (a) unless the user prefers otherwise. The dedup logic:
  collect `{tol: [list of (run_id, dtype)]}` from `rows`, then for each
  distinct tol, one `add_hline` with `annotation_text` listing the
  dtypes that share it.

### UI polish round 2 — from second user review

**Context:** round 1 (2B.3.2–2B.3.4) landed `gridwidth=2` +
`gridcolor="rgba(0,0,0,0.35)"` uniformly across all three figures. The
user's second review found: (a) the accuracy chart is now smushed — the
y-range is tiny (1e-15 to ~1e-3 in log, or 0 to ~1e-2 in linear) so
`gridwidth=2` merges into a solid block; (b) chart height is hardcoded
at 520px and feels too short / skewed, especially the accuracy tab;
(c) hover fractions are too verbose — `speedup=%.3f` (3 decimals),
`perf_pct=%.1f` (1 decimal), `max_rel_err=%.3e` (3 decimals) — the user
wants a 2-decimal ceiling across the board.

- [x] **2B.3.6** `scripts/server.py` `_axis_layout`: parameterize grid
  weight per figure. Add a `grid_weight: str = "bold"` arg to
  `_axis_layout` and `_base_layout` (values: `"bold"` / `"light"`).
  - `"bold"` (timing, comparison): `gridwidth=2`,
    `gridcolor="rgba(0,0,0,0.35)"`. Keep as-is — these charts have a
    wide y-range and the bold grid reads fine.
  - `"light"` (accuracy): `gridwidth=1`,
    `gridcolor="rgba(0,0,0,0.15)"`. The accuracy y-range is tiny
    (1e-15 to ~1e-3); `gridwidth=2` merges into a solid block. Lighter
    grid keeps the chart readable.
  - `tickwidth` follows `gridwidth` (2 for bold, 1 for light) so the
    outside ticks match the grid weight.
  - Pass `grid_weight="light"` from `_accuracy_figure`; default
    `"bold"` everywhere else.

- [x] **2B.3.7** `scripts/server.py`: add a chart-height control to the
  sidebar and thread it through the callback.
  - Sidebar: new `dcc.RadioItems(id="filter-chart-height", ...)` under
    the Scale control. Options: `520 / 640 / 760 / 900` (label them
    "S / M / L / XL" or just the pixel values). Default `640` (bump
    from the current 520 — the user said the chart is too short).
  - `_base_layout`: replace the hardcoded `"height": 520` with an
    `height: int` parameter. Thread it through `_timing_figure`,
    `_comparison_figure`, `_accuracy_figure`.
  - `render_tab` callback: add `Input("filter-chart-height", "value")`
    to the flat `@app.callback` signature (per AGENTS.md "Python /
    Dash" — flat, not list-wrapped). Pass the value to the figure
    builders.
  - Run History tab is a `dt.DataTable`, not a figure — height control
    does not apply. Leave it unchanged.
  - This is the "dynamic layout" the user asked for — runtime control
    over chart height without a server restart.

- [x] **2B.3.8** `scripts/server.py`: round hover fractions to a
  2-decimal ceiling across all three figure builders.
  - `_timing_figure` hovertemplate (custom + cuBLAS):
    - `speedup=%{customdata[6]:.3f}x` → `speedup=%{customdata[6]:.2f}x`
    - `perf=%{customdata[7]:+.1f}%` → `perf=%{customdata[7]:+.2f}%`
    - `median` and `ref_median` stay at `:,.0f` (integer ns — no
      fraction to round).
  - `_comparison_figure` hovertemplate:
    - `perf=%{y:+.1f}%` → `perf=%{y:+.2f}%`
    - `median` / `ref_median` stay at `:,.0f`.
  - `_accuracy_figure` hovertemplate:
    - `max_rel_err=%{y:.3e}` → `max_rel_err=%{y:.2e}`
  - Consistent 2-decimal ceiling across speedup, perf_pct, max_rel_err.
    Integer ns values stay integer (no fraction to round).

### Validation

- [ ] **2B.3.5** Re-run the dashboard and verify all three tabs in the
  browser:
  ```sh
  source pyenv/bin/activate
  python scripts/server.py
  # → open http://localhost:8050
  ```
  - **Timing tab**: hover a custom point — `ref_median` shows the cuBLAS
    median (not the kernel's own), `speedup` shows the ratio (not raw ns).
    Grid is denser + bolder; plot area is visibly distinct from the page
    background; font is larger.
  - **Comparison tab**: parity line at 0 is visually distinct; grid +
    contrast + font consistent with the timing tab.
  - **Accuracy tab**: no overlapping annotations in the top-right; each
    distinct tol has one combined label. Grid + contrast + font
    consistent.
  - **Run History tab**: unchanged (table, not a figure) — verify the
    `median % vs cuBLAS @ N=4096` column still populates.
  - All prior behavior (log/linear toggle, sidebar filters, run
    multi-select) still works.
- [ ] **2B.3.9** Round-2 validation (after 2B.3.6–2B.3.8 land):
  - **Accuracy tab**: grid is no longer a solid block — individual grid
    lines are visible across the tiny y-range. Chart is taller (default
    640, or whatever the user picks in the sidebar). Hover shows
    `max_rel_err=X.YZe±NN` (2-decimal scientific).
  - **Timing tab**: grid weight unchanged from round 1 (bold). Chart
    height responds to the sidebar control. Hover shows
    `speedup=X.YZx` and `perf=±X.YZ%` (2 decimals each).
  - **Comparison tab**: same — 2-decimal `perf`, height responds to
    sidebar.
  - **Sidebar**: new "Chart height" control appears under Scale;
    changing it re-renders the current tab at the new height without a
    server restart. Run History tab is unaffected.

---

## Phase 2C.2 — Methodology / measurement strategy (owner: user)

Goal: before writing optimized (Tiled, double-buffer, …) kernels,
establish a repeatable workflow for testing and measuring them. Drives
the structure of 2C.3+.

**Owner:** user. The agent supports (dashboard extensions, CLI helpers,
documentation) but does not define the methodology.

- [ ] **2C.2.1** Define the iteration workflow: how to label runs
  (`--label "k1-tiling-128"`?), how to compare `k_n` vs `k_(n-1)` in the
  dashboard (run multi-select already exists — sufficient? or need a
  diff view?), how to record the hypothesis for each commit (commit body
  per AGENTS.md — sufficient? or need a sidecar file?).
- [ ] **2C.2.2** Decide on statistical rigor: current 20 warmup / 50
  timed median. Is median enough? Need min (best-case) or p99 (tail)?
  Need confidence intervals? If yes, extend `Profiler::run_sweep` to
  emit more stats and the CSV/DB schema to store them.
- [ ] **2C.2.3** Profiling tool setup — nsys / ncu. Currently out of
  scope (see Out of scope section). Decide whether to bring them in
  here, or defer to a later phase. If bringing in: wrapper scripts,
  report parsing, dashboard integration (TBD scope).
- [ ] **2C.2.4** Document the workflow in `ARD.md` (new section) and
  `AGENTS.md` (workflow blurb) once it's stable.

---

## Phase 2C.3+ — Real custom kernels (DEFERRED, owner: user)

Goal: tiled tensor-core kernels that beat cuBLAS at large N. Deferred
until 2C.2 (methodology) is in place — the user wants the measurement
loop solid before investing in optimization.

**Owner:** user. The agent's first tiled attempt (128×128 CTA, 8 warps,
`nvcuda::wmma` 16×16×16) failed accuracy and was reverted; the failure
analysis is in commit `73ef7fb`. The user will re-attempt from scratch
with a debugging approach driven by the 2C.2 methodology.

- [ ] **2C.3.1** First tiled TC kernel for bf16 on sm_120 (file name
  TBD — `gemm_bf16_k1.cu` during development, renamed to
  `gemm_bf16_tiled128.cu` when finalized, per ARD §16).
- [ ] **2C.3.2** Mirror on sm_90 (same algorithm, sm_90 wmma API).
- [ ] **2C.3.3** Iterate: one variable per commit (tile size, warp count,
  K-dim unroll, memory layout). Ingest each iteration, compare in
  dashboard, record hypothesis + result in commit body.
- [ ] **2C.3.4** Replicate to fp16 (Path 1 sibling) once bf16 is
  competitive. Expect near-identical perf on tensor cores.
- [ ] **2C.3.5** Replicate to tfloat (tf32 path) — same TC MMA, different
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
- nsys / ncu profiling integration (may enter in 2C.2 — TBD by user).
- Multi-GPU / multi-node.
- fp32 pedantic (CUDA cores) — dropped entirely; only tf32 path for
  32-bit float storage (see ARD §9).
- TFLOPS metric — deferred. Needs peak-TFLOPS lookup per `(arch, dtype)`.
  Will be added in a future sub-phase after the `% perf vs cuBLAS` metric
  is in place.
