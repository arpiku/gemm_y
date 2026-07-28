# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

---

## Phase 2G — Fan removal, kernel/harness decoupling, visualization

Goal: three independent tracks. (1) Remove the dead fan-control path from
`bench.sh` (not supported on RTX 5070). (2) Decouple device kernels from
the harness — `__global__` functions take raw pointers + dimension ints,
not `MatrixView`; the `operator()` functor is the thin adapter that unpacks
`GemmArgs`. (3) Build out the dashboard's visualization surface: six new
views (distribution, speedup, TFLOPS, roofline, cross-run diff,
significance overlay). Also: clean up `Stats.h` dead code, fix the
`server.py` color-comment mismatch.

### Step 1 — Remove fan-speed control from `bench.sh`

**Context:** `-fan-gpu` returns "Not Supported" on RTX 5070 (consumer cards
lock the auto-fan curve to the driver). The attempt-and-warn path is dead
weight locally; the auto-fan curve is the only path that ever runs. Remove
it entirely rather than carrying a fallback that never fires.

- [ ] **2G.1.1** `scripts/bench.sh`: remove the `-fan-gpu 100` setup block
  and the `-fan-gpu 0` reset line in `reset_state`. Remove the
  fan-control caveat comment block at the top of the file. Keep:
  persistence mode (`-pm 1`), clock lock (`-lgc <max>,<max>`), drop-privs
  (`sudo -u`), pre/post temperature print, trap on `EXIT`/`INT`/`TERM`.
  The reset trap keeps `-rgc` (clock reset) and `-pm 0` (persistence
  reset); drop the fan reset line.
- [ ] **2G.1.2** `AGENTS.md` → "Reproducible runs (clock locking + fan
  control)": rename section to "Reproducible runs (clock locking)". Drop
  the "spins fans to 100% (if supported)" clause from the first bullet.
  Delete the "Fan-control caveat" bullet entirely. Keep the thermal-safety
  bullet (still relevant — auto-fan curve + clock lock + pre/post temp).
- [ ] **2G.1.3** `ARD.md` §18: rename to "Reproducible benchmarks: clock
  locking (`bench.sh`)". Update the TOC entry. In the Decision: drop
  step 2 (fan spin), drop the fan reset from step 7. In Rationale: delete
  the "Fan control is best-effort" bullet. In Consequences: delete the
  "Cross-arch (H100) may need admin cooperation for fan control" bullet
  (replace with a note that H100 fan control is available but not
  orchestrated by this wrapper — the auto curve suffices). Keep the
  thermal-safety rationale.
- [ ] **2G.1.4** `TODO.md` 2E.2.1: the step is already done in practice;
  this is a documentation sync. No action needed in TODO.md itself (the
  2E section is forward-looking only if incomplete — verify 2E is fully
  checked off; if so, leave it).

### Step 2 — Decouple device kernels from the harness

**Context:** device kernels currently take `MatrixView<const T, Device>`
by value. This couples kernel authoring to the harness's view type — a
kernel author must include `MatrixView.h`, `Space.h`, and understand the
POD-descriptor contract. A kernel should be writable as pure CUDA:
`__global__ void k(T* A, T* B, T* C, int M, int N, int K, int ldA, int ldB,
int ldC)`. The harness's `operator()` functor (the `KernelTraits`-satisfying
adapter) unpacks `GemmArgs<T>` into the raw-pointer call.

**Layering:**
- **Device kernel** (`__global__`): raw pointers + dimension ints. No
  project includes. Pure CUDA.
- **Host functor** (`operator()`): takes `GemmArgs<T>` + `cudaStream_t`
  (the harness ABI). Unpacks `args.A.ptr`, `args.A.ld`, etc. into the
  raw-pointer kernel launch.
- **`GemmArgs` / `MatrixView`**: unchanged. Still the harness ABI.

**Dimension convention:** `(T* A, T* B, T* C, int M, int N, int K, int ldA,
int ldB, int ldC)` where `C = A(M×K) × B(K×N) = C(M×N)`. For the square
sweep, `M == N == K`, but the kernel signature is general.

- [ ] **2G.2.1** `src/sm90/gemm_naive.cu`, `src/sm120/gemm_naive.cu`:
  change `detail::naive_gemm_kernel<T>` signature from
  `(MatrixView<const T, Device> A, MatrixView<const T, Device> B,
  MatrixView<T, Device> C)` to
  `(const T* A, const T* B, T* C, int M, int N, int K, int ldA, int ldB,
  int ldC)`. Update the body: `A.ptr[i + k*A.ld]` → `A[i + k*ldA]`, etc.
  Bounds check uses `M`/`N` (C rows/cols). Inner loop uses `K` (A cols).
  Update `NaiveGemm<T>::operator()` to unpack:
  `detail::naive_gemm_kernel<T><<<grid, block, 0, stream>>>(
      args.A.ptr, args.B.ptr, args.C.ptr,
      args.C.rows, args.C.cols, args.A.cols,
      args.A.ld, args.B.ld, args.C.ld)`.
  Drop `#include "MatrixView.h"` from these `.cu` files (no longer needed
  in the kernel body; `GemmArgs.h` pulls it in transitively for the host
  side).
- [ ] **2G.2.2** `src/sm90/gemm_bf16_k0.cu`, `src/sm120/gemm_bf16_k0.cu`:
  same transformation. `k0_gemm_kernel` takes
  `(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
  int M, int N, int K, int ldA, int ldB, int ldC)`. `k0::operator()`
  unpacks `GemmArgs`. Drop `#include "MatrixView.h"`.
- [ ] **2G.2.3** Verify no other kernel `.cu` files exist in `src/sm90/`
  and `src/sm120/` (only `gemm_naive.cu` and `gemm_bf16_k0.cu` per arch).
  If `gemm_bf16_k1.cu` or similar exist, apply the same transformation.
- [ ] **2G.2.4** `tests/test.cu`: verify no test calls the device kernel
  directly (they're in anonymous namespaces — unlikely). Tests call
  `NaiveGemm<T>::operator()` via the Profiler, which is unchanged at the
  ABI level. If any test constructs `MatrixView` for a kernel call, update
  to the new signature. Run `ctest` to confirm.
- [ ] **2G.2.5** `ARD.md` §3 (Kernel abstraction): add a sub-section
  documenting the layering — device `__global__` functions take raw
  pointers + dimension ints `(M, N, K, ldA, ldB, ldC)`; the `operator()`
  functor is the harness adapter that unpacks `GemmArgs<T>`. This is the
  durable contract for future kernel authors. Note that `MatrixView` is
  not passed across the kernel boundary.
- [ ] **2G.2.6** `AGENTS.md` → "Kernel file organization": add a bullet
  that the device `__global__` function takes raw pointers + dimension
  ints (no `MatrixView`), and the `operator()` functor is the thin adapter.
  Update the "Kernel ABI (`GemmArgs<T>`)" bullet to clarify the layering:
  `GemmArgs` is the harness ABI; the device kernel ABI is raw pointers.

### Step 3 — Clean up `Stats.h` dead code

- [ ] **2G.3.1** `src/bench/Stats.h`: remove the dead `mean_ns` placeholder
  (lines ~46–47: `const double mean_ns = ...; // placeholder`) and the
  `(void)mean_ns;` line (~58). The real mean is computed at line ~51
  (`const double mean = acc / n`). The placeholder is leftover from
  refactoring — confusing dead code.

### Step 4 — Fix `server.py` color-comment mismatch

- [ ] **2G.4.1** `scripts/server.py` (lines ~48–52): the comments say
  "light gray plot area" for `PLOT_BG_COLOR = "#ffffff"` and "white
  margin / page" for `PAPER_BG_COLOR = "#f0f0f0"` — both wrong (inverted).
  Fix the comments to match the actual values: `#ffffff` = white plot
  area, `#f0f0f0` = light gray page/margin. The values themselves are the
  user's manual choice — keep them. Remove the commented-out
  `#PLOT_BG_COLOR = "#f0f0f0"` / `#PAPER_BG_COLOR = "#ffffff"` lines
  (dead alternates).

### Step 5 — Visualization: raw-sample storage (foundation for 2G.6)

**Context:** the distribution tab (2G.6.a) and the cross-run diff view
(2G.6.e) need access to the raw 50 timed samples per (run, N, kernel),
not just the summary stats. Currently only summary stats (min/median/std/
p95/CI) are stored. Add a `samples` table.

- [ ] **2G.5.1** `scripts/db.py`: add a `samples` table schema:
  `CREATE TABLE IF NOT EXISTS samples (
     run_id INTEGER NOT NULL,
     n INTEGER NOT NULL,
     kernel_name TEXT NOT NULL,
     sample_index INTEGER NOT NULL,
     ns REAL NOT NULL,
     FOREIGN KEY (run_id) REFERENCES runs(id)
  )`. Add an index on `(run_id, n, kernel_name)`. Add `insert_sample()`
  and `fetch_samples()` query functions. `_migrate()` should create the
  table idempotently (CREATE TABLE IF NOT EXISTS handles this).
- [ ] **2G.5.2** `src/bench/Profiler.cu`: the 50 raw `ms` samples per
  (N, kernel) are currently discarded after `summarize_ns`. They need to
  reach the CSV so `ingest.py` can store them. Add a `std::vector<double>
  raw_samples_ns` field to `SweepRow` (the 50 samples in ns, serialized).
  Populate it from `ms` (convert each to ns). For cuBLAS rows, populate
  from the cuBLAS `ms`. **CSV format decision:** add a single
  `kernel_samples_ns` column containing the 50 values as a
  semicolon-separated string (e.g. `1234.5;1235.1;...`). Avoids 50
  columns. `ingest.py` splits on `;` and inserts one row per sample into
  the `samples` table. Old CSVs (pre-2G) don't have this column —
  `ingest.py` handles its absence (no samples stored for old runs).
- [ ] **2G.5.3** `src/main.cpp`: add `kernel_samples_ns` to the CSV header
  and `append_row` calls. `CsvWriter` must handle a string field containing
  semicolons (verify it doesn't split on `;` — CSV uses commas; semicolons
  are safe inside a field).
- [ ] **2G.5.4** `scripts/ingest.py`: parse `kernel_samples_ns` (split on
  `;`), insert each sample via `db.insert_sample()`. Handle missing column
  (old CSVs) — write no samples. Handle empty string — write no samples.
- [ ] **2G.5.5** `src/bench/Profiler.h`: add `std::string
  kernel_samples_ns` field to `SweepRow` (serialized semicolon-separated).
  Document the format in a comment.

### Step 6 — Visualization: dashboard views

**Context:** three views, building on the 2E.3 CI error bars and the
2G.5 raw-sample storage. (Distribution / TFLOPS / Roofline tabs were
implemented and removed after manual review — see ARD §20 "Removed
views".)

#### (f) Statistical significance overlay on the Timing tab

- [x] **2G.6.1** `scripts/server.py` `_timing_figure`: significance
  marker on each custom point. A point is **significant** (custom ≠
  cuBLAS) if the custom CI and cuBLAS CI **do not overlap** vertically at
  that N. **Inconclusive** if they overlap. Visual: significant points
  get a green ring (faster) or red ring (slower) around the marker;
  inconclusive points get a gray ring. Compute significance per point
  from `kernel_ci_low_ns`/`kernel_ci_high_ns` and `ref_kernel_ci_low_ns`/
  `ref_kernel_ci_high_ns`. Skip for old runs missing CI data.

#### (b) Speedup tab

- [x] **2G.6.5** `scripts/server.py`: "Speedup" tab. Plots
  `ref_median / kernel_median` (speedup ratio) vs N. `>1` = custom faster
  than cuBLAS. Parity line at 1.0. Per-point CI error bars via error
  propagation: the CI half-width for the ratio is
  `ratio * sqrt((σ_kernel/median_kernel)^2 + (σ_ref/median_ref)^2)`
  (first-order Gaussian propagation). Hover shows ratio, custom median,
  cuBLAS median, propagated CI. cuBLAS trace excluded (ratio = 1 by
  definition). Log-y is the default (ratios span 0.1–10×).

#### (e) Cross-run diff view

- [x] **2G.6.8** `scripts/server.py`: "Diff" tab. Sidebar gets a
  second run selector (run B). Plots `median_A / median_B` vs N, one line
  per kernel name common to both runs. `>1` = run A is slower; `<1` =
  run A is faster. Parity line at 1.0. Per-point CI via propagation (same
  formula as 2G.6.5). Directly answers "did k1 beat k0". If a kernel
  exists in only one run, it's skipped.

#### Removed views

- ~~**2G.6.2–2G.6.4** Distribution tab (box + violin)~~ — removed after
  manual review. `fetch_samples_for_n` deleted from `db.py`; the
  `samples` table and `insert_sample`/`fetch_samples` remain as storage
  infrastructure.
- ~~**2G.6.6** TFLOPS tab~~ — removed. `PEAK_SPECS` dict deleted from
  `server.py`.
- ~~**2G.6.7** Roofline tab~~ — removed. `DTYPE_BYTES` dict deleted from
  `server.py`.
- ~~**2G.6.3** N-selector sidebar control~~ — removed (only the
  Distribution tab used it).
- ~~**2G.6.9–2G.6.11** Shared infrastructure~~ — the tab dispatch and
  callback were updated to handle the remaining tabs; ARD §20 documents
  the methodology.

### Step 7 — Validate

- [ ] **2G.7.1** Build: `cmake -B build && cmake --build build -j`.
- [ ] **2G.7.2** Tests: `ctest --test-dir build`.
- [ ] **2G.7.3** Sweep: `sudo scripts/bench.sh`. Verify:
  - `bench.sh` no longer attempts `-fan-gpu`; no fan warning printed.
  - CSV has `kernel_samples_ns` column with 50 semicolon-separated values.
  - Kernels run with raw-pointer signatures (no functional change —
    verify output CSV matches a pre-2G run at the summary-stat level).
- [ ] **2G.7.4** Dashboard:
  ```sh
  source pyenv/bin/activate
  python scripts/ingest.py results/bench_sm_120_bf16.csv --label "2G-viz"
  python scripts/server.py
  ```
  Verify:
  - Timing tab: significance overlay (green/red/gray rings) on each point.
  - Speedup tab: ratio vs N with propagated CI, parity at 1.0.
  - Diff tab: run A vs run B ratio, parity at 1.0.
  - All tabs respect sidebar filters (arch/dtype/runs/class/scale/height).
- [ ] **2G.7.5** `Stats.h` dead code gone; `server.py` comments match
  values.

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
  diff view? — **2G.6.8 adds a diff view**), how to record the hypothesis
  for each commit (commit body per AGENTS.md — sufficient? or need a
  sidecar file?).
- [ ] **2C.2.2** ~~Decide on statistical rigor: current 20 warmup / 50
  timed median. Is median enough? Need min (best-case) or p99 (tail)?
  Need confidence intervals?~~ **Resolved by Phase 2E.3** — 95% CI for
  the median + p95 + std, with per-point error bars on the dashboard.
  `kTimed` stays at 50 (p95 is the 48th sample; p99 would need 200+
  samples — deferred until tail analysis becomes important).
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
  `gemm_bf16_tiled128.cu` when finalized, per ARD §16). Device kernel
  takes raw pointers + ints per Phase 2G.2.
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
  Note: after Phase 2D, `cudaMemcpy` is called directly from `Profiler.cu`
  (no `Copy.h` wrapper). When async lands, either re-introduce a thin
  wrapper or keep the direct calls with an explicit `cudaStream_t`
  argument threaded through `run_sweep`.

---

## Out of scope (unchanged)

- `cublasLtMatmul` (Phase 3+).
- Batched, transposed, epilogue-fused, non-square variants (AGENTS.md non-goals).
- nsys / ncu profiling integration (may enter in 2C.2 — TBD by user).
- Multi-GPU / multi-node.
- fp32 pedantic (CUDA cores) — dropped entirely; only tf32 path for
  32-bit float storage (see ARD §9).
- RowMajor layout — deleted in Phase 2D. If a future phase needs it,
  re-add a `Layout` tag + the `RowMajor` branches (5-line re-add per
  call site; not worth carrying as dead code).
- Fan control orchestration — removed in Phase 2G. The auto-fan curve is
  the only path; if a future datacenter deployment needs manual fan
  control, re-add a `-fan-gpu` step to `bench.sh` behind a host-detect
  guard.
