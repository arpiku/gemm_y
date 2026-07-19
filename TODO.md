# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

## Phase 2A — cuBLAS references for bf16 / fp16 / tf32

Goal: cuBLAS reference paths available for all three storage dtypes so any
custom kernel variant can be benchmarked against the right baseline.

**Three dtypes (priority order):**
1. **bf16** (primary): tensor cores, fp32 accum. Phase 1 baseline.
2. **fp16** (primary sibling): tensor cores, fp32 accum. Should show no
   meaningful perf difference vs bf16 on Hopper/Blackwell TCs — optimize
   one, replicate ideas to the other.
3. **tfloat** (secondary): `tfloat = float` alias. tf32 compute (TC),
   not pedantic fp32 (CUDA cores). The pedantic CUDA-core path is
   dropped entirely — see ARD §9.

**Key decisions (see ARD §6, §9 for rationale):**
- `tfloat = float` alias, always commented: `// tfloat = tf32 path (TC),
  not pedantic fp32 (CUDA cores).`
- `CublasTypeMap<T>` gains a `math_mode` field; `cublas_gemm` wraps the
  call in `CublasMathModeGuard(handle, TM::math_mode)`. No distinct
  `cublas_gemm_tf32` entry point — `cublas_gemm<float>` *is* the tf32 path.
- `kRelErrTol<T>` per-dtype constant. Failed kernels (rel_err > tol) are
  **skipped at the Profiler level** — no CSV row, stderr FAIL message
  only. Timing of mathematically invalid kernels is meaningless.
- `NaiveGemm<fp16>` and `NaiveGemm<tfloat>` are deferred to Phase 2C.
  2A registers only `NaiveGemm<bf16>` (already exists) + cuBLAS reference
  (auto-included by Profiler) for all three dtypes.

### C++ changes

- [x] **2A.1** `src/dtypes.h`: add `using tfloat = float;` alias with
  inline comment explaining the tf32-only intent. Change
  `name<float>()` to return `"tf32"`. Drop the `fp32` name (no pedantic
  path). Document the alias in ARD §9.
- [x] **2A.2** `src/cublas/cublas_gemm.h`: extend `CublasTypeMap<T>` with
  a `static constexpr cublasMath_t math_mode` field per specialization:
  - `bf16` → `CUBLAS_DEFAULT_MATH`
  - `fp16` → `CUBLAS_DEFAULT_MATH`
  - `float` (tfloat) → `CUBLAS_TF32_TENSOR_OP_MATH`
  Update the `cublas_gemm` body to construct a `CublasMathModeGuard`
  around the `cublasGemmEx` call. No distinct `cublas_gemm_tf32` entry
  point — the math mode is selected by `CublasTypeMap<T>::math_mode`.

  **Implementation note (2026-07-19):** ARD §9 / AGENTS.md / TODO all
  reference `CUBLAS_TF32_CUBLAS_MATH`, but the actual constant in
  CUDA 13.2's `cublas_api.h` is `CUBLAS_TF32_TENSOR_OP_MATH` (enum
  value 3). `CUBLAS_TF32_CUBLAS_MATH` does not exist. Used
  `CUBLAS_TF32_TENSOR_OP_MATH`. ARD/AGENTS.md should be updated to
  reflect the real constant name.
- [x] **2A.3** `src/cublas/CublasHandle.h`: add free class
  `CublasMathModeGuard` (ctor captures prev mode via
  `cublasGetMathMode`, sets new mode; dtor restores prev). Non-copyable,
  non-movable. Used by `cublas_gemm`.
- [x] **2A.4** `src/bench/Accuracy.h`: replace the single global
  `kRelErrTol` constant with `template <typename T> constexpr double
  kRelErrTol<T>()` specializations:
  - `bf16`  → `1e-2`
  - `fp16`  → `1e-3`
  - `tfloat`→ `1e-3`
  Keep `compare<T>(ref, got)` returning `ErrReport<T>` unchanged.
- [x] **2A.5** `src/bench/Profiler.cu`: in `run_sweep`, after
  `compare<T>(...)`, check `err.max_rel > kRelErrTol<T>()`. If true:
  print stderr FAIL message with N, kernel name, rel_err, tol; **skip
  the row** (`continue` to next kernel). Do not write the SweepRow.
  Passing kernels write the row as before. cuBLAS reference rows are
  always written (ground truth, err == 0).
- [x] **2A.6** `src/main.cpp`: extend to run three sequential sweeps
  (bf16, fp16, tfloat). Each sweep: construct `Profiler<T>`, register
  `NaiveGemm<T>` (bf16 only — fp16/tfloat register cuBLAS-only for 2A),
  call `run_sweep`, write CSV + `.meta` sidecar. One `(arch, dtype)`
  pair per CSV. Use explicit blocks (no template metaprogramming).
- [x] **2A.7** `src/main.cpp`: add `write_meta()` helper that writes a
  simple key=value sidecar file `results/bench_<arch>_<dtype>.meta`
  alongside the CSV. Format:
  ```
  arch=sm_120
  dtype=bf16
  warmup_iters=20
  timed_iters=50
  tol=1e-2
  sweep_sizes=32,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096
  kernel=cublas|cublasGemmEx reference (fp32 accum)
  kernel=NaiveGemm|naive GEMM, one thread per element
  timestamp=2026-07-19T17:44:00Z
  ```
  - `|` separates kernel name and description (descriptions may contain
    commas/spaces; `|` will not).
  - Multiple `kernel=` lines (one per registered kernel, cuBLAS first).
  - `timestamp` from `std::chrono::system_clock` (ISO 8601).
  - No git sha (captured by `ingest.py` at ingest time).
  - Hand-rolled `fprintf` (~15 lines). No JSON library, no third-party deps.

### Tests

- [x] **2A.8** `tests/test.cu`: add `test_cublas_gemm_fp16` and
  `test_cublas_gemm_tfloat`. Each: small GEMM (e.g. N=64), compare
  cuBLAS output vs a host-computed fp64 reference, assert
  `max_rel_err <= kRelErrTol<T>()`. The tfloat test must additionally
  verify math-mode restore: capture mode before, run `cublas_gemm<float>`,
  capture mode after, assert unchanged (i.e. `CublasMathModeGuard`
  restored to `CUBLAS_DEFAULT_MATH`).

### Validation

- [x] **2A.9** Build + ctest: `cmake -B build && cmake --build build -j
  && ctest --test-dir build`. All existing checks still pass; new fp16
  + tfloat cuBLAS tests pass. (879 checks, 0 failures — up from 875;
  the 4 new checks are the math-mode restore assertions in
  `test_cublas_gemm_tfloat`.)
- [x] **2A.10** `./build/gemm_y` end-to-end: three sweeps run (bf16,
  fp16, tf32), three CSVs + three `.meta` sidecars written to
  `results/`. bf16 sweep produces 28 rows as before; fp16 and tf32
  sweeps produce 14 rows each (cuBLAS only — no custom kernel
  registered). All rows PASS (cuBLAS vs cuBLAS, err == 0).

---

## Phase 2B — Visualization (Python + Plotly + Dash + SQLite)

Goal: interactive dashboard for benchmark results. Decoupled from C++ —
consumes the CSV + `.meta` sidecar produced by `./build/gemm_y`, stores
in SQLite, serves a Dash app at `localhost:8050`.

**Stack:** Plotly (interactive hover), Dash (reactive checkboxes/toggles
+ built-in Flask server), SQLite (`sqlite3` stdlib, queryable, git-trackable).
No matplotlib. DB lives at `db/gemm_y.db`, tracked in git, declared binary
in `.gitattributes`. CSVs in `results/` stay gitignored (regenerable).

**Workflow:**
```sh
./build/gemm_y                    # writes results/bench_<arch>_<dtype>.csv + .meta
source pyenv/bin/activate
python scripts/ingest.py results/bench_sm_120_bf16.csv [--label "..."]
python scripts/server.py          # localhost:8050
python scripts/dump_db.py         # optional JSONL export for human inspection
```

`ingest.py` auto-discovers the `.meta` sidecar (same path, `.meta`
extension). Captures `git_sha` via `git rev-parse --short HEAD` at ingest
time (use case TBD; captured for traceability).

### Tasks

- [x] **2B.1** `.gitattributes` (new file): `db/gemm_y.db binary`. No
  `.gitignore` changes for `db/` or `results/` (both already in desired
  state: `db/` tracked, `results/` ignored). Added `db/dump.jsonl` to
  `.gitignore` (regenerable JSONL view; the DB is the source of truth).
- [x] **2B.2** `scripts/requirements.txt`: `dash>=2.14`, `plotly>=5.18`,
  `pandas>=2.0`. Python 3.14 in `pyenv/`.
- [x] **2B.3** `scripts/db.py`: SQLite schema + query layer (pure
  functions, no Dash dependency). Schema:
  ```sql
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
  ```
  `tol` lives in `runs` (per-run, from the `.meta` sidecar), not per-
  measurement. No `pass` column — every measurement in the DB passed by
  construction (failed kernels were skipped before CSV write).
  `is_cublas` derived in Python at query time (`kernel_name == 'cublas'`).
- [x] **2B.4** `scripts/ingest.py`: CLI `python ingest.py <csv> [--label
  <name>]`. Reads the CSV + auto-discovered `.meta` sidecar, appends one
  row to `runs` (with `ingested_at` timestamp, `git_sha` from
  `git rev-parse --short HEAD`, optional `label`), appends N rows to
  `measurements`. Idempotent guard: refuse to re-ingest the same
  `(source_csv, source_meta, git_sha)` tuple unless `--force`.
  **Implementation note (2026-07-19):** `source_csv`/`source_meta` are
  stored verbatim (path as given on the CLI) so the guard matches the
  user's mental model of "same command = same ingest". CSV parsing uses
  the stdlib `csv` module (no pandas dep for ingest itself).
- [x] **2B.5** `scripts/server.py`: Dash app at `localhost:8050`.
  Single-page layout, sidebar + tabbed content:
  - **Sidebar**: arch radio (sm_120 / sm_90), dtype checklist
    (bf16 / fp16 / tf32), kernel checklist (Custom / cuBLAS), runs
    multi-select dropdown (populated from `runs` table), scale radio
    (log-log / linear).
  - **Tab 1 — Timing**: `kernel_median_ns` vs `N`, one line per
    (run, kernel). Default log-log; toggle to linear. Hover shows:
    arch, dtype, custom/ref, kernel name, kernel desc, N, median_ns,
    ref_median_ns, speedup vs cuBLAS.
  - **Tab 2 — Accuracy**: `max_rel_err` vs `N` per kernel. Horizontal
    dashed line at `tol` (from the run's metadata). No error bars.
  - **Tab 3 — Run History**: table of all runs in the DB (ingested_at,
    git_sha, label, arch, dtype, kernel count). Selectable for
    comparison.
  Plotly `hovertemplate` with `customdata` carrying
  `[arch, dtype, is_cublas, kernel_desc]`.
- [x] **2B.6** `scripts/dump_db.py`: export DB to JSONL (one line per
  measurement, with run metadata joined). Output to stdout by default;
  `-o <path>` writes to a file (e.g. `db/dump.jsonl`, gitignored).

### Validation

- [x] **2B.7** End-to-end: `./build/gemm_y` → ingest all three CSVs →
  launch dashboard → verify all three dtypes appear, cuBLAS lines
  visible, hover shows correct metadata, log/linear toggle works,
  accuracy tab shows tol line. **Validated 2026-07-19:** 3 runs ingested
  (28 + 14 + 14 = 56 measurements), dashboard serves HTTP 200 on
  `localhost:8050`, tab-switch callbacks fire (POST `_dash-update-component`
  200s in logs), all three dtypes present in DB, both `cublas` and `naive`
  kernels visible. C++ build + ctest still pass (879 checks, 0 failures).
  Visual hover/toggle verification is a browser-side check; the HTTP
  200s + callback responses confirm the data layer is wired correctly.

---

## Phase 2C — Naive variants for fp16 / tfloat + tiled kernel (owner: user)

Goal: give every dtype a custom kernel to compare against cuBLAS, then
land the first tensor-core tiled kernel for bf16.

### Done (this session)

- [x] **2C.3** `NaiveGemm<fp16>` and `NaiveGemm<tfloat>` enabled. The
  `NaiveGemm<T>::operator()` template is dtype-agnostic (casts to fp32
  for accumulation, back to T on write), so enabling fp16/tfloat was a
  3-line change: added `template struct NaiveGemm<__half>;` and
  `template struct NaiveGemm<float>;` explicit instantiations to the
  existing `src/sm120/gemm_bf16_naive.cu` and `src/sm90/gemm_bf16_naive.cu`
  (one definition serves all three dtypes — no new files, no duplication).
  `main.cpp` registers `NaiveGemm<T>` for all three sweeps. Each CSV now
  has 28 rows (14 sizes × 2 kernels: cuBLAS + naive), all PASS.

### Failed / reverted (this session)

- [ ] **2C.1 / 2C.2** ~~`src/sm120/gemm_bf16_tiled_128.cu` (+ `.cuh`) —
  128×128 tile, 8 warps/CTA, `nvcuda::wmma` 16×16×16 bf16 fragments, fp32
  accum.~~ **REVERTED (2026-07-19).** A first attempt at the tiled
  tensor-core kernel was written and built clean, but failed accuracy:
  warps 0–3 (rows 0–63) produced correct output; warps 4–7 (rows 64+)
  produced **zero** output. `max_rel_err = 1.0` at row 64 for N≥128
  (N=64 passed only because rows 64+ are out of bounds and the zero
  output happened to match the bounds-skipped store).

  **What was tried (all produced the same failure):**
  1. Row-major smem A + col-major smem B, `matrix_a` tagged `row_major`.
     Wrong — the smem layout / fragment tag combination was inconsistent.
  2. Col-major smem A + col-major smem B, both fragments tagged
     `col_major`, `ld` matching the col-major stride. Compute should be
     correct by manual trace, but warps 4–7 still produced zero.
  3. Per-warp 16×16 accumulator smem slots (avoiding inter-warp races on
     a shared accumulator buffer). Same failure.
  4. Full 128×128 accumulator smem (64 KB) with
     `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`
     to opt in past the 48 KB default. Same failure.

  **Hypothesis:** the bug is not in the store path (multiple store
  strategies all failed identically) and not in the smem layout (col-major
  trace is correct). The most likely remaining cause is a misunderstanding
  of the `nvcuda::wmma` fragment load semantics for warps beyond the first
  16-row tile — possibly the `matrix_a` fragment load with `ld = kTileM`
  and a non-zero row base does not do what the manual trace suggests, or
  there is an undocumented constraint on the `ld` stride for `col_major`
  fragments. A device-side `printf` of `c_frag[0].x[0..3]` for warp 4 was
  the next debugging step but was not completed.

  **Files removed:** `src/sm120/gemm_bf16_tiled_128.{cu,cuh}` and
  `src/sm90/gemm_bf16_tiled_128.{cu,cuh}`. The `<mma.h>` include in
  `cuda_compat.h` was reverted. `main.cpp` and `tests/test.cu` were
  reverted to pre-tiled-kernel state. The build is clean and ctest passes
  (879 checks, 0 failures).

  **Next step (owner: user):** re-attempt the tiled kernel from scratch.
  Suggested debugging approach: start with a single-CTA, single-warp
  kernel (16×16×16 output) and confirm correctness, then scale to 8
  warps (16×128 strip per warp), then scale to the full 128×128 tile.
  The per-warp isolation will pinpoint whether the bug is in the compute
  or the store. Alternatively, use `mma.sync` PTX directly instead of
  `nvcuda::wmma` — the PTX gives explicit control over the fragment
  layout and may be easier to reason about.

- [ ] **2C.4** Run sweep, ingest to DB, view in dashboard, compare vs
  cuBLAS line. Iterate (one variable per commit per AGENTS.md experiment
  discipline).
- [ ] **2C.5** Once bf16 tiled kernel is competitive, replicate ideas to
  fp16 (Path 1 sibling). Expect near-identical perf on tensor cores.

---

## Phase 2 prep (deferred, not implemented in 2A)

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
