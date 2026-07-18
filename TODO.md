# TODO.md

> Per-branch task list. Edit at the start of each branch with planned steps;
> check off as you go. Do not carry over completed items from previous
> branches. Durable project state lives in `AGENTS.md`.
>
> Architecture rationale: see `ARD.md`.

## DONE — Phase 1 (`phase1/infra`)

Foundation, matrix DS, timing, cuBLAS bf16 reference, profiler, harness
validation via naive bf16 kernel. Full sweep produces clean CSV on sm_120.

- [x] Foundation: `cuda_compat.h`, `CudaCheck.h`, cuBLAS link, `Tracer.h` comment.
- [x] Matrix DS: `Space`, `Layout`, `Buffer`, `MatrixView`, `Matrix`, `Copy` + tests.
- [x] Timing: `CudaTimer`, memcpy microbench, launch overhead microbench.
- [x] cuBLAS reference: `CublasHandle`, `cublas_gemm` (bf16), unit test.
- [x] Profiler: `GemmArgs`, `KernelTraits`, `Profiler`, `Accuracy`, `CsvWriter`, CLI.
- [x] Harness validation: `NaiveGemm<bf16>` on sm120 + sm90, full sweep, ARD §10.

---

## Phase 1.5 — Refactor (cleanups from code review)

Goal: fix correctness issues (sync memcpy, cuBLAS re-timing), decouple
Profiler from CSV, deduplicate, improve test coverage and microbench
readability. No new features. See ARD §13 for rationale per item.

### Core fixes
- [ ] **R1** `Copy.h`: revert to sync `cudaMemcpy` / `cudaMemcpy2D`.
  Drop `cudaStream_t` param + `cudaStreamSynchronize` calls. Rationale:
  async-on-pageable-host is silently sync + extra staging overhead
  (microbench confirmed sync ≈ async). Pinned memory deferred to Phase 2.
- [ ] **R2** `Copy.h`: extract `detail::plan_copy(dst, src)` computing
  `{width_bytes, height, spitch, dpitch, contiguous}` from shape/layout
  checks. `copy_h2d` / `copy_d2h` become 5-line wrappers over
  `detail::copy_contiguous` / `detail::copy_strided`. Eliminates ~50 lines
  of duplication.
- [ ] **R3** `Profiler`: decouple `run_sweep` from CSV. Return
  `SweepResult { std::vector<SweepRow> rows; }`. `main.cpp` owns `CsvWriter`
  and iterates `result.rows`. `Profiler.cu` drops `#include "CsvWriter.h"`.
- [ ] **R4** `Profiler`: measure cuBLAS **once per N**, store in
  `SweepResult`, reuse `ref_*` columns for all kernels at that N. Current
  code re-times cuBLAS per registered kernel (K × S × kTimed extra launches).

### Header hygiene
- [ ] **R5** `cuda_compat.h`: add `#include <cublas_v2.h>` under the
  diagnostic pragma. Remove direct `<cublas_v2.h>` includes from
  `CudaCheck.h`, `CublasHandle.h`, `cublas_gemm.h`.
- [ ] **R6** `Buffer.h`: fix misleading "64-byte alignment" comment —
  `std::vector` default allocator gives ~16 bytes. State: default
  alignment sufficient for Phase 1; swap in aligned allocator later
  without touching call sites.

### Test suite
- [ ] **R7** `test.cu`: remove `test_smoke` (redundant, RAII violation via
  raw `cudaMalloc`/`cudaFree`). Trim `test_buffer_device` to Buffer
  invariants only (drop round-trip — covered by Copy tests).
- [ ] **R8** `test.cu`: strengthen `test_cuda_timer` (assert `0 < ms < 100`
  after empty kernel). Add:
  - `test_matrixview_const_conversion` (T → const T compiles; reverse is
    a compile error — document via comment).
  - `test_matrix_view_from_matrix` (non-const `view()` mutable; const
    `view()` read-only).
  - `test_cublas_gemm_bf16_strided` (N=32 submatrix of 64×64, ld=64 —
    catches the bench runner's exact dependency).
  - `test_naive_gemm_bf16` (N=32 vs host fp32 ref, `max_rel_err ≤ 1e-3`).
  - `test_profiler_run_sweep_small` (post-R3: register NaiveGemm, sweep
    `{32,64}`, assert 4 rows, all `max_rel_err ≤ kRelErrTol`).

### Microbench readability + relocation
- [ ] **R9** Move `memcpy_microbench.cu` + `launch_overhead_microbench.cu`
  to `src/bench/microbench/` subdirectory. CMake: glob
  `src/bench/*.cu` (non-recursive) for `gemm_y` target; glob
  `src/bench/microbench/*.cu` for per-file `EXCLUDE_FROM_ALL` targets.
  Cleaner than the current `list(FILTER ... REGEX "microbench\.cu$")`.
- [ ] **R10** Add `src/bench/microbench/print_table.h` — tiny helper for
  aligned fixed-width column output. Refactor both microbenches to print
  human-readable tables (header row + separator + aligned columns, units
  in header). Drop raw CSV-to-stdout (microbenches are one-off, not
  plotted in Phase 1.5; future plotting util will consume structured output
  if needed).

### Deduplication / extraction
- [ ] **R11** Extract `src/Arch.h` — single `kArchName` definition
  (currently duplicated in `main.cpp`, `test.cu`, `Profiler.cu`).
- [ ] **R12** Extract `src/bench/Stats.h` — `TimedStats` + `summarize_ns`
  (currently duplicated in `Profiler.cu` as `TimedStats` and
  `memcpy_microbench.cu` as `Stats`).
- [ ] **R13** Extract `src/bench/Fill.h` — `fill_sequential(A, B)` host
  fill pattern (currently duplicated in `Profiler.cu` and `test.cu`).
- [ ] **R14** Extract `src/dtypes.h` — `dtypes::name<T>()` returning
  `string_view`. Replace `dtype_name<T>()` in `Profiler.cu`. Co-locate
  with `dtypes::bf16/fp16/fp32` aliases (move from `cuda_compat.h`).

### Minor correctness / style
- [ ] **R15** `Profiler.cu`: `h2d_ns` column — repeat global H2D value in
  every row, not `0.0` (current `0.0` is misleading).
- [ ] **R16** `Profiler.cu`: `Timer<>` default capacity (drop `<4096>` —
  only 3 marks used).
- [ ] **R17** `CudaTimer::elapsed_ms()` — mark `const`.
- [ ] **R18** `MatrixView::block()` — add `#ifndef NDEBUG` bounds asserts
  (`r+m <= rows`, `c+n <= cols`).
- [ ] **R19** `cublas_gemm.h`: extract `GEMM_Y_ASSERT(cond, msg)` macro
  to `CudaCheck.h`. Layout check → debug-only assert (Phase 1 invariant,
  not a runtime API contract).
- [ ] **R20** `CudaCheck.h`: rename macro local vars `_gemm_y_err` →
  `gemm_y_err_` (suffix-underscore convention; avoids reserved-identifier
  edge cases).

---

## Phase 2A — cuBLAS references for all paths

Goal: reference kernels available for all three paths so any custom kernel
variant can be benchmarked against the right cuBLAS baseline.

**Three paths (priority order):**
1. **Path 1 (primary)**: fp16 + bf16, tensor cores, fp32 accum. fp16 and
   bf16 should show no meaningful perf difference on Hopper/Blackwell
   tensor cores — optimize one, replicate ideas to the other.
2. **Path 2 (secondary)**: fp32 storage, tf32 compute, tensor cores.
3. **Path 3 (tertiary, may skip optimization)**: fp32 pedantic, CUDA cores.
   Reference must exist for comparison even if no custom kernel is written.

- [ ] **2A.1** `cublas_gemm.h`: verify fp32-pedantic path (CublasTypeMap<float>
  already maps to CUDA_R_32F data + compute, CUBLAS_DEFAULT_MATH). No code
  change expected — confirm via unit test.
- [ ] **2A.2** `cublas_gemm.h`: add `cublas_gemm_tf32(handle, A, B, C, stream)`
  — `float` storage, `CUBLAS_TF32_CUBLAS_MATH` compute, tensor cores.
  Distinct entry point (not overload) because storage dtype is identical
  to pedantic fp32; the difference is the math mode.
- [ ] **2A.3** `CublasHandle`: add `WithMathMode` RAII guard — ctor sets
  mode, dtor restores. Used by `cublas_gemm_tf32`. Cleaner than manual
  save/restore; documents the non-thread-safe toggle.
- [ ] **2A.4** `Profiler<float>`: register `CublasTf32Reference` functor
  (wraps `cublas_gemm_tf32`) alongside the pedantic fp32 reference. Two
  reference rows per N in the fp32 CSV (pedantic + tf32) — shows the
  tensor-core speedup ceiling for fp32 inputs.
- [ ] **2A.5** `main.cpp`: extend to run sweeps for bf16, fp16, fp32
  (pedantic + tf32). One `Profiler<T>` per storage dtype; one CSV per
  `(arch, dtype)`. tf32 rows live in the fp32 CSV.
- [ ] **2A.6** `Accuracy.h`: per-dtype `kRelErrTol` specializations:
  bf16 → 1e-2, fp16 → 1e-3, fp32 pedantic → 1e-5, tf32 → 1e-3.
- [ ] **2A.7** `test.cu`: add `test_cublas_gemm_fp16`, `test_cublas_gemm_fp32`
  (pedantic), `test_cublas_gemm_tf32`. tf32 test must verify math-mode
  restore (run pedantic → tf32 → pedantic, assert mode restored).

---

## Phase 2B — Plotting (parallel to 2A, no C++ dependency)

Goal: visual feedback during kernel optimization. Log-log plot, one
subplot per `(arch, dtype)`, one line per kernel, cuBLAS as reference.

**Future scope**: the plotting util will be extended in later phases to
also plot microbench data and other structured outputs. Design it to
consume CSV with a flexible schema, not hardcode the bench-CSV columns.

- [ ] **2B.1** `scripts/plot.py` — Python + matplotlib. Reads
  `results/bench_<arch>_<dtype>.csv`. Outputs `results/plot_<arch>_<dtype>.png`.
- [ ] **2B.2** `scripts/requirements.txt` — `matplotlib>=3.7`, `pandas>=2.0`.
- [ ] **2B.3** CLI: `python scripts/plot.py <csv>` → single plot;
  `python scripts/plot.py <dir>` → all CSVs in dir, one PNG each.
- [ ] **2B.4** Plot spec: X = N (log), Y = kernel_median_ns (log). One
  line per `kernel_name`, label = `kernel_desc`. cuBLAS line dashed
  (distinct color, e.g. black). Custom kernels: solid, Okabe-Ito palette.
  Title: `GEMM <dtype> on <arch> (lower is better)`. Legend top-right,
  sorted by median at largest N.
- [ ] **2B.5** Structure the CSV-reading layer as a reusable module
  (e.g. `scripts/csv_loader.py`) so future plot types (microbench, etc.)
  can reuse it.

---

## Phase 2C — First tiled bf16 kernel (after 2A + 2B)

Goal: first tensor-core kernel for bf16, beat cuBLAS at large N. Detailed
plan in a separate `bf16_tiling_128` branch TODO.

- [ ] **2C.1** `src/sm120/gemm_bf16_tiled_128.cu` (+ `.cuh`) — 128×128
  tile, 8 warps/CTA, `wmma`/`mma.sync` for bf16 on sm_120.
- [ ] **2C.2** `src/sm90/gemm_bf16_tiled_128.cu` (+ `.cuh`) — same
  algorithm, sm_90 `wmma` API.
- [ ] **2C.3** Register in `main.cpp` alongside `NaiveGemm<bf16>`. Run
  sweep, plot, compare vs cuBLAS line. Iterate (one variable per commit
  per AGENTS.md experiment discipline).
- [ ] **2C.4** Once bf16 tiled kernel is competitive, replicate ideas to
  fp16 (Path 1 sibling). Expect near-identical perf on tensor cores.

---

## Phase 2 prep (deferred, not implemented in 1.5 or 2A)

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
