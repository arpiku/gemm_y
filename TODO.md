# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

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
