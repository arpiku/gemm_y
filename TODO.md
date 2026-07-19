# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

## Phase 1.7 — ctest wiring, `-Werror` fix, `GemmArgs` const-correctness

Goal: close the two infra gaps surfaced by Phase 1.6 validation
(`ctest` not wired; `-DENABLE_WERROR=ON` broken by a `CMakeLists.txt`
flag-joining bug) and tighten the kernel ABI so the type system
catches accidental writes to `A`/`B` and silent RowMajor misconfiguration.
Small, surgical phase; no algorithm or perf work.

**Context (carried from Phase 1.6):**
- `test_cuda` is a plain executable with a hand-rolled `main()` calling
  15 test functions sequentially. `CMakeLists.txt` has no
  `enable_testing()` / `add_test()`, so `ctest --test-dir build` reports
  "No tests were found". Minimal fix (Option A): register the existing
  binary with CTest. Full per-test split (Option B) is premature for 15
  tests — defer until the count grows past ~30.
- `-DENABLE_WERROR=ON` fails with `nvcc fatal : Value '-Xcompiler=-Wall'
  is not defined for option 'Werror'`. Root cause: in `CMakeLists.txt`,
  `-Werror` is appended to `_gemm_y_nvcc_xcompiler_warnings` and then
  joined into one `-Xcompiler=<a;b;c;-Werror>` arg, which nvcc misparses.
  Pre-existing (confirmed by stashing the Phase 1.6 refactor and
  rebuilding on baseline — same failure). Not a regression from 1.6.
- `GemmArgs<T>` currently holds `A`, `B`, `C` all as
  `MatrixView<T, Space::Device>` (writable). `A` and `B` are kernel
  *inputs* — the kernel only reads them — but nothing in the type
  system prevents an accidental write. `MatrixView` already has the
  converting constructor `MatrixView<T, S> -> MatrixView<const T, S>`
  (line 49–55), so const-ifying `A`/`B` in `GemmArgs` is a 3-line change
  with zero call-site churn (implicit conversion at the Profiler
  boundary). `C` stays mutable (it's the output).
- `NaiveGemm` (and future kernels) hardcode ColMajor addressing
  (`A.ptr[i + k*A.ld]`) but never assert `layout == ColMajor`. A RowMajor
  view entering the kernel path would produce wrong results silently.
  One `GEMM_Y_ASSERT` per launch (host-side, in `operator()`) closes
  this — debug-only, zero runtime cost.

### CMakeLists.txt — ctest

- [x] **1.7.1** Add `enable_testing()` near the top of `CMakeLists.txt`
  (after `project()`, before target definitions). Add
  `add_test(NAME test_cuda COMMAND test_cuda)` after the `test_cuda`
  target is defined. Verify `ctest --test-dir build` now runs `test_cuda`
  as a single test entry and propagates its exit code. No changes to
  `test.cu`'s hand-rolled `main()` — Option A (minimal ctest wiring).
  ```cmake
  enable_testing()
  # ... after add_executable(test_cuda ...) ...
  add_test(NAME test_cuda COMMAND test_cuda)
  ```

### CMakeLists.txt — `-Werror` for nvcc

- [x] **1.7.2** ~~Fix `-DENABLE_WERROR=ON` for nvcc.~~ **Simplified
  (2026-07-19): removed the `ENABLE_WERROR` option entirely.** nvcc's
  `-Werror` requires a `<kind>` argument (e.g. `-Werror all-warnings`);
  the bare `-Werror` form greedily consumes the next flag as its kind,
  breaking the build regardless of whether it's passed via `-Xcompiler`
  or as a standalone nvcc-native flag. Every attempted fix (move to
  nvcc-native, pass as separate arg, use `--Werror all-warnings`) either
  broke the build or required maintaining two code paths for marginal
  benefit. The strict warning set (`-Wall -Wextra -Wpedantic -Wshadow
  -Wconversion ...` for host CXX, reduced subset for nvcc host compiler,
  `-Wreorder -Winit-self` for nvcc native) is compiled in by default;
  CI/local vigilance catches warnings without `-Werror`. `AGENTS.md`
  updated to remove the `-DENABLE_WERROR=ON` build command.

### GemmArgs.h — const-correctness

- [x] **1.7.3** Change `GemmArgs<T>` so `A` and `B` are read-only:
  ```cpp
  template <typename T>
  struct GemmArgs {
      MatrixView<const T, Space::Device> A;  // input, read-only
      MatrixView<const T, Space::Device> B;  // input, read-only
      MatrixView<T,       Space::Device> C;  // output, mutable
  };
  ```
  Relies on `MatrixView`'s existing converting constructor (line 49–55):
  `MatrixView<T, S>` -> `MatrixView<const T, S>` is implicit, so
  `Profiler.cu` / `test.cu` / `main.cpp` call sites passing writable
  views for `A`/`B` compile unchanged (implicit const-conversion at the
  `GemmArgs` construction site). If any call site fails to compile, the
  failure is the type system catching a real bug (e.g. someone trying to
  write through `A` or `B`) — investigate, do not silence.

  **Implementation note (2026-07-19):** `cublas_gemm` was NOT const-ified
  (despite the same read-only rationale applying). `cublas_gemm` is a
  function template, and C++ template argument deduction does not
  consider implicit conversions — so `MatrixView<T,S>` ->
  `MatrixView<const T,S>` (via MatrixView's converting constructor) fails
  deduction at every call site. The const contract is enforced at the
  `GemmArgs` level (NaiveGemm and future kernels take `GemmArgs<T>`
  with const A/B); `cublas_gemm` is the reference path and takes
  writable views for API simplicity. Documented in `cublas_gemm.h`.
- [x] **1.7.4** Update `NaiveGemm` kernel signature to accept
  `MatrixView<const T, Space::Device>` for `A` and `B`. The kernel body
  (`A.ptr[i + k*A.ld]`, `B.ptr[k + j*B.ld]`) compiles unchanged because
  `const T` is readable. `C` stays `MatrixView<T, Space::Device>`.
  ```cpp
  template <typename T>
  __global__ void naive_gemm_kernel(MatrixView<const T, Space::Device> A,
                                    MatrixView<const T, Space::Device> B,
                                    MatrixView<T,       Space::Device> C);
  ```
  Repeat the same const-ification for every future kernel as it lands
  (Phase 2C tiled kernels, etc.) — this is the new ABI contract.

### Kernel layout invariant

- [x] **1.7.5** Add a debug-only ColMajor assertion in `NaiveGemm::operator()`
  on the host, before the kernel launch:
  ```cpp
  template <typename T>
  void NaiveGemm<T>::operator()(GemmArgs<T> args, cudaStream_t) const {
      GEMM_Y_ASSERT(args.A.layout == Layout::ColMajor &&
                    args.B.layout == Layout::ColMajor &&
                    args.C.layout == Layout::ColMajor,
                    "NaiveGemm assumes ColMajor inputs");
      // ... existing launch ...
  }
  ```
  One assert per launch (not per element). Debug-only, zero runtime cost
  in Release. Catches silent misconfiguration if RowMajor views ever
  enter the kernel path. Repeat for every future kernel as it lands.

### Validation

- [x] **1.7.6** Build + ctest: `cmake -B build && cmake --build build -j
  && ctest --test-dir build`. Verify `test_cuda` runs as a ctest entry
  (1.7.1), exit code propagates, all 875 checks still pass.
- [x] **1.7.7** ~~Strict build: `cmake -B build -DENABLE_WERROR=ON &&
  cmake --build build -j`.~~ **N/A (2026-07-19):** `ENABLE_WERROR`
  removed entirely (see 1.7.2). Default build is clean under the full
  strict host warning set; no warning regressions introduced by the
  refactor.
- [x] **1.7.8** `./build/gemm_y` end-to-end: full 14-size bf16 sweep runs
  identically to before — 28 rows, all PASS, CSV written. Behavior
  preserved (the const-ification is a type-level change, not a runtime
  behavior change).

---

## Phase 1.8 — Comment clarity pass (no logic change)

Goal: address the `MatrixView` opacity concern with a documented
dual-use contract; trim verbose comments and phase trailers
(`Phase X.Y`, `RXX`, `Chunk X.X`) across the codebase. Comment-only —
zero code logic changes, zero behavior change. `Layout` enum kept
(future RowMajor); `MatrixView` type unchanged (no split, no rename,
no field changes); `Tracer.h` kept; ColMajor asserts (1.7.5) kept;
`copy_kind_v` poison primary + `static_assert` kept (machinery correct).

### Additions (clarity)

- [x] **1.8.1** `MatrixView.h` header: add a dual-use contract comment.
  The type serves (1) host-side view (with `block`/`operator()`/
  `is_contiguous`/converting ctor) and (2) kernel-side POD descriptor
  (only `ptr`/`rows`/`cols`/`ld` read directly). Explicitly note the
  host methods are **not** `__device__`-callable; kernels read fields
  directly. Makes the "opaque properties" concern explicit at the type
  definition.
- [x] **1.8.2** `MatrixView.h::block`: one-line arg-order comment —
  `// block(row, col, rows, cols) — BLAS convention; ld is unchanged.`
- [x] **1.8.3** `NaiveGemm` sm90 + sm120 `.cu`: one-line comment at
  `naive_gemm_kernel` signature — `// MatrixView used as POD descriptor
  — only ptr/rows/cols/ld are read.`

### Removals (verbosity + phase trailers)

- [x] **1.8.4** `CudaCheck.h`: trim header from ~20 lines to ~10 (one
  layer of explanation, not three). Drop per-helper comments that
  restate the function body. Drop `Phase 1.6.x` trailers.
- [x] **1.8.5** `Copy.h`: drop all `Phase 1.6.x` trailers. Trim
  `copy_kind_v`/`plan_copy`/`copy` comments to essentials. Keep the
  machinery (poison primary + `static_assert`), just less prose.
- [x] **1.8.6** `cublas_gemm.h`: trim 8-line const-ification note to
  ~4 lines. Trim the `Phase 1 invariant` comment on the layout check.
- [x] **1.8.7** `CMakeLists.txt`: trim `-Werror` comment from 6 lines
  to 3. Drop `Phase 1.7.1` trailers from CTest comments.
- [x] **1.8.8** `test.cu`: trim `test_copy_kind_compile_time` block
  comment. Drop phase trailers from file header.
- [x] **1.8.9** `Profiler.cu` + `Profiler.h`: drop `Phase 1.5 (R3)`/
  `Phase 1.5 (R4)`/`R15` trailers. Keep durable explanations.
- [x] **1.8.10** `NaiveGemm` sm90 + sm120 `.cu`: drop `Phase 1.7.5`
  trailers from ColMajor assert comments. Keep rationale (one line).
- [x] **1.8.11** Sweep remaining files for `RXX`/`Phase X.Y`/
  `Chunk X.X` trailers; trim where they add no durable value. Rule:
  drop the trailer tag, keep the durable explanation. Files:
  `Buffer.h`, `bench/Stats.h` (R12), `bench/Fill.h` (R13),
  `bench/Accuracy.h` (Phase 1/2 refs), `bench/microbench/
  memcpy_microbench.cu` (Chunk 2.2, Phase 1.5 R10),
  `bench/microbench/launch_overhead_microbench.cu` (Chunk 2.4, R10),
  `bench/microbench/print_table.h` (R10), `main.cpp` (Phase 1.5/Phase 1),
  `Arch.h` (R11).

### Validation

- [x] **1.8.12** Build + ctest: `cmake -B build && cmake --build build -j
  && ctest --test-dir build`. Verify all 875 checks still pass.
  Comment-only change; build should be a no-op.

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
