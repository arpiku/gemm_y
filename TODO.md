# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

## Phase 1.6 — Copy.h / CudaCheck.h dedup + Profiler D2H fix

Goal: remove code duplication in `Copy.h` and `CudaCheck.h`, tighten the
error-checking macros, and fix a redundant D2H copy in `Profiler.cu`.
Refactor only — no behavior change at the public API (`copy_h2d` /
`copy_d2h` keep their signatures; both `cudaMemcpy` and `cudaMemcpy2D`
paths are preserved). Driven by code review; precedes Phase 2A.

**Context:**
- `Copy.h` has two byte-identical `plan_copy` overloads (Space tags
  swapped) and two near-identical `copy_h2d`/`copy_d2h` bodies (only
  `cudaMemcpyKind` differs). The Space tags enforce direction at the
  caller site but are unused in the planning body — the duplication is
  the symptom of not pushing the Space tag down to a single template
  parameter and deducing `kind` from it at compile time.
- `cudaMemcpy2D` IS used: every `copy_d2h` of a `.block(0,0,N,N)` from
  the 4096-ld pre-alloc buffer takes the strided path for 13 of 14 sweep
  sizes (only N=4096 is contiguous). Removing it would break the ARD §5
  pre-alloc + submatrix-slicing design. Keep both paths.
- `CudaCheck.h` has `GEMM_Y_CUDA_CHECK_IMPL_` (used once, indirection
  buys nothing) and three macros (`CUDA_CHECK`, `CUDA_CHECK_LAST_ERROR`,
  `CUBLAS_CHECK`) that each inline the same `fprintf`+`abort` tail.

### CudaCheck.h

- [x] **1.6.1** Delete `GEMM_Y_CUDA_CHECK_IMPL_`. Add two `[[noreturn]]
  noexcept` helpers in `namespace gemm_y::detail`:
  ```cpp
  [[noreturn]] inline void fail_cuda(cudaError_t e, const char* tag,
                                     const char* file, int line) noexcept;
  [[noreturn]] inline void fail_cublas(cublasStatus_t s,
                                     const char* file, int line) noexcept;
  ```
  Each holds the `fprintf`+`abort` tail exactly once. `fail_cuda` is
  shared by `CUDA_CHECK` and `CUDA_CHECK_LAST_ERROR` (only the tag
  string differs: `"CUDA error"` vs `"CUDA async error"`).
- [x] **1.6.2** Rewrite `CUDA_CHECK`, `CUDA_CHECK_LAST_ERROR`,
  `CUBLAS_CHECK` as 5-line macros delegating to the `detail::fail_*`
  helpers. Use fully-qualified `::gemm_y::detail::fail_*` so the macros
  expand correctly whether used inside or outside `namespace gemm_y`
  (matches existing usage in `Buffer.h`, `CudaTimer.h`, etc.). Keep the
  `e_` / `s_` suffix-underscore locals (R20 hygiene).
  ```cpp
  #define CUDA_CHECK(expr)                                                            \
      do {                                                                            \
          const cudaError_t e_ = (expr);                                              \
          if (e_ != cudaSuccess)                                                      \
              ::gemm_y::detail::fail_cuda(e_, "CUDA error", __FILE__, __LINE__);      \
      } while (0)
  ```
- [x] **1.6.3** Tighten `GEMM_Y_ASSERT` NDEBUG branch to
  `do { (void)sizeof(cond); } while (0)` — keeps the condition parsed
  for warnings under NDEBUG (standard `assert` hygiene; current
  `do { } while (0)` silently drops syntax errors in the condition).
  Keep the name `GEMM_Y_ASSERT` (project-scoped, only project macro).
  Debug branch unchanged.

### Copy.h

- [x] **1.6.4** Collapse the two `plan_copy` overloads into one:
  ```cpp
  template <Space Dst, Space Src, typename T, typename U>
  CopyPlan plan_copy(MatrixView<T, Dst> dst, MatrixView<U, Src> src);
  ```
  Body unchanged. Space tags are caller-enforcement only; the body
  reads `rows`/`cols`/`ld`/`layout`/`is_contiguous()` — Space-agnostic.
- [x] **1.6.5** Add `detail::copy_kind_v<Dst, Src>` constexpr table:
  ```cpp
  template <Space Dst, Space Src>
  constexpr cudaMemcpyKind copy_kind_v = cudaMemcpyKind(-1);  // poison
  template <> constexpr cudaMemcpyKind copy_kind_v<Space::Device, Space::Host> = cudaMemcpyHostToDevice;
  template <> constexpr cudaMemcpyKind copy_kind_v<Space::Host,   Space::Device> = cudaMemcpyDeviceToHost;
  ```
  Other combos intentionally undefined → compile error if used
  (e.g. `copy` with two Host views won't link, catching misuse).

  **Implementation note (2026-07-19):** the primary template with a
  poison value makes wrong-direction instantiations *compile* (the
  poison value is well-formed) and only fail at *link time* (ODR-use
  of an undefined specialization) — or worse, silently pass the poison
  to `cudaMemcpy` at runtime. To get the *compile-time* error that
  1.6.11 expects ("the poison specialization makes `copy<Host, Host>`
  and `copy<Device, Device>` a compile error"), add a `static_assert`
  inside `detail::copy` rejecting invalid `(Dst, Src)` pairs:
  ```cpp
  static_assert((Dst == Space::Device && Src == Space::Host) ||
                (Dst == Space::Host   && Src == Space::Device),
                "copy: only Host<->Device directions are supported");
  ```
  Keep the poison primary template as a defensive secondary net and
  to satisfy 1.6.11's `static_assert(copy_kind_v<...> == ...)` test
  for the valid specializations.
- [x] **1.6.6** Collapse `copy_h2d`/`copy_d2h` bodies into a single
  `detail::copy(dst, src)` with `kind` deduced via `copy_kind_v`.
  Delete `copy_contiguous`/`copy_strided` helpers (one-line
  `CUDA_CHECK` wrappers with no abstraction value); inline the two
  `CUDA_CHECK` calls directly into `detail::copy`. Both `cudaMemcpy`
  (contiguous) and `cudaMemcpy2D` (strided) paths preserved.
  ```cpp
  template <Space Dst, Space Src, typename T, typename U, /*SFINAE*/>
  void copy(MatrixView<T, Dst> dst, MatrixView<U, Src> src) {
      constexpr cudaMemcpyKind kind = copy_kind_v<Dst, Src>;
      const CopyPlan p = plan_copy(dst, src);
      if (p.contiguous) {
          CUDA_CHECK(cudaMemcpy(dst.ptr, src.ptr, p.width_bytes * p.height, kind));
      } else {
          CUDA_CHECK(cudaMemcpy2D(dst.ptr, p.dpitch, src.ptr, p.spitch,
                                  p.width_bytes, p.height, kind));
      }
  }
  ```
- [x] **1.6.7** Keep `copy_h2d`/`copy_d2h` as 2-line public wrappers
  delegating to `detail::copy` — preserves call-site API, no churn in
  `Profiler.cu` / `test.cu`. Do NOT add `inline` (templates are
  implicitly inline; the keyword is noise).
- [x] **1.6.8** Shape mismatch: keep `fprintf` + `abort` (runtime
  contract — caller bug, not invariant). Layout mismatch: keep
  `GEMM_Y_ASSERT` (debug-only — ARD §1 says bench runner guarantees
  ColMajor, so it's an invariant, not a runtime contract). No change
  from current behavior; just unified across the single `plan_copy`.

### Profiler.cu

- [ ] **1.6.9** ~~Fix redundant D2H in the debug OOB check~~ **REJECTED
  after analysis (2026-07-19).** The premise is incorrect: the line-189
  `copy_d2h(hC_ref_max.view().block(0,0,N,N), dC_ref)` is the *post-kernel*
  D2H that makes the OOB corruption check work. The line-108 D2H (timed,
  reported as `ref_d2h_ns`) captures `dC_ref` *before* the custom kernel
  runs; the line-189 D2H re-reads `dC_ref` *after* the kernel so that
  `memcmp(now, cref_snapshot)` can detect out-of-bounds writes that
  corrupted `C_ref_max`. Reusing the line-108 host snapshot (as the
  original TODO proposed) would make `now == cref_snapshot` always,
  silently breaking corruption detection — a behavior regression dressed
  as a refactor. The two D2Hs are not redundant; they serve different
  points in time. **Do not implement as written.** If the debug-build
  D2H cost becomes a real concern, the correct fix is a device-side
  comparison kernel against a device-side pre-kernel snapshot (new code,
  not a refactor) — out of scope for Phase 1.6. Leave `Profiler.cu`
  unchanged.

### Validation

- [~] **1.6.10** Build + test: `cmake -B build && cmake --build build -j
  && ctest --test-dir build`. Verify no warning regressions under the
  strict host warning set (`-Wall -Wextra -Wpedantic -Wshadow
  -Wconversion ...`); verify `-DENABLE_WERROR=ON` still compiles clean.

  **Partial (2026-07-19):** default build is clean under the full strict
  host warning set; `./build/test_cuda` passes 875 checks / 0 failures.
  Two sub-goals unverified — see the Phase 1.6 feedback section below
  for details (`ctest` not wired; `-DENABLE_WERROR=ON` has a pre-existing
  `CMakeLists.txt` bug unrelated to this phase).
- [x] **1.6.11** Add `test_copy_kind_compile_time` in `test.cu`:
  `static_assert` that `detail::copy_kind_v<Space::Device, Space::Host>`
  equals `cudaMemcpyHostToDevice` and the reverse — compile-time
  guarantee that wrong-direction calls fail to compile (the poison
  specialization makes `copy<Host, Host>` and `copy<Device, Device>` a
  compile error).

---

## Phase 1.6 feedback (2026-07-19)

Outcome of the refactor + validation, recorded here (rather than cut
from `TODO.md` per the AGENTS.md "completed work leaves TODO.md" rule)
because two validation sub-goals could not be cleanly verified and the
deviations need a durable record. Phase 1.6 is functionally complete;
the open items are tooling/infra, not refactor correctness.

### Implemented (10/11 steps)

- **1.6.1–1.6.3** (`CudaCheck.h`): `GEMM_Y_CUDA_CHECK_IMPL_` deleted;
  `gemm_y::detail::fail_cuda` / `fail_cublas` `[[noreturn]] noexcept
  inline` helpers hold the `fprintf`+`abort` tail once each; `CUDA_CHECK`,
  `CUDA_CHECK_LAST_ERROR`, `CUBLAS_CHECK` are 5-line macros delegating
  via `::gemm_y::detail::fail_*`; macro locals renamed to `e_` / `s_`
  (R20 hygiene); `GEMM_Y_ASSERT` NDEBUG branch tightened to
  `do { (void)sizeof(cond); } while (0)`.
- **1.6.4–1.6.8** (`Copy.h`): two `plan_copy` overloads collapsed into
  one `plan_copy<Space Dst, Space Src, T, U>`; `detail::copy_kind_v`
  constexpr table added; `copy_h2d`/`copy_d2h` bodies collapsed into
  `detail::copy` with `kind` deduced at compile time; `copy_contiguous`
  / `copy_strided` deleted (inlined the `CUDA_CHECK` calls);
  `copy_h2d`/`copy_d2h` kept as 2-line public wrappers — call-site API
  unchanged in `Profiler.cu` / `test.cu`.
- **1.6.11** (`test.cu`): `test_copy_kind_compile_time` added with three
  `static_assert`s pinning the H2D/D2H `copy_kind_v` specializations and
  their distinctness. Registered in `main()`.

### Rejected (1/11 steps)

- **1.6.9** (`Profiler.cu` redundant D2H): rejected after analysis. The
  line-189 `copy_d2h` is the *post-kernel* D2H that makes the OOB
  corruption check work; reusing the line-108 (pre-kernel) snapshot
  would make `now == cref_snapshot` always, silently breaking the
  detector. `Profiler.cu` unchanged. Full rationale in the step body.

### Deviations from the TODO (two, both mechanical)

1. **`inline constexpr` on `copy_kind_v`** (1.6.5). The TODO wrote
   `constexpr cudaMemcpyKind copy_kind_v = ...` (no `inline`). In C++17,
   a `constexpr` variable template specialization has external linkage
   and emits a definition in every TU that names it — the first build
   failed with multiple-definition linker errors across `Profiler.cu` +
   `test.cu` + `main.cpp`. Adding `inline` (C++17 inline variables) fixes
   the ODR issue. Mechanical correctness fix, not a design change.

2. **`static_assert` inside `detail::copy`** (1.6.5). The TODO's literal
   code (poison primary template + two specializations) makes
   wrong-direction instantiations *compile* and only fail at link time
   (or worse, silently pass the poison to `cudaMemcpy` at runtime). To
   deliver the *compile-time* error that 1.6.11 expects, a `static_assert`
   rejecting invalid `(Dst, Src)` pairs was added inside `detail::copy`.
   The poison value is kept as a defensive secondary net. Documented in
   the 1.6.5 step body.

### Validation results

- **Default build** (`cmake -B build && cmake --build build -j`):
  **clean** on Blackwell/sm_120, Release, with the full strict host
  warning set (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion
  -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
  -Woverloaded-virtual -Wformat=2 -Wnull-dereference`). No warning
  regressions introduced by the refactor.
- **`./build/test_cuda`**: **875 checks, 0 failures**. Includes the new
  `test_copy_kind_compile_time` (its `static_assert`s fired at compile
  time and passed; `g_checks` incremented by 1).
- **`./build/gemm_y`** end-to-end: full 14-size bf16 sweep runs
  identically to before — 28 rows, all PASS, CSV written. Behavior
  preserved (the refactor goal).

### Open tooling/infra items (not Phase 1.6 work)

These are pre-existing project issues surfaced by 1.6.10's validation,
not regressions caused by the refactor. Tracked here so they don't get
lost; should be addressed in a separate phase (e.g. a `chore:` commit
or a small Phase 1.7 infra cleanup).

1. **`ctest --test-dir build` reports "No tests were found".**
   `CMakeLists.txt` has no `enable_testing()` / `add_test(test_cuda)` —
   `test_cuda` is a plain executable, not registered with CTest. The
   TODO's 1.6.10 command (`ctest --test-dir build`) cannot work as
   written. Workaround used for 1.6.10: run `./build/test_cuda` directly.
   **Fix:** add `enable_testing()` + `add_test(NAME test_cuda COMMAND
   test_cuda)` to `CMakeLists.txt`.

2. **`-DENABLE_WERROR=ON` fails to build.** Confirmed pre-existing by
   stashing the refactor and rebuilding on the baseline — same failure.
   Symptom: `nvcc fatal : Value '-Xcompiler=-Wall' is not defined for
   option 'Werror'`. Root cause: in `CMakeLists.txt`, the nvcc-host
   warnings are joined with `;` into a single `-Xcompiler=<a;b;c>` arg,
   and `-Werror` is appended to that same list, producing
   `-Xcompiler=-Wall;...;-Werror`. nvcc's `-Werror` parser then sees
   `-Xcompiler=-Wall` as the value for `-Werror` and rejects it. The
   `-Xcompiler` flag and `-Werror` need to be passed as separate nvcc
   args, not joined into one `;-`-separated string. **Fix:** in
   `gemm_y_apply_common_flags`, pass `-Werror` as a standalone nvcc
   native flag (alongside `-Wreorder -Winit-self`), not via the
   `-Xcompiler=...` joined arg.

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
