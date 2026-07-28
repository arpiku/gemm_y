# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

---

## Phase 2D — Bench-runner refactor: per-N allocation + dead-code trim

Goal: replace the pre-allocated 4096×4096 + `block()` slicing strategy with
per-`N` allocation so `ld == N` for every kernel launch. Trim dead surface
area (`Tracer.h`, `Copy.h`, `Layout`, `block()`, `is_contiguous()`,
strided-copy path, RowMajor branches, strided microbench). No new features;
correctness + hygiene only.

**Context:** the pre-alloc strategy (ARD §5, superseded) forced every kernel
to run on `ld=4096` regardless of `N`. A kernel author writing a 128×128
tiled TC kernel for `N=128` cannot exercise the `ld == N` fast path; the
4096 stride pollutes L2/TLB and makes benchmark numbers unrepresentative of
a real `N=128` problem. cuBLAS itself runs on the same strided layout, so
the comparison is apples-to-apples only because both sides are equally
penalized. Additionally, `h2d_ns` was a single global number stamped into
every CSV row regardless of `N` — semantically wrong.

### Step 1 — Delete `Tracer.h`, inline its one use

- [ ] **2D.1** Delete `src/Tracer.h`.
- [ ] **2D.2** `src/bench/Profiler.cu`: drop `#include "Tracer.h"`, drop the
  `tracer::Timer<> sweep_timer` + two `mark()`s + the wall-time `printf`
  block. Replace with a 3-line `std::chrono::steady_clock` start/end +
  `printf` for sweep wall time (host orchestration context only — not
  kernel timing; `CudaTimer` remains the only device timer).

### Step 2 — Refactor `Profiler::run_sweep` to allocate per-N

- [ ] **2D.3** `src/bench/Profiler.cu`: remove `constexpr int kMaxN = 4096;`
  and the `if (N > kMaxN) skip` guard.
- [ ] **2D.4** Move all 8 `Matrix<...>::alloc(...)` calls **inside** the
  `for (int N : sizes)` loop, sized `N×N`. Buffers are RAII — constructed
  and destroyed per iteration, no manual `cudaFree`.
- [ ] **2D.5** `bench::fill_sequential<T>(hA.view(), hB.view())` per `N`
  (filling an `N×N` host buffer directly — no strided source).
- [ ] **2D.6** H2D A+B per `N`, timed per `N` via `CudaTimer`, reported per
  row. The `h2d_ns` column now means "H2D cost for this `N`" — semantically
  correct (was: global 4096² value repeated per row).
- [ ] **2D.7** Drop all `.block(0,0,N,N)` calls — views are the full buffer;
  `ld == N` (ColMajor default in `Matrix::alloc`).
- [ ] **2D.8** D2H C and C_ref per `N` (already per-`N` in the inner loop;
  drop the `.block()`).
- [ ] **2D.9** Debug OOB snapshot → single contiguous `cudaMemcpy` of the
  `N×N` `C_ref` buffer + `memcmp` (replaces the element-loop snapshot).
- [ ] **2D.10** Update the `[Profiler]` startup `printf` (drop `kMaxN`
  reference; `h2d` is now per-`N`, so the startup line no longer reports a
  single H2D number — drop it from the startup print, it's per-row in the
  CSV).

### Step 3 — Delete `Copy.h`, inline `cudaMemcpy`

- [ ] **2D.11** Delete `src/Copy.h` entirely.
- [ ] **2D.12** `src/bench/Profiler.cu`: replace `copy_h2d(dst, src)` /
  `copy_d2h(dst, src)` calls with direct `CUDA_CHECK(cudaMemcpy(...))` using
  `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`. Buffers are
  contiguous (`ld == N`), so `cudaMemcpy` (not `cudaMemcpy2D`) is correct.
  Direction is explicit at the call site — no `copy_kind_v` dispatch needed.

### Step 4 — Trim `MatrixView.h`

- [ ] **2D.13** Delete `block()` (no call sites after Step 2).
- [ ] **2D.14** Delete `is_contiguous()` (no call sites after Step 3).
- [ ] **2D.15** Delete the `Layout` field and all `Layout::RowMajor`
  branches in `operator()` (dead — ColMajor-only is the stated invariant
  everywhere; `Layout.h` is being deleted in Step 6).
- [ ] **2D.16** Rewrite the header comment: remove the "pre-allocate
  4096×4096" paragraph and the `block()`/`is_contiguous()` documentation.
  The dual-use contract (host utility + kernel POD descriptor) still holds
  — just without `block`/`is_contiguous`. `operator()` is host-side
  element access (used by `Accuracy.h::compare` and tests); the converting
  ctor stays (const-correctness for `GemmArgs`).

### Step 5 — Trim `Matrix.h`

- [ ] **2D.17** `Matrix::alloc`: drop the `Layout` parameter and the
  `RowMajor` branch. `ld = rows` unconditionally (ColMajor is the only
  layout). Drop the `layout` field and the `layout()` accessor.
- [ ] **2D.18** Update the `Matrix` ctor signature (drop `Layout` arg) and
  the `view()` methods (drop `layout_` from the constructed `MatrixView`).

### Step 6 — Delete `Layout.h`

- [ ] **2D.19** Delete `src/Layout.h`. After Steps 4 and 5, no code
  references `Layout` or `Layout::ColMajor`.

### Step 7 — Update `tests/test.cu`

- [ ] **2D.20** Delete `test_matrixview_block` (exercises `block()`).
- [ ] **2D.21** Delete `test_matrixview_is_contiguous` (exercises
  `is_contiguous()`).
- [ ] **2D.22** Delete `test_copy_roundtrip_submatrix` (exercises strided
  `cudaMemcpy2D` path via `Copy.h`).
- [ ] **2D.23** Delete `test_cublas_gemm_bf16_strided` (exercises `block()`
  + strided cuBLAS).
- [ ] **2D.24** Update `test_copy_roundtrip_full`, `test_matrix_view_from_matrix`,
  `test_matrixview_const_conversion`, and any other test constructing
  `MatrixView`/`Matrix` to drop the `Layout` argument.
- [ ] **2D.25** Update `main()`'s test-call list to match the deletions.

### Step 8 — Delete strided microbench variant

- [ ] **2D.26** `src/bench/microbench/memcpy_microbench.cu`: delete
  `bench_h2d_strided_2d` and `bench_d2h_strided_2d` (the strided
  `cudaMemcpy2D` variants — no longer representative of the bench runner,
  which now uses contiguous `cudaMemcpy` exclusively). Delete the calls in
  `run_memcpy_microbench_main`. Keep the contiguous + async variants.

### Step 9 — Update `cublas_gemm.h`

- [ ] **2D.27** `src/cublas/cublas_gemm.h`: drop the `Layout::ColMajor`
  check (the `layout` field is gone). Drop the `GEMM_Y_ASSERT` on layout
  (no longer applicable — ColMajor is the only layout, enforced by
  `Matrix::alloc` setting `ld = rows`).

### Step 10 — Update `gemm_naive.cu` (both arches) + `gemm_bf16_k0.cu`

- [ ] **2D.28** `src/sm90/gemm_naive.cu`, `src/sm120/gemm_naive.cu`,
  `src/sm120/gemm_bf16_k0.cu`: drop the `GEMM_Y_ASSERT` on
  `args.A.layout == Layout::ColMajor` (the `layout` field is gone). The
  ColMajor invariant is now structural (`Matrix::alloc` sets `ld = rows`),
  not a runtime check.

### Step 11 — Update `AGENTS.md`

- [ ] **2D.29** Repository Layout: drop `Tracer.h`, `Copy.h`, `Layout.h`
  lines; update `Matrix.h`/`MatrixView.h` descriptions (no more
  "pre-allocate 4096×4096", no more `block()`/`is_contiguous()`).
- [ ] **2D.30** Coding Conventions → C++ / CUDA Style: drop the Tracer
  `string_view` lifetime bullet.
- [ ] **2D.31** Coding Conventions → CUDA-Specific: drop the Tracer
  sentence in "Kernel timing" (CudaTimer is the only timer now). Update
  the `MatrixView` dual-use bullet (drop `block`/`is_contiguous`, drop
  "pre-allocate 4096×4096 buffer"). Drop the `Copy.h` "single touch-point"
  implication if present.
- [ ] **2D.32** Coding Conventions → CUDA-Specific: drop the `Layout`
  mentions; ColMajor is the only layout, enforced structurally.

### Step 12 — Update `ARD.md`

- [ ] **2D.33** §1 (Memory model): drop `Layout` from the `MatrixView` row
  (`{ptr, rows, cols, ld}` only). Drop the `Layout` compile-time tag
  paragraph. Drop `block()`/`is_contiguous()` from the dual-use contract.
  Drop the "pre-allocate 4096×4096" consequence. Drop the `copy_h2d`/
  `copy_d2h` "single touch-point" consequence (`Copy.h` is deleted).
- [ ] **2D.34** §2 (Memcpy variant selection): supersede with "§2
  (revised): contiguous `cudaMemcpy` only". The strided `cudaMemcpy2D`
  path is deleted (no strided buffers after per-`N` alloc). `Copy.h` is
  deleted; `cudaMemcpy` is called directly from `Profiler.cu`. Strike
  through the old decision; keep the historical context.
- [ ] **2D.35** §5 (Bench runner): supersede with "§5 (revised): per-N
  allocation". Document: per-`N` alloc of A/B/C/C_ref (device + host),
  per-`N` H2D timing (reported per row), `ld == N` for every kernel
  launch, debug OOB check via contiguous `cudaMemcpy` + `memcmp`. Strike
  through the pre-alloc + `block()` decision; keep the historical context.
- [ ] **2D.36** §5.1 (C_ref storage): update the OOB-mitigation paragraph
  to reflect the contiguous `cudaMemcpy` + `memcmp` (no element loop).
- [ ] **2D.37** §7 (Timing): drop the `Tracer` bullet; `CudaTimer` is the
  only timer. Host sweep wall-time uses inline `std::chrono::steady_clock`
  (3 lines in `Profiler.cu`). Strike through the `Tracer` decision.
- [ ] **2D.38** §13 (Phase 1.5 refactor inventory): add a new row for the
  2D refactor (per-`N` alloc, `Copy.h`/`Tracer.h`/`Layout.h` deletion,
  `block()`/`is_contiguous()` removal). Or add a new §17 phase-summary
  section — see Step 13.

### Step 13 — ARD phase-summary section

- [ ] **2D.39** Add `## 17. Phase 2D — per-N allocation + dead-code trim`
  to `ARD.md`. Document: what changed, why (the `ld == N` motivation), what
  was deleted (`Tracer.h`, `Copy.h`, `Layout.h`, `block()`,
  `is_contiguous()`, strided copy path, RowMajor branches, strided
  microbench), validation results (test count, sweep sanity).

### Step 14 — Validate

- [ ] **2D.40** `cmake -B build && cmake --build build -j` — clean build,
  no warnings from the strict set.
- [ ] **2D.41** `ctest --test-dir build` — tests pass after Step 7
  deletions. Confirm the reduced test count matches the deletions.
- [ ] **2D.42** `./build/gemm_y` — sweep runs end-to-end. Verify:
  - CSV `h2d_ns` now **varies per `N`** (was a single global value).
  - Kernels see `ld == N` (add a temporary `printf` in `gemm_naive.cu`
    to confirm `args.A.ld == N`, then remove before commit).
  - Sweep wall-time `printf` still appears (inline `steady_clock`).
- [ ] **2D.43** Ingest + dashboard sanity:
  ```sh
  source pyenv/bin/activate
  python scripts/ingest.py results/bench_sm_120_bf16.csv --label "per-N-alloc"
  python scripts/server.py
  ```
  - No schema change (CSV columns unchanged). `h2d_ns` semantics shifted
    (per-`N` now, was global) — confirm the dashboard renders without
    errors. The Timing tab's `h2d_ns` is not plotted as a line (it's a
    per-row context column), so no visual regression expected.
- [ ] **2D.44** Microbench sanity:
  ```sh
  cmake --build build --target memcpy_microbench
  ./build/memcpy_microbench
  ```
  - Contiguous + async variants still run; strided variants are gone.

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
- TFLOPS metric — deferred. Needs peak-TFLOPS lookup per `(arch, dtype)`.
  Will be added in a future sub-phase after the `% perf vs cuBLAS` metric
  is in place.
- RowMajor layout — deleted in Phase 2D. If a future phase needs it,
  re-add a `Layout` tag + the `RowMajor` branches (5-line re-add per
  call site; not worth carrying as dead code).
