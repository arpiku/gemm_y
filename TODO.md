# TODO.md

> Per-branch task list. Edit at the start of each branch with planned steps;
> check off as you go. Do not carry over completed items from previous
> branches. Durable project state lives in `AGENTS.md`.
>
> Architecture rationale: see `ARD.md`.

## Current branch: `phase1/infra`

**Goal:** Stand up the foundation (CUDA wrappers, matrix DS, timing, cuBLAS
reference, profiler) and validate the harness end-to-end with one naive bf16
kernel. No perf targets this phase — correctness + harness validation only.

**Exit criteria:**
- `gemm_y` runs the full size sweep, produces CSV with cuBLAS + naive kernel
  rows, all accuracy checks pass (max_rel_err ≤ 1e-2).
- `test_cuda` covers Buffer / MatrixView / CudaTimer / CublasHandle unit tests.
- Memcpy microbench results documented in `ARD.md` and reflected in `Copy.h`.

---

## Chunk 0 — Foundation

- [x] **0.1** `src/cuda_compat.h`
  - Single include wrapper around `cuda_runtime.h`, `cuda_bf16.h`, `cuda_fp16.h`.
  - Push/pop `GCC diagnostic` for `-Wold-style-cast`, `-Wconversion`, `-Wpedantic`.
  - Re-export `__nv_bfloat16`, `__half`, `float` as dtype aliases under `namespace gemm_y::dtypes`.
  - All other headers include CUDA via this wrapper only.

- [x] **0.2** `src/CudaCheck.h`
  - `CUDA_CHECK(expr)`, `CUDA_CHECK_LAST_ERROR()`, `CUBLAS_CHECK(expr)`.

- [x] **0.3** CMake: link cuBLAS
  - `target_link_libraries(gemm_y PRIVATE CUDA::cublas)`.
  - `find_package(CUDAToolkit REQUIRED)` made explicit.
  - `#include <cublas_v2.h>` compiles in `test_cuda`.

- [x] **0.4** `src/Tracer.h` — top-of-file comment added.

---

## Chunk 1 — Matrix data structures

- [x] **1.1** `src/Space.h` — `enum class Space : std::uint8_t { Host, Device };`
- [x] **1.2** `src/Layout.h` — `enum class Layout : std::uint8_t { ColMajor, RowMajor };`
- [x] **1.3** `src/Buffer.h` — RAII `Buffer<T,S>`, move-only, Host (aligned
  std::vector) + Device (cudaMalloc/cudaFree) specializations.
- [x] **1.4** `src/MatrixView.h` — POD `{ptr, rows, cols, ld, layout}`,
  `block()`, `is_contiguous()`, `operator()`, const-converting constructor.
- [x] **1.5** `src/Matrix.h` — owns `Buffer<T,S>`, `view()`, `alloc()` factory.
- [x] **1.6** `src/Copy.h` — `copy_h2d` / `copy_d2h` with contiguous vs
  strided dispatch (cudaMemcpy vs cudaMemcpy2D). Async path not wired
  (Phase 2). No separate `.cu` needed — all paths are header-only templates.
- [x] **1.7** Unit tests in `tests/test.cu` — Buffer round-trip (Host +
  Device), MatrixView `block()` offset/ld, `is_contiguous()`, copy
  round-trip (full + submatrix).

---

## Chunk 2 — CUDA timing + memcpy microbench

- [x] **2.1** `src/CudaTimer.h` — RAII `cudaEvent_t` pair, move-only,
  `start`/`stop`/`elapsed_ms`.
- [x] **2.2** `src/bench/memcpy_microbench.cu` — sweeps H2D/D2H variants
  (sync, async, strided 2D) over N in {32..4096}, warmup=20, timed=50,
  reports min + median. Output to `results/microbench_memcpy.csv`.
- [x] **2.3** `src/Copy.h` implementation locked per microbench: contiguous
  → `cudaMemcpyAsync` (synced on default stream); strided →
  `cudaMemcpy2DAsync` (synced). Async wired through `cudaMemcpyAsync` /
  `cudaMemcpy2DAsync` + `cudaStreamSynchronize` for forward-compat with
  Phase 2 explicit streams (no behavior change vs sync on default stream).
- [x] **2.4** `src/bench/launch_overhead_microbench.cu` — 1000 empty
  launches, min/median/max. Result: min=1184 ns, median=2112 ns (recorded
  in `ARD.md` §2).

---

## Chunk 3 — cuBLAS reference

- [x] **3.1** `src/cublas/CublasHandle.h` — RAII wrapper, move-only, math
  mode + pointer mode set in ctor, per-call stream binding in `cublas_gemm`.
- [x] **3.2** `src/cublas/cublas_gemm.h` — `cublas_gemm(handle, A, B, C,
  stream)` wrapping `cublasGemmEx` with `computeType=CUDA_R_32F`,
  `transa=transb=CUBLAS_OP_N`, implicit `α=1, β=0`. Specializations for
  bf16/fp16/fp32 via `CublasTypeMap`.
- [x] **3.3** Unit test: `cublas_gemm` on N=64 bf16 vs host fp32 reference,
  tolerance 1e-3. Result: max_abs=0, max_rel=0 (deterministic pattern).

---

## Chunk 4 — Profiler + BenchRunner

- [x] **4.1** `src/bench/GemmArgs.h` — POD `{A, B, C}` views, by value.
- [x] **4.2** `src/bench/KernelTraits.h` — SFINAE trait checking
  `name()`, `description()`, `operator()(GemmArgs<T>, cudaStream_t) const`.
- [x] **4.3** `src/bench/Profiler.h` (+ `Profiler.cu`) — type-erased
  registry, `register_kernel<K>()`, `run_sweep(sizes, csv_path)`. Pre-alloc
  4096×4096 A/B/C_max/C_ref_max (4 separate device allocs) + host buffers.
  Per N: cuBLAS reference → C_ref; per kernel: warmup 20 / timed 50,
  D2H, accuracy vs C_ref, Debug-build OOB snapshot check, CSV row.
- [x] **4.4** `src/bench/Accuracy.h` — `ErrReport<T>`, `compare()` in fp64,
  `kRelErrTol = 1e-2`.
- [x] **4.5** `src/bench/CsvWriter.h` — minimal, variadic `append_row`,
  flushes on dtor.
- [x] **4.6** CLI in `src/main.cpp` — hardcoded bf16 sweep, output
  `results/bench_<arch>_bf16.csv`.

---

## Chunk 5 — Harness validation (naive bf16 kernel)

- [x] **5.1** `src/sm120/gemm_bf16_naive.cu` (+ `.cuh`) — triple-loop
  kernel, 1 thread/element, fp32 accum, ld-aware. `NaiveGemm<bf16>`
  satisfying KernelTraits.
- [x] **5.2** `src/sm90/gemm_bf16_naive.cu` (+ `.cuh`) — identical kernel,
  separate file.
- [x] **5.3** `NaiveGemm<bf16>` wired into `main.cpp` via the arch-specific
  `.cuh` header.
- [x] **5.4** Run full sweep on RTX 5070 (sm_120). CSV well-formed (29 rows),
  all `max_rel_err = 0 ≤ 1e-2`, `kernel_ns` monotonic in N, cuBLAS row
  present and reasonable. Results in `ARD.md` §10.
- [ ] **5.5** Cross-arch sanity: rebuild with `-DGEMM_Y_CUDA_ARCH=sm_90`,
  run on H100 (not available locally — deferred to server session).
- [x] **5.6** Phase 1 results summary committed to `ARD.md` §10.

---

## Out of scope for Phase 1 (deferred to Phase 2+)

- Tiled / wmma / mma kernels (Phase 2 = `bf16_tiling_128` branch).
- `cublasLtMatmul` reference (Phase 3+).
- fp16, fp32, tf32 dtype paths (after bf16 parity).
- Async stream pipelining (H2D/kernel/D2H overlap).
- `scripts/plot.py` (can be added once CSV exists; not blocking).
- nsys / ncu profiling integration.
- Non-square, transposed, batched, epilogue-fused variants (AGENTS.md non-goals).
