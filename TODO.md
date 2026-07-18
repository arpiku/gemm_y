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

- [ ] **0.1** `src/cuda_compat.h`
  - Single include wrapper around `cuda_runtime.h`, `cuda_bf16.h`, `cuda_fp16.h`.
  - Push/pop `GCC diagnostic` for `-Wold-style-cast`, `-Wconversion`, `-Wpedantic`.
  - Re-export `__nv_bfloat16`, `__half`, `float` as dtype aliases under `namespace gemm_y::dtypes`.
  - All other headers include CUDA via this wrapper only.

- [ ] **0.2** `src/CudaCheck.h`
  - `CUDA_CHECK(expr)`: evaluate `expr`, check `cudaSuccess`, print
    `__FILE__`, `__LINE__`, `cudaGetErrorString`, `abort()`.
  - `CUDA_CHECK_LAST_ERROR()`: wraps `cudaPeekAtLastError()` for post-launch.
  - `CUBLAS_CHECK(expr)`: same shape, uses `cublasGetStatusString`.
  - No exceptions across boundary — abort on failure (AGENTS.md).

- [ ] **0.3** CMake: link cuBLAS
  - `target_link_libraries(gemm_y PRIVATE CUDA::cublas)`.
  - `find_package(CUDAToolkit REQUIRED)` (already implicit via CUDA lang, make explicit).
  - Verify `#include <cublas_v2.h>` compiles in `test_cuda`.

- [ ] **0.4** `src/Tracer.h` — add top-of-file comment
  - One-liner: "Host-only (`steady_clock`). Do NOT use for kernel timing;
    use `CudaTimer` for device-side timing including H2D/D2H."

---

## Chunk 1 — Matrix data structures

- [ ] **1.1** `src/Space.h`
  - `enum class Space : std::uint8_t { Host, Device };`
  - Compile-time tag — drives `copy_h2d` / `copy_d2h` overload resolution.

- [ ] **1.2** `src/Layout.h`
  - `enum class Layout : std::uint8_t { ColMajor, RowMajor };`
  - Default = `ColMajor` (cuBLAS native). Extensible — no RowMajor code path
    needed in Phase 1, but the tag must propagate through views.

- [ ] **1.3** `src/Buffer.h`
  - `template <typename T, Space S> class Buffer`
  - RAII: `cudaMalloc` (Device) / aligned `std::vector` (Host).
  - Move-only (delete copy ctor/assign). `[[nodiscard]]` on factory.
  - `T* data()`, `std::size_t size()`, `std::size_t bytes()`.
  - No realloc / resize — fixed at construction.

- [ ] **1.4** `src/MatrixView.h`
  - `template <typename T, Space S> struct MatrixView`
  - POD: `{ T* ptr; int rows; int cols; int ld; Layout layout; }`.
  - Pass by value (fits registers, no aliasing).
  - `MatrixView<T,S> block(int r, int c, int m, int n) const noexcept`
    → sub-view, `ld` unchanged, `ptr` offset. Zero-copy slicing.
  - `bool is_contiguous() const noexcept` → `ld == rows` (ColMajor) or
    `ld == cols` (RowMajor). Drives `copy_*` dispatch.
  - `T& operator()(int i, int j)` — element access (host-side test only).

- [ ] **1.5** `src/Matrix.h`
  - `template <typename T, Space S> class Matrix`
  - Owns `Buffer<T,S>`, exposes `MatrixView<T,S> view() const`.
  - `int rows()`, `int cols()`, `int ld()` (== rows for ColMajor).
  - Factory: `Matrix<T,S>::alloc(rows, cols, Layout = ColMajor)`.

- [ ] **1.6** `src/Copy.h` (+ `src/Copy.cu` for device impl)
  - `copy_h2d(MatrixView<T,Space::Device> dst, MatrixView<const T,Space::Host> src, cudaStream_t = nullptr)`
  - `copy_d2h(MatrixView<T,Space::Host> dst, MatrixView<const T,Space::Device> src, cudaStream_t = nullptr)`
  - Dispatch: contiguous → `cudaMemcpy`, strided → `cudaMemcpy2D`.
  - Assert `dst.rows/cols == src.rows/cols`. Layout must match (no transpose).
  - **Implementation deferred until Chunk 2 microbench locks the variant choice.**

- [ ] **1.7** Unit tests in `tests/test.cu`
  - Buffer alloc/free round-trip (Host + Device).
  - MatrixView `block()` produces correct `ptr` offset and `ld`.
  - `is_contiguous()` correct for full matrix vs submatrix.
  - `copy_h2d` / `copy_d2h` round-trip preserves data (full + submatrix).

---

## Chunk 2 — CUDA timing + memcpy microbench

- [ ] **2.1** `src/CudaTimer.h`
  - RAII `cudaEvent_t` pair (start/stop). Move-only, no copy.
  - `start(cudaStream_t = nullptr)`, `stop(cudaStream_t = nullptr)`.
  - `float elapsed_ms()` — `cudaEventSynchronize(stop_)` + `cudaEventElapsedTime`.
  - Document: not thread-safe; one timer per stream.

- [ ] **2.2** `src/bench/memcpy_microbench.cu`
  - Sweep: `N ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}`.
  - Variants per direction (H2D, D2H):
    - `cudaMemcpy` (sync, contiguous)
    - `cudaMemcpyAsync` (default stream + sync)
    - `cudaMemcpy2D` (strided: 4096×4096 buffer, N×N submatrix, ld=4096)
  - Warmup=20, timed=50, report min + median.
    Rationale: 20 warmup covers GPU boost-clock + thermal steady state for
    both small and large kernels; 50 timed gives stable min (P(min ≤ true × 1.01)
    ≈ 0.995) and tight median CI (SE ≈ 0.18σ). Higher counts give diminishing
    returns at proportionally longer sweep time (see ARD.md §11).
  - Output: CSV to `results/microbench_memcpy.csv` (gitignored).
  - **Hypothesis**: sync vs async identical (no overlap); `cudaMemcpy2D` is
    the only correct option for strided submatrix copies.
  - **Decision recorded in `ARD.md` §Memcpy variant selection.**

- [ ] **2.3** Lock `src/Copy.cu` implementation per microbench results.
  - Expected: contiguous → `cudaMemcpy`; strided → `cudaMemcpy2D`.
  - Async path not wired (no stream pipelining in Phase 1).

- [ ] **2.4** `src/bench/launch_overhead_microbench.cu`
  - Empty kernel launch × 1000, measure per-launch overhead via `CudaTimer`.
  - Sets the floor for kernel time interpretation. Document in `ARD.md`.

---

## Chunk 3 — cuBLAS reference

- [ ] **3.1** `src/cublas/CublasHandle.h`
  - RAII wrapper: `cublasCreate` / `cublasDestroy`. Move-only.
  - Singleton-per-process is wrong for future multi-stream; owned by `Profiler`.
  - `cublasHandle_t get() const noexcept`.
  - `CUBLAS_CHECK` applied to create/destroy.
  - Setup in ctor (one-time, not per-call):
    - `cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)` — no TF32 for bf16 path.
    - `cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST)` — alpha/beta on host.
  - Stream binding is **per-call** (in `cublas_gemm`), not on the handle —
    lets future Phase 2 async pipelining bind different streams without
    handle recreation. Phase 1 uses `stream = nullptr` (legacy default).
  - Not thread-safe — one handle per `Profiler`, single-threaded bench.
    Documented in header.

- [ ] **3.2** `src/cublas/cublas_gemm.h` (+ `.cu`)
  - `template <typename T> void cublas_gemm(CublasHandle&, MatrixView<T,Device> A, MatrixView<T,Device> B, MatrixView<T,Device> C, cudaStream_t = nullptr)`
  - Wraps `cublasGemmEx`:
    - `computeType = CUDA_R_32F` (bf16/fp16 accumulate in fp32 — cuBLAS default).
    - `transa = transb = CUBLAS_OP_N` (no transpose; AGENTS.md non-goal).
    - `lda/b/c` from views' `ld`.
    - `cublasSetStream(handle, stream)` called per-invocation before `cublasGemmEx`.
    - `alpha = 1.0f`, `beta = 0.0f` as host scalars (pointer mode host),
      passed by address (`&alpha`, `&beta`).
  - Specializations: `__nv_bfloat16`, `__half`, `float` (fp32 → CUDA_R_32F compute).
  - **No alpha/beta params in our API** — implicit `α=1, β=0` (AGENTS.md non-goal: epilogue).

- [ ] **3.3** Unit test: `cublas_gemm` on small (N=64) bf16 matrix vs host fp32 reference.
  - Host reference: naive triple-loop in fp32, cast inputs to bf16, accumulate fp32.
  - Tolerance: max_rel_err ≤ 1e-3 (cuBLAS is the ground truth here).

---

## Chunk 4 — Profiler + BenchRunner

- [ ] **4.1** `src/bench/GemmArgs.h`
  - `template <typename T> struct GemmArgs { MatrixView<T,Device> A, B, C; };`
  - POD, by value. Future extension point for `alpha`/`beta` (commented out).

- [ ] **4.2** `src/bench/KernelTraits.h`
  - Concept (SFINAE in C++17): a kernel `K<T>` must provide:
    - `static constexpr std::string_view name()`
    - `static constexpr std::string_view description()`
    - `void operator()(GemmArgs<T>, cudaStream_t) const`
  - `description()` carries design info: "naive triple-loop", "wmma 16×16×16",
    "tiled 128×128 8-warps", etc. — flows to CSV → plotter labels.

- [ ] **4.3** `src/bench/Profiler.h` (+ `.cu`)
  - `template <typename T> class Profiler`
  - `register_kernel<K>()` where `K` satisfies KernelTraits — stores
    `{name, description, std::function<void(GemmArgs<T>,cudaStream_t)>}`.
    Stateless `K` → SBO, no heap.
  - `run_sweep(sizes, out_csv_path)`:
    - Pre-alloc 4096×4096 `Matrix<T,Device>` A, B, C_max, **C_ref_max** (×4).
      Four **separate** `cudaMalloc` allocations — CUDA guarantees disjoint
      address ranges; no aliasing risk (see ARD.md §5).
      Memory: 4 × 4096² × 2 B (bf16) = 128 MB device. Trivial on both targets.
    - Pre-alloc 4096×4096 `Matrix<T,Host>` for H2D source + D2H sink.
    - Fill host A, B with deterministic pattern (e.g. `i+j` / `i-j`).
    - `copy_h2d` A, B once (timed, reported in CSV).
    - Per `N`:
      1. Build sub-views `dA = A_max.block(0,0,N,N)`, etc. (ld=4096).
      2. Run cuBLAS reference → `C_ref_max.block(0,0,N,N)`.
         `C_ref_max` is **read-only** after this point per N.
      3. `copy_d2h` `C_ref` once (timed).
      4. Per registered kernel:
         - Warmup 20 launches (untimed).
         - Timed 50 launches → `CudaTimer` records `kernel_ns` (min + median).
         - `copy_d2h` `C` once (timed).
         - Accuracy: `max_abs_err`, `max_rel_err` vs `C_ref` (host-side, fp64).
         - **Debug-build OOB check**: after custom kernel runs, re-zero
           `C_ref_max`'s N×N block is NOT done — instead, in Debug builds,
           snapshot `C_ref_max`'s N×N block before the custom kernel and
           verify it's unchanged after. Catches out-of-bounds writes that
           would otherwise silently corrupt the reference. Cheap, debug-only.
         - Append CSV row.
  - CSV schema:
    `arch,dtype,N,kernel_name,kernel_desc,h2d_ns,kernel_ns,d2h_ns,ref_kernel_ns,max_abs_err,max_rel_err`

- [ ] **4.4** `src/bench/Accuracy.h`
  - `template <typename T> struct ErrReport { double max_abs; double max_rel; };`
  - `ErrReport compare(MatrixView<const T,Host> got, MatrixView<const T,Host> ref)`
  - Promotes to fp64 for comparison. Returns both metrics.
  - Tolerance threshold constant: `constexpr double kRelErrTol = 1e-2;`
    (conservative for Phase 1; tighten to 1e-3 in Phase 2 once cuBLAS
    non-determinism across reduction orders is characterized).

- [ ] **4.5** `src/bench/CsvWriter.h`
  - Minimal: open file, write header, `append_row(...)` with variadic args.
  - No external deps. Flushes on destruction.

- [ ] **4.6** CLI in `src/main.cpp`
  - Hardcode sweep for Phase 1 (no argparse dependency).
  - `--dtype bf16` (only bf16 wired in Phase 1).
  - Output path: `results/bench_<arch>_<dtype>.csv` (gitignored).

---

## Chunk 5 — Harness validation (naive bf16 kernel)

- [ ] **5.1** `src/sm120/gemm_bf16_naive.cu`
  - Triple-loop kernel: 1 thread per `C[i][j]`, inner `k` loop.
  - `ld`-aware: reads `A[i][k]` via `A.ptr + i + k*A.ld` (ColMajor).
  - Accumulate in fp32, cast back to bf16 on write.
  - `struct NaiveGemm<__nv_bfloat16>` satisfying KernelTraits.
  - `description() = "naive triple-loop, 1 thread/element, fp32 accum"`.
  - **Perf-irrelevant** — exists only to validate the harness end-to-end.

- [ ] **5.2** `src/sm90/gemm_bf16_naive.cu`
  - Identical kernel, separate file (AGENTS.md: no `#ifdef` arch branches).
  - CMake picks the right one via `GEMM_Y_CUDA_ARCH`.

- [ ] **5.3** Wire `NaiveGemm<__nv_bfloat16>` into `main.cpp`:
  ```cpp
  Profiler<__nv_bfloat16> prof;
  prof.register_kernel<NaiveGemm<__nv_bfloat16>>();
  prof.run_sweep(kSweepSizes, "results/bench_sm120_bf16.csv");
  ```

- [ ] **5.4** Run full sweep on RTX 5070 (sm_120).
  - Verify CSV well-formed.
  - Verify all `max_rel_err` rows ≤ 1e-2.
  - Verify `kernel_ns` is monotonic-ish in `N` (sanity, not a perf target).
  - Verify cuBLAS row present and reasonable.

- [ ] **5.5** Cross-arch sanity: rebuild with `-DGEMM_Y_CUDA_ARCH=sm_90`,
  run on H100 (if available), verify CSV produces.

- [ ] **5.6** Commit results summary to `ARD.md` §Phase 1 validation.

---

## Out of scope for Phase 1 (deferred to Phase 2+)

- Tiled / wmma / mma kernels (Phase 2 = `bf16_tiling_128` branch).
- `cublasLtMatmul` reference (Phase 3+).
- fp16, fp32, tf32 dtype paths (after bf16 parity).
- Async stream pipelining (H2D/kernel/D2H overlap).
- `scripts/plot.py` (can be added once CSV exists; not blocking).
- nsys / ncu profiling integration.
- Non-square, transposed, batched, epilogue-fused variants (AGENTS.md non-goals).
