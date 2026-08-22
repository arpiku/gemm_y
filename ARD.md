# gemm_y architecture decisions

This document records the current design, not the history of superseded
experiments. The project is a learning harness: correctness and reproducible
measurements come before optimization claims.

## 1. Scope and execution model

The benchmark evaluates square, column-major GEMM:

```text
C(M×N) = A(M×K) × B(K×N)
```

The current sweep uses `N = {32, 64, 96, 128, 192, 256, 384, 512, 768,
1024, 1536, 2048, 3072, 4096}` except the `sm_100` smoke/benchmark path, which
uses its aligned subset. One architecture is selected at configure time. CMake
compiles only that architecture's source directory.

Supported storage types are BF16, FP16, and TF32 (`float` storage), all with
FP32 accumulation. BF16 is the active optimization path. Transpose, batched,
grouped, multi-GPU, fused epilogues, FP8, INT8, and pedantic FP32 GEMM are out
of scope.

## 2. Ownership, layout, and API boundaries

`Buffer<T, Space>` owns host or device storage through RAII. `Matrix<T, Space>`
adds shape and leading dimension; allocation is contiguous column-major with
`ld == rows`. `MatrixView` is a non-owning POD descriptor and permits only
mutable-to-const conversion.

The harness adapter receives `GemmArgs<T>`:

```cpp
MatrixView<const T, Space::Device> A;
MatrixView<const T, Space::Device> B;
MatrixView<T, Space::Device> C;
```

A device kernel must not depend on harness types. Its raw-pointer ABI is:

```text
(const T* A, const T* B, T* C,
 int M, int N, int K, int ldA, int ldB, int ldC)
```

Kernels must write every logical output element, use the supplied leading
dimensions, and guard partial tiles.

## 3. Kernel organization and dispatch

Every kernel is a stateless functor with compile-time identity:

```cpp
static constexpr std::string_view name();
static constexpr std::string_view description();
void operator()(GemmArgs<T>, cudaStream_t) const;
```

The functor launch is defined in an architecture-specific `.cu` file and
declared in that directory's shared `.cuh`. New `.cu` files are discovered by
CMake. Architecture-specific implementation code does not use architecture
preprocessor branches.

`src/main.cpp` contains the explicit production registry. A dispatcher may
select compile-time specializations by shape, but production dispatch must not
allocate, perform I/O, synchronize for measurement, benchmark candidates, query
the database, or use runtime string lookup. Tuning registries belong in
temporary profiling targets, never in the normal benchmark.

## 4. Reference and correctness

cuBLAS v2 `cublasGemmEx` is the implicit reference. It uses column-major
`CUBLAS_OP_N`, host pointer mode, `alpha=1`, `beta=0`, and FP32 accumulation.
BF16/FP16 use default math mode; TF32 uses TF32 tensor-core math mode.

For each size, the profiler:

1. allocates A, B, C, and C-ref on host/device;
2. fills deterministic inputs and copies A/B to the device;
3. runs and measures cuBLAS once, retaining C-ref;
4. warms up and measures each custom kernel;
5. copies C back and compares against C-ref in FP64 on the host;
6. writes only accuracy-passing custom rows.

Relative-error gates are BF16 `1e-2`, FP16 `1e-3`, and TF32 `1e-3`. `kernel_smoke`
is the focused gate: it exercises small sizes, padded leading dimensions, and
architecture-specific registrations before the broader test suite or benchmark.

## 5. Measurement and results

All work uses one profiler-owned CUDA stream. CUDA events measure only the
reference/kernel region; host `steady_clock` is used only for sweep wall time.
The default policy is 20 warmups and 50 timed samples. Results report minimum,
median, sample standard deviation, p95, and an approximate 95% CI for the
median, plus raw samples. cuBLAS statistics are measured once per size and
copied into each custom row's `ref_*` fields.

CSV and `.meta` output are written by the benchmark; `scripts/ingest.py` is an
explicit, manual import into the SQLite dashboard. Reproducible performance
claims require `scripts/bench.sh` with GPU clocks locked where supported.
Direct executable runs are correctness/quick-check runs, not stable performance
claims.

## 6. Build and validation decisions

Requirements are CUDA Toolkit >= 12.8, CMake >= 3.21, and C++/CUDA 17.
Typical validation is:

```sh
cmake -S . -B build -DGEMM_Y_CUDA_ARCH=sm_90
cmake --build build --target kernel_smoke -j
./build/kernel_smoke
ctest --test-dir build --output-on-failure
```

The default configure target is `sm_120`; use a separate build directory for a
different architecture. Remote validation is manual through `rtest`: pack the
minimal source archive, upload, replace the remote source tree, run smoke, then
fetch logs/results before benchmarking.

## 7. Hopper readiness and optimization plan

The harness is ready for Hopper kernel work: `sm_90` has a separate source
path, BF16 declarations and a baseline implementation, cuBLAS comparison,
padded-LD smoke coverage, explicit benchmark registration, and event timing.
The current Hopper kernel is intentionally a scalar workflow baseline, not an
optimized tensor-core implementation.

The next work should add one isolated Hopper candidate at a time and preserve
its identity and results:

1. establish a correct tiled BF16 baseline and launch configuration;
2. introduce shared-memory/register staging with explicit synchronization;
3. add Hopper tensor-core/WGMMA-oriented paths only after boundary tests pass;
4. measure against cuBLAS using locked clocks and inspect generated SASS/PTX;
5. promote a validated policy into deterministic source dispatch.

Candidate-specific assumptions (tile shape, warpgroup layout, staging,
barriers, alignment, and supported shapes) belong in the kernel description and
smoke tests.

## 8. Known risks and deferred work

- `rtest` currently stores connection values in the script and its smoke command
  destructively replaces a fixed remote path. Move settings to environment/CLI
  and require explicit confirmation before destructive replacement.
- The current profiler allocates per size and uses synchronous host copies. This
  is deliberate for simple, isolated measurements; overlap and allocator
  effects must not be confused with kernel throughput.
- The generic test binary exercises harness components, while `kernel_smoke`
  is the architecture-specific kernel gate. New kernels should be added to the
  latter as regression controls when promoted.
- FP16 and TF32 custom kernels are deferred until the BF16 Hopper path is
  understood.
