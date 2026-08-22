# AGENTS.md — gemm_y

## Goal

Develop custom CUDA GEMM kernels that match or beat cuBLAS on:

- Hopper (`sm_90`)
- Blackwell (`sm_120`, local development target)
- `bf16`, `fp16`, and `tf32` storage with fp32 accumulation

The current primary focus is BF16 optimization on Hopper (`sm_90`). The local
RTX 50-series `sm_120` target remains useful for existing CUDA-core/tensor-core
kernels, but it does not support `tcgen05.mma`.

## Scope

Supported:

- `C = A × B`
- ColMajor matrices
- Square benchmark sizes
- One architecture per build
- cuBLAS v2 as the reference

Not currently supported:

- Transposed, batched, grouped, or multi-GPU GEMM
- Epilogue fusion (`alpha`, `beta`, bias, activation)
- `fp8`, `int8`, or pedantic fp32 CUDA-core GEMM

## Toolchain and builds

- C++17 host code
- CUDA C++17 device code
- CUDA Toolkit >= 12.8
- CMake >= 3.21

```sh
cmake -B build
cmake --build build --target kernel_smoke -j
./build/kernel_smoke
cmake --build build -j
ctest --test-dir build
```

Select Hopper explicitly:

```sh
cmake -B build -DGEMM_Y_CUDA_ARCH=sm_90
cmake --build build -j
```

The default architecture is `sm_120`. Do not commit `build/`, benchmark results,
profiler reports, Python environments, remote-test artifacts, or
`compile_commands.json`. Root-level generated CMake files and directories are
also disposable and should not be included in remote source snapshots.

## Repository structure

```text
src/main.cpp                 benchmark entry point and kernel registration
src/bench/                   profiler, timing, accuracy, output helpers
src/cublas/                  cuBLAS reference implementation
src/sm90/                    Hopper kernels
src/sm120/                   Blackwell kernels
tests/test.cu                build and correctness smoke tests
tests/kernel_smoke.cu        focused smoke test for new kernels
scripts/                     benchmark, ingestion, and dashboard tools
scripts/ingest.py            CSV/.meta ingestion, including custom-only runs
scripts/db.py                SQLite schema and dashboard query layer
```

CMake compiles only the selected architecture directory. New `.cu` files in that directory are discovered automatically.

## Kernel contract

Each custom kernel is independently identifiable and should have its own `.cu` file. During development kernels use names `k0`, `k1`, etc.; finalized strategies may use descriptive names.

The shared declaration header contains the harness adapter:

```cpp
static constexpr std::string_view name();
static constexpr std::string_view description();
void operator()(GemmArgs<T>, cudaStream_t) const;
```

The device `__global__` function must be pure CUDA and take raw pointers and dimensions:

```text
(const T* A, const T* B, T* C,
 int M, int N, int K, int ldA, int ldB, int ldC)
```

It must not depend on `MatrixView` or profiler code. The functor unpacks `GemmArgs` and launches it. Layout is ColMajor:

```text
A(i,k) = A[i + k*ldA]
B(k,j) = B[k + j*ldB]
C(i,j) = C[i + j*ldC]
```

Kernels must fully overwrite `C`, support arbitrary valid leading dimensions, and guard all tile boundaries.

## Dtypes and accuracy

- `bf16` and `fp16`: tensor cores with fp32 accumulation
- `tfloat = float`: TF32 tensor-core path only, never pedantic fp32
- Relative-error tolerances: bf16 `1e-2`, fp16 `1e-3`, tf32 `1e-3`

The profiler compares custom output with cuBLAS. Kernels exceeding the dtype tolerance are reported as failures and omitted from benchmark results.

## CUDA and C++ rules

- Include CUDA headers through `src/cuda_compat.h`.
- Check every CUDA runtime and cuBLAS call.
- Use RAII for CUDA resources.
- Use CUDA events for kernel timing; use `steady_clock` only for host orchestration timing.
- Do not use `using namespace` in headers.
- Use `constexpr`, `const`, `[[nodiscard]]`, and `noexcept` where honest.
- Keep comments focused on intent, constraints, or non-obvious decisions.
- Do not add historical experiment or chunk labels to source comments.

## Benchmarking

The square sweep is:

```text
32, 64, 96, 128, 192, 256, 384, 512, 768,
1024, 1536, 2048, 3072, 4096
```

Each size runs cuBLAS and every registered custom kernel with warmups and timed CUDA-event samples. For quick validation, build and run `kernel_smoke` before the full suite; it checks only the kernels under active development at sizes `{32, 96, 128}`. Reproducible benchmark runs must use the clock-locked wrapper. It accepts an
optional executable path; the default remains the full benchmark:

```sh
sudo scripts/bench.sh
sudo scripts/bench.sh ./build/gemm_y
```

Results are written to `results/` and can be ingested into the SQLite dashboard with the scripts under `scripts/`. For experiment runs that should not duplicate explicit cuBLAS rows, use `python scripts/ingest.py <csv> --custom-only --label <label>`; custom rows retain their embedded `ref_kernel_*` reference statistics.

When validating code, the model must run the produced executable where practical, including `./build/kernel_smoke` after building it. Direct `./build/gemm_y` runs are suitable for quick checks only; reproducible performance claims require the clock-locked wrapper. Do not run commands requiring `sudo`; the user will run commands such as `sudo scripts/bench.sh`.

### Remote GPU validation

Remote validation is currently manual and container-based. First reproduce the
remote RunPod environment locally with Docker, then use the same image for
remote execution over SSH. The initial workflow is intentionally explicit:
create a minimal source/archive payload, copy it with `scp`, connect with
`ssh`, run the container and validation commands, and copy logs/results back.

Do not add RunPod lifecycle automation, credentials, SSH private keys, automatic
SQLite ingestion, or a Python orchestration harness. Validate `kernel_smoke`
before running benchmarks, and keep benchmark interpretation and database
ingestion as manual local operations.

## Experiment workflow

The current workflow uses documentation-first contracts followed by isolated bf16 experiments. Change one kernel variable at a time, keep each candidate independently benchmarkable, and validate new kernels with `kernel_smoke` before the full suite or clock-locked benchmark. A completed tuning round is promoted into source and its temporary tuning infrastructure is removed.

Full-benchmark registration and smoke/regression registration may differ. The full benchmark explicitly lists the generic baseline and active experiments; completed kernels may be removed from that list while remaining in source, historical result storage, and the smoke test as regression controls. Registration lists must be explicit and reproducible; do not use commented-out registrations or runtime lifecycle flags.

- Keep each kernel independently identifiable and benchmarkable.
- Record tile shape, warp count, staging, layouts, and other relevant parameters in the kernel description.
- Change one kernel variable at a time during later optimization work.
- Validate correctness before comparing performance; the profiler accuracy gate is the primary kernel-output check.
- Keep failed or slower experiments available for comparison; do not hide negative results.
- Do not commit changes or create branches unless explicitly requested.

The active optimization workflow is documented by the kernel contract, explicit registration lists, focused smoke validation, and manual benchmark-result review.

### Active and completed kernel lifecycle

- The full benchmark registry contains only the generic baseline and currently active kernels.
- Completed kernel implementations may remain in source and historical results. They may remain in `kernel_smoke` only when they are deliberately retained as explicit regression controls; tuning-only validation executables and candidate-matrix tests are removed after policy promotion.
- Do not delete completed kernel implementations or historical CSV/results solely because they are no longer active.
- Registration lists must be explicit and reproducible; do not use commented-out registrations or runtime lifecycle flags.
- Keep the active `k1_` family in the full benchmark until its replacement is validated and deliberately promoted; after promotion, retain only the production dispatcher and any explicitly chosen comparison control.

### Tuning and offline dispatch

- Compile-time kernel candidates are independently benchmarkable functor specializations. Candidate registration belongs in a temporary, dedicated tuning/profile executable, never in `src/main.cpp` or the normal full benchmark.
- A tuning profile records accuracy-passing candidates and CUDA-event timing rows in CSV plus metadata. It must not modify SQLite; the user may manually ingest selected results.
- Tuning is an offline, one-off workflow for a sound kernel family or major optimization step. A selector may read one architecture/dtype CSV and emit a temporary deterministic policy, but the selector and profile executable are deleted after the validated policy is promoted.
- The permanent production dispatcher must be committed in source. It contains only compile-time candidate calls and a deterministic fallback. It must not benchmark, synchronize for measurement, allocate, perform file I/O, query SQLite, or use runtime string lookup.
- Candidate names must be unique per concrete specialization within each tuning round. The canonical `k1_t<64,64,16>` name is `k1_t`; do not create a second benchmark identity for that specialization. Recreating a purpose-built selector for a later kernel family is preferred over maintaining a premature generic selector.
- Validate the temporary policy separately from candidate correctness by checking selected names, fallback behavior, output accuracy, and dispatcher timing before source promotion. After promotion, replace tuning-only validation with production smoke and regression coverage.
- New or materially modified CUDA kernels should check launch errors consistently. Existing kernels do not require an unrelated cleanup pass.
- Autotuning must remain architecture- and dtype-specific. The policy key is `(arch, dtype, N)` for the current square-GEMM scope; a future extension may key by `(arch, dtype, M, N, K)` when nonsquare shapes are supported. Do not add runtime shape generalization prematurely.
