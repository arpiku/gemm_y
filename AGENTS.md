# AGENTS.md — gemm_y

Project spec and contributor guidelines. Read this first.

## Project Goal

Write custom CUDA kernels that **match or beat cuBLAS** for GEMM (`C = A × B`)
on Hopper (sm_90) and Blackwell (sm_120), for `bf16`, `fp16`, and `tf32`.

- **First milestone:** `bf16` GEMM at parity with cuBLAS.
- **Subsequent milestones:** `fp16`, then `tf32`.
- **Target hardware:** RTX 5070 (Blackwell, sm_120) for local dev;
  Hopper server (sm_90) for cross-arch validation.
- **Success metric:** kernel time ≤ cuBLAS time across the full size sweep
  (see Benchmarking), within the accuracy tolerance (see ARD §6).

## Non-Goals (for now)

- Non-square matrices.
- Batched GEMM, strided batched GEMM, grouped GEMM.
- Transposed variants (`A^T`, `B^T`) — assume `C = A × B` only until baseline is hit.
- Epilogue fusion (`α`, `β`, bias, activation) — plain `C = A × B` only.
- Lower-precision (`fp8`, `int8`).
- Multi-GPU or multi-node.
- **fp32 pedantic (CUDA cores)** — dropped entirely. Only the tf32 path
  (tensor cores) is implemented for 32-bit float storage. See ARD §9.

## Tech Stack

- **C++17** on the host (modern, idiomatic; no C++20 features).
- **CUDA 17** on the device (`cuda_std_17`).
- **CMake ≥ 3.21**, modern target-based usage.
- **CUDA toolkit ≥ 12.8** (first release supporting both sm_90 and sm_120).
- **One binary per arch** — selected at configure time via `GEMM_Y_CUDA_ARCH`.
  No fat binaries, no runtime dispatch.
- **Hardware** : RTX 5070 (default, BlackWell, sm_120 ,local machine), H100 (other, Hopper, sm_90 ,server)

## Tech Stack (Continued)  
- **cuBLAS** - v2 (CUDA 12.8) to be used
- **Reference comparison**: compare against `cublasGemmEx` fist, then the goal post will shift to beating `cublasLtMatmul` (the newer, lower-level API) 
- **Dtypes**: `bf16` and `fp16` target Tensor Cores by default (fp32 accum). `tfloat = float` is the tf32 path (TC, `CUBLAS_TF32_CUBLAS_MATH`); no pedantic fp32 / CUDA-core path. See ARD §9.
- **Python tooling**: Plotly + Dash + SQLite for the benchmark dashboard. venv at `pyenv/` (Python 3.14). See "Python tooling" under Build Commands.

## Build Commands

```sh
# Blackwell (default, RTX 5070), Release, -O3
cmake -B build
cmake --build build -j

# Hopper, Release
cmake -B build -DGEMM_Y_CUDA_ARCH=sm_90
cmake --build build -j

# Debug build (-O0 -g)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Run tests (ctest; test_cuda is registered as a single CTest entry)
ctest --test-dir build

# Note: -Werror is intentionally not supported (nvcc's -Werror requires a
# <kind> argument and greedily consumes the next flag as its value, breaking
# the build). Rely on the strict warning set compiled in by default.
```

### Python tooling

The benchmark dashboard (Phase 2B) lives under `scripts/`. It uses a
project-local venv at `pyenv/` (Python 3.14). Activate before running
any script:

```sh
source pyenv/bin/activate
pip install -r scripts/requirements.txt   # first time only
python scripts/ingest.py results/bench_sm_120_bf16.csv [--label "..."]
python scripts/server.py                   # dashboard at localhost:8050
python scripts/dump_db.py                  # optional JSONL export
```

Workflow: `./build/gemm_y` writes CSV + `.meta` sidecar to `results/`
(gitignored). `ingest.py` appends to `db/gemm_y.db` (tracked in git,
declared binary in `.gitattributes`). `server.py` reads from the DB,
never from CSV. Git sha is captured by `ingest.py` at ingest time.

Targets:
- `gemm_y` — main executable (`src/main.cpp` + arch-specific `.cu`).
- `test_cuda` — build-verification smoke test (`tests/test.cu`).

## Repository Layout

```
gemm_y/
├── AGENTS.md              # this file — durable project spec
├── ARD.md                 # architecture decision record (rationale)
├── TODO.md                # forward-looking task list (no completed items)
├── CMakeLists.txt         # build system
├── .gitattributes         # declares db/gemm_y.db as binary
├── pyenv/                 # Python venv for scripts/ (gitignored via pyenv/.gitignore)
├── db/
│   └── gemm_y.db          # SQLite benchmark DB (tracked, binary)
├── src/
│   ├── main.cpp           # entry point
│   ├── Tracer.h           # host-side timer (steady_clock, C++17)
│   ├── cuda_compat.h      # single CUDA include wrapper
│   ├── CudaCheck.h        # CUDA_CHECK / CUBLAS_CHECK / GEMM_Y_ASSERT
│   ├── CudaTimer.h        # RAII cudaEvent pair (device timing)
│   ├── Space.h, Layout.h  # compile-time memory-space / layout tags
│   ├── Buffer.h, Matrix.h, MatrixView.h, Copy.h
│   ├── Arch.h, dtypes.h   # arch name + dtype aliases/names (bf16/fp16/tfloat)
│   ├── bench/             # Profiler, GemmArgs, KernelTraits, Accuracy,
│   │   ├── Stats.h, Fill.h, CsvWriter.h
│   │   └── microbench/    # memcpy + launch-overhead microbenches
│   ├── cublas/            # CublasHandle, CublasMathModeGuard, cublas_gemm
│   ├── sm90/              # Hopper-specific kernels
│   └── sm120/             # Blackwell-specific kernels
├── tests/
│   └── test.cu            # unit tests + build-verification
└── scripts/               # ingest.py, server.py, db.py, dump_db.py, requirements.txt
```

## LLM Context Loading

At session start, load:
- `AGENTS.md` — always (durable contract).
- `TODO.md` — always (what to do next; no completed items).
- `ARD.md` — table of contents only; load specific sections on demand.

Do not load `ARD.md` in full unless reviewing a specific decision.

## Coding Conventions

### C++ / CUDA Style
- **Modern C++17**: `constexpr`, `[[nodiscard]], `noexcept` where honest.
- **Modern C++17**: `const` qualifiers for constants, things that may be calculated at compile time, are calculated at compile time.
- **RAII** for all CUDA resources. Wrap `cudaMalloc`/`cudaFree` in a small
  device-buffer type; never leak raw `cudaFree` calls across early returns.
- **No exceptions across the CUDA boundary** — kernels can't throw, and host
  code calling CUDA APIs should check return codes and propagate via return
  value or `std::optional`/`expected`-like pattern.
- **`string_view` lifetime**: when used (e.g. in `Tracer.h`), the caller
  must outlive the consumer. Document at the call site.
- **No `using namespace` in headers.**
- **Header guards**: `#pragma once` (already the convention).
- **No phase trailers in source comments**: do not carry `Phase X.Y`,
  `RXX`, or `Chunk X.X` tags in source comments. The durable record lives
  in git history (`git log --grep="Phase: X.X"`) and ARD phase-summary
  sections. Source comments explain *why*, not *which phase*.
- **Comment discipline**: prefer one concise explanation at the type/function
  level over restating the same fact in the header, the helper, and the
  macro. Drop comments that restate the code; keep comments that explain
  intent, constraints, or non-obvious tradeoffs.

### Dtype conventions
- **`tfloat = float`** alias (in `src/dtypes.h`). Always commented at the
  alias declaration and at every use site with: `// tfloat = tf32 path
  (TC), not pedantic fp32 (CUDA cores).` The pedantic fp32 / CUDA-core
  path is dropped entirely — only the tf32 tensor-core path is implemented
  for 32-bit float storage. See ARD §9.
- **`CublasTypeMap<T>::math_mode`** selects the cuBLAS math mode per
  dtype: `CUBLAS_DEFAULT_MATH` for bf16/fp16, `CUBLAS_TF32_CUBLAS_MATH`
  for tfloat. `cublas_gemm` wraps the call in `CublasMathModeGuard` — no
  distinct `cublas_gemm_tf32` entry point.

### Accuracy / tolerance
- **`kRelErrTol<T>`** is a per-dtype compile-time constant
  (`template <typename T> constexpr double kRelErrTol<T>()`):
  bf16 → 1e-2, fp16 → 1e-3, tfloat → 1e-3. See ARD §6.
- **Failed kernels are skipped at the Profiler level**: if
  `err.max_rel > kRelErrTol<T>`, the row is not written to the CSV
  (timing of mathematically invalid kernels is meaningless). A stderr
  FAIL message is printed with N, kernel name, rel_err, tol. The cuBLAS
  reference row is always written (ground truth, err == 0).

### CUDA-Specific
- **Kernel timing**: use `cudaEvent` for device-side timing, **not** the host
  `Tracer` (which uses `steady_clock` and measures wall time including
  launch overhead). `Tracer` is for host-side orchestration only.
- **Error checking**: every CUDA runtime call must be checked. Provide a
  `CUDA_CHECK(expr)` macro that prints `cudaGetErrorString` and aborts.
- **Headers**: include `cuda_runtime.h` (and other CUDA headers, including
  `<cublas_v2.h>`) through a single wrapper `src/cuda_compat.h` that
  suppresses `-Wold-style-cast` / `-Wconversion` noise from NVIDIA's headers.
- **Arch-specific code**: prefer separate `.cu` files under `src/sm90/` and
  `src/sm120/` over `#ifdef` branches. CMake only compiles the directory
  matching `GEMM_Y_CUDA_ARCH`.
- **Warp-level primitives**: prefer `__shfl_sync`, `wmma`/`mma` over
  shared-memory reductions where the arch supports it natively.
- **Kernel ABI (`GemmArgs<T>`)**: `A`/`B` are `MatrixView<const T, Device>`
  (read-only inputs), `C` is `MatrixView<T, Device>` (mutable output). Relies
  on `MatrixView`'s implicit converting ctor (`MatrixView<T,S> ->
  `MatrixView<const T,S>`) — zero call-site churn. `cublas_gemm` is the
  exception: it takes writable views for A/B because C++ template argument
  deduction does not consider implicit conversions (see ARD §3).
- **`MatrixView` dual-use**: (1) host-side view (`block`/`operator()`/
  `is_contiguous`/converting ctor), (2) kernel-side POD descriptor (only
  `ptr`/`rows`/`cols`/`ld` read directly). Host methods are **not**
  `__device__`-callable; kernels read fields directly (see ARD §1).
- **`CublasMathModeGuard`** (free class in `src/cublas/CublasHandle.h`):
  RAII guard that captures the current math mode via
  `cublasGetMathMode`, sets a new mode, and restores the previous mode in
  the dtor. Used by `cublas_gemm` to apply `CublasTypeMap<T>::math_mode`
  per call. Non-copyable, non-movable. See ARD §9.

### Warnings
- Host CXX: full strict set (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion`
  `-Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused`
  `-Woverloaded-virtual -Wformat=2 -Wnull-dereference`).
- `nvcc` host compiler: reduced subset (see `CMakeLists.txt` for the
  rationale — CUDA's own headers trip the dropped flags).
- `nvcc` native: `-Wreorder -Winit-self`.
- `-Werror` is not supported (nvcc's `-Werror` requires a `<kind>` argument
  and greedily consumes the next flag as its value, breaking the build).
  Rely on the strict warning set compiled in by default.

## Benchmarking Protocol

### Matrix Size Sweep
- Square matrices only: `N ∈ {32, 64, 96, 128, 192, 256, 384, 512, 768,
  1024, 1536, 2048, 3072, 4096}`.
- Powers of 2 plus midpoints — captures both tiling-aligned and
  misaligned cases.
- For each `N`: benchmark both the custom kernel and cuBLAS reference.

### Output artifacts
- `results/bench_<arch>_<dtype>.csv` — one row per (N, kernel). Failed
  kernels (rel_err > tol) are absent. Schema documented in ARD §5.
- `results/bench_<arch>_<dtype>.meta` — key=value sidecar with run
  metadata (arch, dtype, warmup/timed iters, tol, sweep sizes, kernel
  list, timestamp). Parsed by `scripts/ingest.py`. No git sha (captured
  at ingest time).
- `db/gemm_y.db` — SQLite store of all ingested runs. Tracked in git,
  declared binary in `.gitattributes`. Source of truth for the dashboard.

### Visualization
- **Tool**: Python + Plotly + Dash + SQLite (see "Python tooling" under
  Build Commands). No matplotlib.
- **Dashboard** (`scripts/server.py`, `localhost:8050`): single page,
  sidebar + three tabs (Timing / Accuracy / Run History). Sidebar
  filters: arch, dtype, custom-vs-cuBLAS, runs (multi-select), scale
  (log-log / linear). Hover shows arch, dtype, custom/ref, kernel name,
  kernel desc, N, median_ns, ref_median_ns, speedup vs cuBLAS.
- **Default plot**: log-log, `kernel_median_ns` vs `N`. One line per
  (run, kernel). cuBLAS is the reference line.

## Experiment Discipline

- **Hypothesis-driven**: each experiment has a stated hypothesis
  ("tiling 128×128 with 8 warps per CTA beats cuBLAS at N≥512") and a
  pass/fail criterion before the run.
- **One variable at a time**: tile size, warp count, K-dim unroll, memory
  layout — change one per commit.
- **Record negative results**: a kernel that loses to cuBLAS still gets
  committed with a clear commit message explaining *why* it was tried and
  *what* the bottleneck was. Future agents will otherwise re-try dead ends.
- **Microbenchmarks first**: before touching GEMM, measure raw copy
  bandwidth (`cudaMemcpy` H2D/D2H), kernel launch overhead, and
  matrix-generation time on the host. These set the floor and ceiling.

## Git Workflow

- **Branch per feature/experiment**: `bf16_baseline`, `bf16_tiling_128`,
  `sm90_wmma_v2`, etc.
- **Granular commits**: one logical change per commit. A new tile size is
  one commit; a bugfix to the same kernel is a separate commit.
- **Commit message format**: Conventional Commits (`feat:`, `perf:`,
  `fix:`, `chore:`, `docs:`, `test:`, `bench:`). Body explains *why*,
  not just *what*.
- **Phase trailer**: every commit body ends with a `Phase: X.X` footer
  (e.g. `Phase: 1.5`, `Phase: 2A`). This is the durable phase-history
  marker — query with `git log --grep="Phase: 1.5"`. No merge commits
  required; fast-forward merges are fine.
- **Completed work leaves `TODO.md`**: when a phase completes, cut its
  section from `TODO.md`. The phase's commits (found via the trailer)
  and the ARD phase-summary section (e.g. §10, §14) are the durable
  record. `TODO.md` is forward-looking only.
- **Never commit** `build/`, `results/`, `*.nsys-rep`, `*.ncu-rep`,
  `compile_commands.json` — all gitignored.
- **`db/gemm_y.db` IS committed** (tracked, declared binary in
  `.gitattributes`). It is the durable record of benchmark history.

## Profiling Tools
- Profiling will be setup later (nsys & ncu)
