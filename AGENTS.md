# AGENTS.md — gemm_y

Project spec and contributor guidelines. Read this first.

## Project Goal

Write custom CUDA kernels that **match or beat cuBLAS** for GEMM (`C = A × B`)
on Hopper (sm_90) and Blackwell (sm_120), for `fp16`, `bf16`, and `fp32`.

- **First milestone:** `bf16` GEMM at parity with cuBLAS.
- **Subsequent milestones:** `fp16`, then `fp32`.
- **Target hardware:** RTX 5070 (Blackwell, sm_120) for local dev;
  Hopper server (sm_90) for cross-arch validation.
- **Success metric:** kernel time ≤ cuBLAS time across the full size sweep
  (see Benchmarking), within the accuracy tolerance (see Open Decisions).

## Non-Goals (for now)

- Non-square matrices.
- Batched GEMM, strided batched GEMM, grouped GEMM.
- Transposed variants (`A^T`, `B^T`) — assume `C = A × B` only until baseline is hit.
- Epilogue fusion (`α`, `β`, bias, activation) — plain `C = A × B` only.
- Lower-precision (`fp8`, `int8`).
- Multi-GPU or multi-node.

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
- **Tolerance**: fp32 (pedantic) will run on conventional cuda cores, tf32 (fp32) variant will target Tensor cores, bf16 and fp16 wil target Tensor Cores by default 

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

# Note: -Werror is intentionally not supported (nvcc's -Werror requires a
# <kind> argument and greedily consumes the next flag as its value, breaking
# the build). Rely on the strict warning set compiled in by default.
```

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
├── src/
│   ├── main.cpp           # entry point
│   ├── Tracer.h           # host-side timer (steady_clock, C++17)
│   ├── cuda_compat.h      # single CUDA include wrapper
│   ├── CudaCheck.h        # CUDA_CHECK / CUBLAS_CHECK / GEMM_Y_ASSERT
│   ├── CudaTimer.h        # RAII cudaEvent pair (device timing)
│   ├── Space.h, Layout.h  # compile-time memory-space / layout tags
│   ├── Buffer.h, Matrix.h, MatrixView.h, Copy.h
│   ├── Arch.h, dtypes.h   # arch name + dtype aliases/names
│   ├── bench/             # Profiler, GemmArgs, KernelTraits, Accuracy,
│   │   ├── Stats.h, Fill.h, CsvWriter.h
│   │   └── microbench/    # memcpy + launch-overhead microbenches
│   ├── cublas/            # CublasHandle, cublas_gemm
│   ├── sm90/              # Hopper-specific kernels
│   └── sm120/             # Blackwell-specific kernels
├── tests/
│   └── test.cu            # unit tests + build-verification
└── scripts/              # (Phase 2B) plot.py + csv_loader.py
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


### Visualization
- **Log-log plot**: x = `N` (log), y = `time_ns` (log).
- One subplot per `(arch, dtype)`; one line per `kernel`.
- Reference line: cuBLAS. Goal: custom kernel line ≤ cuBLAS line.
- Tool: Python + matplotlib (script in `scripts/plot.py`, not yet created).

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

## Profiling Tools
- Profiling will be setup later (nsys & ncu)
