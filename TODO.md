# TODO.md — Post-implementation validation and benchmark tooling

## Review status

The `sm120` bf16 low-level experiments are implemented:

- `k0_ldg` in `src/sm120/gemm_bf16_k0_ldg.cu`
- `k0_coalesced` in `src/sm120/gemm_bf16_k0_coalesced.cu`
- Both are declared in `src/sm120/gemm_bf16.cuh` and registered from `src/main.cpp`.
- Registration and metadata now share `register_bf16_kernel()`.
- Direct executable validation passed the reported sweep and accuracy checks.

The remaining clock-locked benchmark still belongs to the user because it requires `sudo`.

## Current goals

1. Avoid duplicating baseline cuBLAS rows in the database for every newer-kernel run while preserving the reference timing needed for speedup calculations.
2. Add a small executable that checks only newly added kernels over a few sizes before running the full test suite or benchmark.
3. Keep the full benchmark executable and existing `ctest` suite unchanged in purpose.

## 1. Custom-focused result ingestion

### Problem

Every `gemm_y` run writes cuBLAS and custom rows to the same CSV. The baseline cuBLAS data already exists, so repeatedly ingesting cuBLAS rows for each experiment adds duplicate reference measurements and makes newer-kernel runs harder to compare.

The custom CSV rows still contain these reference fields:

```text
ref_kernel_min_ns
ref_kernel_median_ns
ref_kernel_std_ns
ref_kernel_p95_ns
ref_kernel_ci_low_ns
ref_kernel_ci_high_ns
```

Therefore, custom-only ingestion must retain those fields and the dashboard must not require a separately inserted `kernel_name='cublas'` row for every run.

### Planned edits

- [x] Add an ingestion option to `scripts/ingest.py`, for example:

  ```sh
  python scripts/ingest.py results/bench_sm_120_bf16.csv \
      --custom-only --label "k0-ldg"
  ```

- [x] Make `--custom-only` skip CSV measurement rows where:

  ```python
  row["kernel_name"] == "cublas"
  ```

- [x] Continue inserting all custom rows, including their `ref_kernel_*` values.
- [x] Keep the existing default behavior available for complete historical runs.
- [x] Update the ingestion help text and module documentation.
- [x] Make the option explicit in the run metadata or database representation so a run is not mistaken for a complete cuBLAS/custom run.
- [x] Preserve idempotency behavior for `(source_csv, source_meta, git_sha)`.

### Dashboard/query changes required

The current `scripts/db.py` query path joins each custom measurement to a cuBLAS measurement from the same run to calculate comparison data. That join must be changed if custom-only runs are supported.

- [x] For custom rows, derive the cuBLAS comparison from the row's own `ref_kernel_*` fields.
- [x] Keep `kernel_name == 'cublas'` classification working for complete historical runs.
- [x] Keep speedup calculation equivalent:

  ```text
  speedup = ref_kernel_median_ns / kernel_median_ns
  ```

- [x] Preserve reference CI/std/p95 fields for the Speedup, Timing, and Diff views.
- [x] Ensure old complete runs with explicit cuBLAS rows continue to query correctly.
- [x] Ensure custom-only runs do not produce missing-reference rows or SQL join omissions.
- [x] Do not insert a synthetic cuBLAS measurement into each new run; the reference values already travel with each custom row.

### Ingestion validation

- [x] Test ingestion against one complete historical CSV.
- [x] Test ingestion with `--custom-only` against a newer CSV.
- [x] Verify the custom-only run contains no `cublas` measurement rows.
- [x] Verify custom rows retain non-null reference medians and CI fields.
- [x] Verify speedup and timing queries return the same reference comparison for both run types.
- [x] Verify ingesting the same source without `--force` remains idempotent.

## 2. Small kernel smoke executable

### Goal

Add a dedicated executable for fast sanity checking of newly written kernels. It must run only the selected new kernels over a small explicit size list and must not execute the full benchmark sweep.

The smoke executable should use the existing `Profiler<T>` so it receives the same:

- cuBLAS reference calculation
- accuracy comparison
- CUDA stream behavior
- launch-error checks
- failed-kernel filtering

It should register only the kernels under development:

```cpp
using T = gemm_y::dtypes::bf16;
gemm_y::Profiler<T> profiler;
profiler.register_kernel<gemm_y::k0_ldg>();
profiler.register_kernel<gemm_y::k0_coalesced>();
const auto result = profiler.run_sweep({32, 96, 128});
```

The exact small size list may be adjusted, but it must include:

- A small size such as `32`.
- A non-power-of-two or boundary-sensitive size such as `96`.
- A representative aligned size such as `128`.

### Required result checks

`Profiler::run_sweep()` skips invalid custom kernels instead of returning an error. The smoke executable must therefore validate the result explicitly:

```cpp
constexpr std::size_t kExpectedRowsPerN = 3; // cuBLAS + 2 custom kernels
const std::size_t expected_rows = sizes.size() * kExpectedRowsPerN;
if (result.rows.size() != expected_rows) {
    std::fprintf(stderr, "kernel smoke failed: missing kernel rows\n");
    return EXIT_FAILURE;
}
```

Also verify that each expected kernel name appears for every requested size. Do not treat the presence of cuBLAS rows alone as success.

### Planned files and build wiring

- [x] Add `tests/kernel_smoke.cu`.
- [x] Include the selected architecture's bf16 declaration header using the same architecture selection pattern as `main.cpp` and `tests/test.cu`.
- [x] Register only the newly implemented kernels.
- [x] Use a small local size vector; do not reuse the full `main.cpp` sweep.
- [x] Return nonzero when an expected custom row is missing or the profiler reports a failed kernel.
- [x] Add a `kernel_smoke` executable in `CMakeLists.txt` using the selected architecture sources and required bench sources.
- [x] Link it against `CUDA::cublas` and apply the normal CUDA/C++ flags.
- [x] Keep it separate from the full `test_cuda` executable.

Recommended commands:

```sh
cmake -B build
cmake --build build --target kernel_smoke -j
./build/kernel_smoke
```

The model must run `./build/kernel_smoke` after building. The user will run only commands requiring `sudo`.

### Optional CTest registration

- [x] Decide whether to register `kernel_smoke` as a separate CTest entry.
- [x] If registered, keep its name distinct from the full `test_cuda` test.
- [x] The direct executable command must remain available for quick iteration.

## 3. Documentation cleanup discovered during review

- [x] Update the stale top-level comments in `src/main.cpp` that still say tensor-core variants are deferred and only `NaiveGemm` is registered.
- [x] Update the stale `gemm_bf16.cuh` comments that say bf16 kernel structs have no template parameter if/when templated kernel families are introduced.
- [x] Keep comments focused on current behavior; do not reintroduce historical phase labels.

## Completion criteria

- [x] `AGENTS.md` requires the model to run produced executables where practical.
- [x] `AGENTS.md` makes clear that the user runs `sudo` benchmark commands.
- [x] Custom-only ingestion preserves reference timing and speedup behavior.
- [x] Complete historical ingestion remains supported.
- [x] `kernel_smoke` builds and runs independently of the full suite.
- [x] `kernel_smoke` validates every newly registered kernel at every smoke size.
- [x] The full `test_cuda` suite remains available.
- [x] No `sudo` command is run by the model.
