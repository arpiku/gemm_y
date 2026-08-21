# TODO.md — Templated kernel profiling and offline dispatch

## Current state

- `k1_smem` is the fixed shared-memory control and remains active in the full benchmark.
- `k1_t<64,64,16>` is implemented with dynamic shared memory and is named `k1_t`.
- `k1_t` passes the current focused smoke and padded-leading-dimension checks.
- `k0_ldg` and `k0_coalesced` are completed kernels retained for smoke/regression validation only.
- The full benchmark contains the generic baseline and active `k1_smem`; tuning candidates are not registered there.
- Candidate tuning uses a separate profile executable and must not modify `db/gemm_y.db`.
- The user manually ingests selected CSV results.

## 1. Base `k1_t` result

- [x] Keep `k1_t<64,64,16>` as the canonical base candidate with the short name `k1_t`.
- [x] Keep `k1_smem` unchanged as the fixed comparison and fallback control.
- [x] Use compile-time `TileM`, `TileN`, and `TileK` parameters with a fixed `16x16` thread block.
- [x] Use single-buffered dynamic shared memory and pass the computed byte count at launch.
- [x] Preserve arbitrary valid leading dimensions, ColMajor addressing, boundary guards, and FP32 accumulation.
- [x] Register `k1_t` in `kernel_smoke` and validate contiguous, partial-tile, K-boundary, and padded-leading-dimension cases.
- [x] Build and run:

  ```sh
  cmake --build build --target kernel_smoke -j
  ./build/kernel_smoke
  ctest --test-dir build
  ```

- [ ] Run the full active benchmark for the base `k1_t` result. Keep the full registry limited to the active kernel set; do not add completed k0 controls.
- [ ] Preserve the resulting CSV and `.meta` files, then manually ingest the selected base result if desired.

## 2. Candidate profile for the tuning branch

Run this on the separate tuning branch after the base result is recorded.

- [ ] Keep the candidate set explicit and small initially:

  ```text
  k1_t
  k1_t_32x64x16
  k1_t_64x32x16
  k1_t_64x64x8
  ```

- [ ] Do not register the candidate set in `src/main.cpp` or the normal full benchmark.
- [ ] Ensure each concrete specialization has exactly one unique benchmark name. Do not create a second identity for `k1_t<64,64,16>`.
- [ ] Keep `k1_smem` in the profile as the fixed fallback/control.
- [ ] Use the complete supported square sweep:

  ```text
  32, 64, 96, 128, 192, 256, 384, 512, 768,
  1024, 1536, 2048, 3072, 4096
  ```

- [ ] Make the tuning executable emit reusable CSV and sidecar metadata containing accuracy-passing rows, CUDA-event timing statistics, candidate descriptions, architecture, and dtype.
- [ ] Filter correctness failures before using timing rows.
- [ ] Compare candidates by `kernel_median_ns`; retain p95/std/CI for stability review and deterministic tie-breaking.
- [ ] Preserve the profile CSV and metadata for manual review. Never modify `db/gemm_y.db` automatically.

## 3. Offline policy generation

- [x] Add `scripts/select_kernel.py` with CSV-only input and no database access.
- [x] Use `k1_smem` as the deterministic fallback.
- [x] Use an explicit improvement threshold, initially 2%, against the fallback median.
- [x] Generate an explicit compile-time `switch (N)` policy for the current square-GEMM scope.
- [x] Reject unsupported architecture/dtype policy generation.
- [x] Keep the canonical candidate map consistent with the concrete functor names.
- [ ] Generate a policy from the real tuning CSV, not only `tests/dispatch_fixture.csv`.
- [ ] Record source identifier, architecture, dtype, selected candidate per size, fallback, metric, threshold, and accuracy tolerance in generated-header metadata.
- [ ] Keep generated policy files under `build/generated` or another explicitly requested output directory; do not hand-edit them.

## 4. Dispatcher validation

- [x] Add a fixture-based policy generation check in the build.
- [x] Add `k1_dispatch` as a thin generated-policy functor.
- [x] Keep dispatcher execution free of benchmarking, measurement synchronization, allocation, file I/O, SQLite access, and runtime string lookup.
- [x] Add generated-policy name inspection so tests can verify selected candidates and fallback behavior.
- [ ] Generate the policy from the real comprehensive profile.
- [ ] Verify every profiled size selects the expected candidate or `k1_smem` fallback.
- [ ] Verify unsupported sizes use the fallback.
- [ ] Run dispatcher correctness against cuBLAS and compare against the selected candidate.
- [ ] Measure dispatcher timing separately and confirm it matches the selected candidate within normal measurement noise.
- [ ] Do not promote `k1_dispatch` into `src/main.cpp` until the profile and dispatcher checks pass.

## 5. Validation and promotion

- [ ] After candidate/profile changes, run:

  ```sh
  cmake --build build --target kernel_smoke kernel_tuning dispatch_check -j
  ./build/kernel_smoke
  ./build/dispatch_check
  ctest --test-dir build
  ```

- [ ] Run the comprehensive candidate profile with the user-run clock-locked benchmark wrapper when reproducible timing is required.
- [ ] Compare candidates and dispatcher against cuBLAS and `k1_smem`.
- [ ] Promote only the validated dispatcher or selected active candidate to the full benchmark registration.
- [ ] Remove superseded kernels from the full benchmark registry without deleting their source, smoke coverage, or historical results.
- [ ] Keep the final production registration explicit and free of commented-out candidates or runtime lifecycle flags.
- [ ] Manually ingest only the benchmark CSV selected by the user, using `--custom-only` when explicit cuBLAS rows should not be duplicated.

## Completion criteria

- [ ] Base `k1_t` benchmark result is preserved and manually ingested if selected.
- [ ] Comprehensive candidate profile covers the full square sweep and emits CSV plus metadata.
- [ ] Generated policy is deterministic, auditable, and database-independent.
- [ ] Dispatcher selected-name, fallback, accuracy, and timing checks pass.
- [ ] Production registration contains only the promoted active strategy.
- [ ] `db/gemm_y.db` remains unchanged until the user explicitly ingests selected results.
