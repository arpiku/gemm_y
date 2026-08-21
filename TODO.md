# TODO.md — Promoted `k1_` dispatch policy

## Current state

- `k1_smem` is the fixed shared-memory comparison control and deterministic
  dispatcher fallback.
- The validated `sm_120/bf16` tuning profile selected a source-level policy for
  the supported square sizes:

  ```text
  32,64                    -> k1_t_32x64x16
  96,128,192,256,384,512   -> k1_t_64x32x16
  768                      -> k1_t_64x64x8
  1024,1536,2048,3072,4096 -> k1_t
  unsupported sizes        -> k1_smem
  ```

- The user owns benchmark-result review and database ingestion. Development,
  building, testing, and dispatching do not modify `db/gemm_y.db`.
- The generated policy and one-off tuning workflow were temporary promotion
  artifacts and have been removed after the policy was frozen in source.

## 1. Permanent dispatcher

- [x] Add `src/sm120/k1_dispatch.cuh` with the validated explicit `switch (N)`.
- [x] Keep the stable public name `k1_dispatch`.
- [x] Call concrete compile-time `k1_t` specializations directly.
- [x] Retain `k1_smem` as the deterministic unsupported-size fallback.
- [x] Keep `selected_kernel_name(int)` as a constexpr inspection helper.
- [x] Keep dispatch free of timing synchronization, allocation, I/O, SQLite,
      and runtime string lookup.
- [x] Keep candidate implementations independently available in source.

## 2. Production registration

- [x] Register `NaiveGemm`, `k1_smem`, and `k1_dispatch` in `src/main.cpp`.
- [x] Remove direct production registration of individual tuned variants.
- [x] Preserve explicit registration and metadata for the production CSV/meta
      output.
- [x] Keep completed k0 kernels available as smoke/regression controls without
      rerunning them in the normal full benchmark.

## 3. Tuning infrastructure cleanup

- [x] Remove the generated-policy CMake command and build dependency.
- [x] Remove the temporary `kernel_tuning`, `dispatch_check`, and
      `k1_dispatch_profile` targets.
- [x] Remove the temporary selector, profile/check sources, and fixture.
- [x] Keep the concrete `k1_t` implementations and historical result artifacts.
- [x] Keep the database untouched by this promotion.

## 4. Production validation

- [x] Update `kernel_smoke` to validate `k1_dispatch` at `{32, 96, 128}`.
- [x] Keep k0 controls and `k1_smem` in smoke/regression coverage.
- [x] Keep the padded-leading-dimension correctness check through the dispatcher.
- [x] Check unsupported-size fallback through `selected_kernel_name(999)`.
- [x] Build and run the production smoke target:

  ```sh
  cmake --build build --target kernel_smoke -j
  ./build/kernel_smoke
  ```

- [x] Build the complete production tree and run CTest:

  ```sh
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

## 5. User-run final benchmark and ingestion

After the production validation above passes, run the reproducible benchmark as
an ordinary user through the clock-locking wrapper:

```sh
sudo scripts/bench.sh ./build/gemm_y
```

Review that the generated `results/bench_sm_120_bf16.csv` and `.meta` contain:

```text
cublas, naive, k1_smem, k1_dispatch
```

The user may then manually ingest the selected production result. Use
`--custom-only --kernel-name k1_dispatch` if only dispatcher rows are wanted, or
`--custom-only` if both custom production kernels are wanted. Do not run `sudo`
or perform ingestion automatically as part of development.

## Future tuning policy

- Recreate a small architecture-/dtype-specific profile only for a justified
  future `k1_` optimization round.
- Register each concrete candidate exactly once in that temporary profile,
  collect accuracy-passing CUDA-event timings, select offline, materialize the
  result in `k1_dispatch.cuh`, and remove the temporary tooling again.
- Keep the policy key `(arch, dtype, N)` while GEMM supports only square shapes.
  Do not introduce runtime shape generalization prematurely.
