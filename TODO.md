# TODO.md — Current production state and future tuning policy

## Current production state

- The promoted `sm_120/bf16` source dispatcher is `k1_dispatch` in
  `src/sm120/k1_dispatch.cuh`.
- Its current policy for the supported square benchmark sizes is:

  ```text
  32,64                    -> k1_t_32x64x16
  96,128,192,256,384,512   -> k1_t_64x32x16
  768                      -> k1_t_64x64x8
  1024,1536,2048,3072,4096 -> k1_t
  unsupported sizes        -> k1_smem
  ```

- `k1_smem` remains the comparison control and deterministic fallback.
- Production registration contains `naive`, `k1_smem`, and `k1_dispatch`; the
  individual tuned variants are not registered in the normal full benchmark.
- Benchmark review and database ingestion are manual user actions. Development,
  building, testing, and dispatching must not modify `db/gemm_y.db`.

## Completed `k1_t_x` reference experiment

- `src/sm120/gemm_bf16_k1_t_x.cu` preserves an independently identifiable
  `k1_t_x` implementation based on the `64x64x16` `k1_t` configuration.
- The experiment tested explicit `__fmaf_rn` accumulation, targeted loop
  unrolling, and compile-time shift/mask indexing for power-of-two dimensions.
- Focused CUDA-event timings showed no meaningful improvement over `k1_t`; the
  candidate also used more registers. It is therefore retained for reference
  only and is not part of the smoke registry, production registry, or dispatch
  policy.
- No `k1_t_x` promotion is pending. Future optimization work should begin from
  the production dispatcher and change one kernel variable at a time.

## Future tuning policy

- Recreate a small, purpose-built profile and offline selector only after a
  kernel family or major optimization step is sound and tuning is justified.
- The temporary workflow may be rewritten for the next kernel family rather
  than maintained as a generic autotuning framework.
- Candidate registration belongs only in the temporary tuning/profile
  executable. It must collect accuracy-passing CUDA-event timings and must not
  modify SQLite automatically.
- Materialize a validated policy into committed source, then delete temporary
  selectors, profile executables, generated headers, fixtures, and tuning-only
  tests. Preserve historical benchmark results when useful.
- Keep the policy key `(arch, dtype, N)` while GEMM supports only square shapes;
  retain the raw-pointer and leading-dimension kernel ABI for future extension.
- New or materially modified CUDA kernels must check launch errors. Do not add
  unrelated error-checking cleanup to old kernels as part of future work.
