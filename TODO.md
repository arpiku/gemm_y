# TODO.md — Current documentation prologue

## Objective

Make the kernel-development workflow explicit before changing kernel behavior. This work documents the existing contracts, experiment identity, registration flow, and review loop for the upcoming `sm120` bf16 `k0` optimization.

No kernel implementation changes are part of this task.

## 1. Document the current kernel boundary

- [x] Confirm the durable device-kernel contract in `AGENTS.md`:

  ```cpp
  __global__ void kernel(
      const T* A, const T* B, T* C,
      int M, int N, int K,
      int ldA, int ldB, int ldC);
  ```

- [x] Confirm that device kernels use raw pointers and dimensions only.
- [x] Confirm that `GemmArgs<T>` and `MatrixView` belong to the host-side harness adapter.
- [x] Confirm the ColMajor addressing convention:

  ```text
  A(i,k) = A[i + k*ldA]
  B(k,j) = B[k + j*ldB]
  C(i,j) = C[i + j*ldC]
  ```

- [x] Confirm that every kernel must fully overwrite `C`, support valid leading dimensions, and guard tile boundaries.
- [x] Do not modify `src/sm120/gemm_bf16_k0.cu` during this documentation task.

## 2. Document kernel identity and file organization

For each experiment:

- [x] Use one implementation file per kernel:

  ```text
  src/sm120/gemm_bf16_k0.cu
  src/sm120/gemm_bf16_k1.cu
  src/sm120/gemm_bf16_k2.cu
  ```

- [x] Keep the declaration and harness-facing metadata in:

  ```text
  src/sm120/gemm_bf16.cuh
  ```

- [x] Keep development names stable by kernel family: `k0`, `k1`, `k2`, etc.
- [x] Treat each compile-time specialization as a separately benchmarked variant.
- [x] Give every registered specialization a unique name, for example:

  ```text
  k0_64x64_k32_w4_s2
  k0_128x128_k32_w8_s2
  ```

- [x] Make `description()` identify the actual strategy and its important parameters.
- [x] Use descriptions in this form once implementation begins:

  ```cpp
  static constexpr std::string_view description() {
      return "bf16 tensor-core GEMM; tile=<MxN>; k=<K>; "
             "warps=<count>; stages=<count>";
  }
  ```

- [x] Do not encode historical experiment labels or undocumented abbreviations in source comments or metadata.

### Template and autotuning rules

- [x] Use templates for compile-time kernel configuration such as tile sizes,
      warp count, K tile size, staging depth, layouts, or swizzles.
- [x] Keep runtime GEMM arguments (`M`, `N`, `K`, `ldA`, `ldB`, `ldC`) as kernel
      parameters; do not encode matrix dimensions in template parameters.
- [x] Keep registered functors default-constructible and stateless initially;
      `Profiler::register_kernel<K>()` dispatches `K{}`.
- [x] Explicitly instantiate every registered specialization in its `.cu` file.
- [x] Benchmark each candidate specialization separately. Do not hide a tuning
      search inside one timed `operator()` call.
- [x] Keep the candidate matrix small to control CUDA compile time and binary
      size.

Example family declaration:

```cpp
template<int TileM, int TileN, int TileK, int Warps, int Stages>
struct k0 {
    static constexpr std::string_view name();
    static constexpr std::string_view description();
    void operator()(GemmArgs<__nv_bfloat16>, cudaStream_t) const;
};
```

## 3. Document the registration flow

Current registration is performed by `profile_bf16_kernels()` in `src/main.cpp`:

```cpp
prof.register_kernel<gemm_y::NaiveGemm<T>>();
prof.register_kernel<gemm_y::k0>();
```

The same kernels are separately listed for `.meta` output. Before adding more experiments:

- [x] Replace the duplicated registration and metadata code with one helper.
- [x] The helper must register the kernel and append its metadata together:

  ```cpp
  template <typename K, typename T>
  void register_bf16_kernel(
      gemm_y::Profiler<T>& profiler,
      std::vector<std::pair<std::string, std::string>>& metadata) {
      profiler.template register_kernel<K>();
      metadata.emplace_back(
          std::string(K::name()),
          std::string(K::description()));
  }
  ```

- [x] Preserve registration order: cuBLAS reference, `naive`, then `k0` and later experiments.
- [x] Register concrete template specializations, not an unexpanded template family.
- [x] Ensure every registered kernel appears in the generated metadata with a unique name.
- [x] Do not change profiler timing, accuracy, or dispatch behavior as part of this cleanup.

## 4. Define the upcoming `k0` experiment record

The first implementation task will optimize the existing workflow-only `sm120` bf16 `k0`. It is intentionally deferred until this documentation task is complete.

Before implementation begins, define the record for each experiment:

```text
kernel: k0
architecture: sm120
dtype: bf16
CTA tile: <M>x<N>
K tile: <K>
warp count: <count>
stages: <count>
operand layout: <description>
accumulator: fp32
output: bf16
correctness: PASS/FAIL
max relative error: <value>
median time by N: <benchmark result>
comparison: naive / cuBLAS
notes: <one-variable change and outcome>
```

Each later experiment must change one principal variable at a time, for example:

- CTA tile shape.
- K tile size.
- Warp count.
- Shared-memory staging depth.
- Operand layout or swizzle.
- Global-memory loading strategy.

Keep slower or failed experiments available for comparison. Do not overwrite their results or hide them from the benchmark history.

## 5. Review and validation plan after documentation cleanup

No kernel benchmark is required for this documentation-only task. After the cleanup is reviewed, the implementation loop will be:

1. Implement one `k0` kernel change in `src/sm120/gemm_bf16_k0.cu`.
2. Update `k0::description()` with the actual strategy parameters.
3. Build the selected architecture:

   ```sh
   cmake -B build
   cmake --build build -j
   ```

4. Run the existing smoke tests:

   ```sh
   ctest --test-dir build
   ```

5. Run the reproducible benchmark:

   ```sh
   sudo scripts/bench.sh
   ```

6. Use the profiler accuracy gate to determine whether the output is valid.
7. Compare accepted timing results against `naive` and cuBLAS across:

   ```text
   32, 64, 96, 128, 192, 256, 384, 512, 768,
   1024, 1536, 2048, 3072, 4096
   ```

8. Record the result using the experiment format above before making the next change.

## Completion criteria for this prologue

- [x] `AGENTS.md` states that the current work is documentation-first.
- [x] The raw-pointer device ABI and host adapter boundary are documented.
- [x] Per-kernel file organization and identity rules are documented.
- [x] Registration and metadata deduplication is specified.
- [x] The `k0` experiment record format is defined.
- [x] The post-cleanup implementation and validation loop is explicit.
- [x] No kernel implementation or behavior has been changed.
