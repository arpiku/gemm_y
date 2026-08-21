// tests/kernel_smoke.cu — fast smoke test for newly added sm120 bf16 kernels.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/Profiler.h"
#include "dtypes.h"

#if defined(CUDA_ARCH_SM_90)
#error "kernel_smoke currently targets sm_120 bf16 experiments only"
#elif defined(CUDA_ARCH_SM_120)
#include "sm120/gemm_bf16.cuh"
#else
#error "kernel_smoke requires a supported CUDA architecture"
#endif

int main() {
  using T = gemm_y::dtypes::bf16;
  const std::vector<int> sizes = {32, 96, 128};
  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k0_ldg>();
  profiler.register_kernel<gemm_y::k0_coalesced>();

  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 3;
  const std::size_t expected_rows = sizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected_rows) {
    std::fprintf(stderr, "kernel smoke failed: expected %zu rows, got %zu\n",
                 expected_rows, result.rows.size());
    return EXIT_FAILURE;
  }

  for (const int size : sizes) {
    bool has_cublas = false;
    bool has_ldg = false;
    bool has_coalesced = false;
    for (const auto &row : result.rows) {
      if (row.N != size)
        continue;
      if (row.kernel_name == "cublas")
        has_cublas = true;
      if (row.kernel_name == "k0_ldg")
        has_ldg = true;
      if (row.kernel_name == "k0_coalesced")
        has_coalesced = true;
    }
    if (!has_cublas || !has_ldg || !has_coalesced) {
      std::fprintf(stderr,
                   "kernel smoke failed at N=%d: cublas=%d k0_ldg=%d "
                   "k0_coalesced=%d\n",
                   size, has_cublas, has_ldg, has_coalesced);
      return EXIT_FAILURE;
    }
  }

  std::printf("kernel smoke passed: %zu rows across %zu sizes\n",
              result.rows.size(), sizes.size());
  return EXIT_SUCCESS;
}
