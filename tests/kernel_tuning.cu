// tests/kernel_tuning.cu — correctness/timing profile for explicit k1_t
// candidates.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench/Profiler.h"
#include "dtypes.h"

#if defined(CUDA_ARCH_SM_120)
#include "sm120/gemm_bf16.cuh"
#else
#error "kernel_tuning requires sm_120"
#endif

int main() {
  using T = gemm_y::dtypes::bf16;
  const std::vector<int> sizes = {32, 96, 128, 192, 256, 384, 512};
  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k1_smem>();
  profiler.register_kernel<gemm_y::k1_t<64, 64, 16>>();
  profiler.register_kernel<gemm_y::k1_t<32, 64, 16>>();
  profiler.register_kernel<gemm_y::k1_t<64, 32, 16>>();
  profiler.register_kernel<gemm_y::k1_t<64, 64, 8>>();

  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 6;
  const std::size_t expected = sizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected) {
    std::fprintf(stderr, "kernel tuning failed: expected %zu rows, got %zu\n",
                 expected, result.rows.size());
    return EXIT_FAILURE;
  }
  std::printf("kernel tuning passed: %zu rows across %zu sizes\n",
              result.rows.size(), sizes.size());
  return EXIT_SUCCESS;
}
