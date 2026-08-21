// tests/dispatch_check.cu — runtime check for generated offline dispatch.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/Accuracy.h"
#include "bench/Profiler.h"
#include "dtypes.h"
#include "generated/k1_policy.h"

int main() {
  using T = gemm_y::dtypes::bf16;
  const std::vector<int> sizes = {32, 256};
  if (gemm_y::k1_dispatch::selected_kernel_name(32) != "k1_t_32x64x16" ||
      gemm_y::k1_dispatch::selected_kernel_name(256) != "k1_smem" ||
      gemm_y::k1_dispatch::selected_kernel_name(999) != "k1_smem") {
    std::fprintf(stderr,
                 "dispatch check failed: unexpected generated policy\n");
    return EXIT_FAILURE;
  }

  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k1_dispatch>();
  profiler.register_kernel<gemm_y::k1_smem>();
  profiler.register_kernel<gemm_y::k1_t<32, 64, 16>>();
  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 4;
  const std::size_t expected = sizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected) {
    std::fprintf(stderr, "dispatch check failed: expected %zu rows, got %zu\n",
                 expected, result.rows.size());
    return EXIT_FAILURE;
  }
  for (const int n : sizes) {
    const gemm_y::SweepRow *dispatch = nullptr;
    const gemm_y::SweepRow *selected = nullptr;
    for (const auto &row : result.rows) {
      if (row.N != n)
        continue;
      if (row.kernel_name == "k1_dispatch")
        dispatch = &row;
      if ((n == 32 && row.kernel_name == "k1_t_32x64x16") ||
          (n == 256 && row.kernel_name == "k1_smem"))
        selected = &row;
    }
    if (dispatch == nullptr || selected == nullptr ||
        dispatch->max_rel_err > gemm_y::kRelErrTol<T>() ||
        selected->max_rel_err > gemm_y::kRelErrTol<T>()) {
      std::fprintf(stderr,
                   "dispatch check failed at N=%d: missing/invalid row\n", n);
      return EXIT_FAILURE;
    }
    const double ratio =
        dispatch->kernel_median_ns / selected->kernel_median_ns;
    if (ratio < 0.5 || ratio > 2.0) {
      std::fprintf(stderr, "dispatch check failed at N=%d: timing ratio=%.3f\n",
                   n, ratio);
      return EXIT_FAILURE;
    }
  }
  std::printf("dispatch check passed: %zu rows\n", result.rows.size());
  return EXIT_SUCCESS;
}
