// tests/dispatch_check.cu — runtime check for generated offline dispatch.

#include <cstdio>
#include <cstdlib>
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
  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 2;
  const std::size_t expected = sizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected) {
    std::fprintf(stderr, "dispatch check failed: expected %zu rows, got %zu\n",
                 expected, result.rows.size());
    return EXIT_FAILURE;
  }
  for (const auto &row : result.rows) {
    if (row.kernel_name == "k1_dispatch" &&
        row.max_rel_err > gemm_y::kRelErrTol<T>()) {
      std::fprintf(stderr, "dispatch check failed at N=%d: rel_err=%.3e\n",
                   row.N, row.max_rel_err);
      return EXIT_FAILURE;
    }
  }
  std::printf("dispatch check passed: %zu rows\n", result.rows.size());
  return EXIT_SUCCESS;
}
