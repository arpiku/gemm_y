// tests/kernel_tuning.cu — reusable full-sweep profile for explicit k1_t
// candidates.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

#include "Arch.h"
#include "bench/Accuracy.h"
#include "bench/CsvWriter.h"
#include "bench/Profiler.h"
#include "dtypes.h"

#if defined(CUDA_ARCH_SM_120)
#include "sm120/gemm_bf16.cuh"
#else
#error "kernel_tuning requires sm_120"
#endif

namespace {

const std::vector<int> kSweepSizes = {32,  64,  96,   128,  192,  256,  384,
                                      512, 768, 1024, 1536, 2048, 3072, 4096};

void write_csv(const gemm_y::SweepResult &result, const std::string &path) {
  gemm_y::CsvWriter csv;
  if (!csv.open(path)) {
    std::fprintf(stderr, "Failed to open %s for write\n", path.c_str());
    std::abort();
  }
  csv.write_header(
      "arch,dtype,N,kernel_name,kernel_desc,"
      "kernel_min_ns,kernel_median_ns,kernel_std_ns,kernel_p95_ns,"
      "kernel_ci_low_ns,kernel_ci_high_ns,"
      "ref_kernel_min_ns,ref_kernel_median_ns,ref_kernel_std_ns,"
      "ref_kernel_p95_ns,ref_kernel_ci_low_ns,ref_kernel_ci_high_ns,"
      "max_abs_err,max_rel_err,kernel_samples_ns");
  for (const auto &row : result.rows) {
    csv.append_row(row.arch, row.dtype, row.N, row.kernel_name, row.kernel_desc,
                   row.kernel_min_ns, row.kernel_median_ns, row.kernel_std_ns,
                   row.kernel_p95_ns, row.kernel_ci_low_ns,
                   row.kernel_ci_high_ns, row.ref_kernel_min_ns,
                   row.ref_kernel_median_ns, row.ref_kernel_std_ns,
                   row.ref_kernel_p95_ns, row.ref_kernel_ci_low_ns,
                   row.ref_kernel_ci_high_ns, row.max_abs_err, row.max_rel_err,
                   row.kernel_samples_ns);
  }
}

std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

void write_meta(
    const std::string &path,
    const std::vector<std::pair<std::string, std::string>> &kernels) {
  std::FILE *file = std::fopen(path.c_str(), "w");
  if (file == nullptr) {
    std::fprintf(stderr, "Failed to open %s for write\n", path.c_str());
    std::abort();
  }
  std::fprintf(file, "arch=%s\n", gemm_y::kArchName);
  std::fprintf(file, "dtype=bf16\n");
  std::fprintf(file, "warmup_iters=20\n");
  std::fprintf(file, "timed_iters=50\n");
  std::fprintf(file, "tol=%.6g\n", gemm_y::kRelErrTol<gemm_y::dtypes::bf16>());
  std::fprintf(file, "sweep_sizes=");
  for (std::size_t i = 0; i < kSweepSizes.size(); ++i) {
    std::fprintf(file, "%s%d", i == 0 ? "" : ",", kSweepSizes[i]);
  }
  std::fprintf(file, "\nprofile=fixed-fallback-plus-compile-time-candidates\n");
  std::fprintf(file, "kernel=cublas|cublasGemmEx reference (fp32 accum)\n");
  for (const auto &[name, description] : kernels) {
    std::fprintf(file, "kernel=%s|%s\n", name.c_str(), description.c_str());
  }
  std::fprintf(file, "timestamp=%s\n", timestamp().c_str());
  std::fclose(file);
}

template <typename Kernel>
void register_kernel(
    gemm_y::Profiler<gemm_y::dtypes::bf16> &profiler,
    std::vector<std::pair<std::string, std::string>> &metadata) {
  profiler.template register_kernel<Kernel>();
  metadata.emplace_back(std::string(Kernel::name()),
                        std::string(Kernel::description()));
}

} // namespace

int main() {
  using T = gemm_y::dtypes::bf16;
  gemm_y::Profiler<T> profiler;
  std::vector<std::pair<std::string, std::string>> metadata;
  register_kernel<gemm_y::k1_smem>(profiler, metadata);
  register_kernel<gemm_y::k1_t<64, 64, 16>>(profiler, metadata);
  register_kernel<gemm_y::k1_t<32, 64, 16>>(profiler, metadata);
  register_kernel<gemm_y::k1_t<64, 32, 16>>(profiler, metadata);
  register_kernel<gemm_y::k1_t<64, 64, 8>>(profiler, metadata);

  const gemm_y::SweepResult result = profiler.run_sweep(kSweepSizes);
  constexpr std::size_t kExpectedRowsPerN = 6;
  const std::size_t expected = kSweepSizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected) {
    std::fprintf(stderr, "kernel tuning failed: expected %zu rows, got %zu\n",
                 expected, result.rows.size());
    return EXIT_FAILURE;
  }

  const std::string stem =
      "results/tuning_" + std::string(gemm_y::kArchName) + "_bf16";
  write_csv(result, stem + ".csv");
  write_meta(stem + ".meta", metadata);
  std::printf("kernel tuning passed: %zu rows across %zu sizes\n",
              result.rows.size(), kSweepSizes.size());
  std::printf("kernel tuning wrote %s.csv and %s.meta\n", stem.c_str(),
              stem.c_str());
  return EXIT_SUCCESS;
}
