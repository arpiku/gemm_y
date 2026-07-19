// launch_overhead_microbench.cu — measures per-launch overhead of an empty
// kernel via CudaTimer. Sets the floor for kernel time interpretation (a
// kernel reporting < launch overhead is measurement noise). Documented in
// ARD.md. Prints a human-readable aligned table.

#include <string>
#include <vector>

#include "CudaCheck.h"
#include "CudaTimer.h"
#include "cuda_compat.h"
#include "bench/Stats.h"
#include "bench/microbench/print_table.h"

namespace gemm_y {

__global__ void empty_kernel() {}

int run_launch_overhead_main() {
    constexpr int kIters = 1000;
    std::vector<float> ms; ms.reserve(kIters);
    CudaTimer t;
    // Warmup
    for (int i = 0; i < 50; ++i) {
        empty_kernel<<<1, 1>>>();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < kIters; ++i) {
        t.start();
        empty_kernel<<<1, 1>>>();
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    const double max_ns = static_cast<double>(ms.back()) * 1e6;

    bench::Table tbl({"metric", "value (ns)"});
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d", kIters);
    tbl.add_row({"iters", std::string(buf)});
    std::snprintf(buf, sizeof(buf), "%.0f", s.min_ns);
    tbl.add_row({"min", std::string(buf)});
    std::snprintf(buf, sizeof(buf), "%.0f", s.median_ns);
    tbl.add_row({"median", std::string(buf)});
    std::snprintf(buf, sizeof(buf), "%.0f", max_ns);
    tbl.add_row({"max", std::string(buf)});
    tbl.print();
    return 0;
}

} // namespace gemm_y

int main() {
    return gemm_y::run_launch_overhead_main();
}
