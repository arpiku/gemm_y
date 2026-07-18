// launch_overhead_microbench.cu — Chunk 2.4.
//
// Measures per-launch overhead of an empty kernel via CudaTimer. Sets the
// floor for kernel time interpretation (a kernel reporting < launch overhead
// is measurement noise). Documented in ARD.md.

#include <algorithm>
#include <cstdio>
#include <vector>

#include "CudaCheck.h"
#include "CudaTimer.h"
#include "cuda_compat.h"

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
    std::sort(ms.begin(), ms.end());
    const double min_ns = static_cast<double>(ms.front()) * 1e6;
    const double med_ns = (kIters % 2 == 0)
        ? static_cast<double>(ms[kIters/2 - 1] + ms[kIters/2]) * 0.5 * 1e6
        : static_cast<double>(ms[kIters/2]) * 1e6;
    const double max_ns = static_cast<double>(ms.back()) * 1e6;
    std::printf("launch_overhead: iters=%d  min=%.0f ns  median=%.0f ns  max=%.0f ns\n",
                kIters, min_ns, med_ns, max_ns);
    return 0;
}

} // namespace gemm_y

int main() {
    return gemm_y::run_launch_overhead_main();
}
