// memcpy_microbench.cu — Chunk 2.2 (Phase 1.5: relocated to microbench/ subdir).
//
// Sweeps H2D / D2H memcpy variants across N in {32,64,128,256,512,1024,
// 2048,4096} and reports min + median per variant. Warmup=20, timed=50
// (see ARD.md §11).
//
// Variants per direction:
//   - cudaMemcpy       (sync, contiguous)
//   - cudaMemcpyAsync  (default stream + sync)
//   - cudaMemcpy2D     (strided: 4096x4096 buffer, N x N submatrix, ld=4096)
//
// Hypothesis: sync vs async identical (no overlap); cudaMemcpy2D is the
// only correct option for strided submatrix copies. Decision recorded in
// ARD.md §2.
//
// Phase 1.5 (R10): prints a human-readable aligned table instead of raw CSV.

#include <string>
#include <vector>

#include "CudaCheck.h"
#include "CudaTimer.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Space.h"
#include "cuda_compat.h"
#include "bench/Stats.h"
#include "bench/microbench/print_table.h"

namespace gemm_y {
namespace {

constexpr int kMaxN = 4096;
constexpr int kWarmup = 20;
constexpr int kTimed = 50;

// Helper: format a double ns value as a string.
std::string ns(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0f", v);
    return std::string(buf);
}

std::string i(int v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", v);
    return std::string(buf);
}

void bench_h2d_contiguous_sync(int N, bench::Table& t) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t_;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.bytes(), cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < kTimed; ++i) {
        t_.start();
        CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.bytes(), cudaMemcpyHostToDevice));
        t_.stop();
        ms.push_back(t_.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    t.add_row({"h2d", "contiguous_sync", i(N), ns(s.min_ns), ns(s.median_ns)});
}

void bench_h2d_async(int N, bench::Table& t) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t_;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.bytes(),
                                   cudaMemcpyHostToDevice, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    for (int i = 0; i < kTimed; ++i) {
        t_.start();
        CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.bytes(),
                                   cudaMemcpyHostToDevice, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        t_.stop();
        ms.push_back(t_.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    t.add_row({"h2d", "async_default_stream", i(N), ns(s.min_ns), ns(s.median_ns)});
}

void bench_h2d_strided_2d(int N, bench::Table& t) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(kMaxN, kMaxN);
    auto hsub = h.view().block(0, 0, N, N);
    auto dsub = d.view().block(0, 0, N, N);
    const std::size_t width = static_cast<std::size_t>(N) * sizeof(float);
    const std::size_t spitch = static_cast<std::size_t>(kMaxN) * sizeof(float);
    const std::size_t dpitch = spitch;
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t_;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy2D(dsub.ptr, dpitch, hsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < kTimed; ++i) {
        t_.start();
        CUDA_CHECK(cudaMemcpy2D(dsub.ptr, dpitch, hsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyHostToDevice));
        t_.stop();
        ms.push_back(t_.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    t.add_row({"h2d", "strided_2d", i(N), ns(s.min_ns), ns(s.median_ns)});
}

void bench_d2h_contiguous_sync(int N, bench::Table& t) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t_;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.bytes(), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < kTimed; ++i) {
        t_.start();
        CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.bytes(), cudaMemcpyDeviceToHost));
        t_.stop();
        ms.push_back(t_.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    t.add_row({"d2h", "contiguous_sync", i(N), ns(s.min_ns), ns(s.median_ns)});
}

void bench_d2h_async(int N, bench::Table& t) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t_;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.bytes(),
                                   cudaMemcpyDeviceToHost, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    for (int i = 0; i < kTimed; ++i) {
        t_.start();
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.bytes(),
                                   cudaMemcpyDeviceToHost, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        t_.stop();
        ms.push_back(t_.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    t.add_row({"d2h", "async_default_stream", i(N), ns(s.min_ns), ns(s.median_ns)});
}

void bench_d2h_strided_2d(int N, bench::Table& t) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(kMaxN, kMaxN);
    auto hsub = h.view().block(0, 0, N, N);
    auto dsub = d.view().block(0, 0, N, N);
    const std::size_t width = static_cast<std::size_t>(N) * sizeof(float);
    const std::size_t spitch = static_cast<std::size_t>(kMaxN) * sizeof(float);
    const std::size_t dpitch = spitch;
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t_;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy2D(hsub.ptr, dpitch, dsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < kTimed; ++i) {
        t_.start();
        CUDA_CHECK(cudaMemcpy2D(hsub.ptr, dpitch, dsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyDeviceToHost));
        t_.stop();
        ms.push_back(t_.elapsed_ms());
    }
    const bench::TimedStats s = bench::summarize_ns(ms);
    t.add_row({"d2h", "strided_2d", i(N), ns(s.min_ns), ns(s.median_ns)});
}

} // namespace

int run_memcpy_microbench_main() {
    bench::Table t({"direction", "variant", "N", "min (ns)", "median (ns)"});
    const int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (int N : sizes) {
        bench_h2d_contiguous_sync(N, t);
        bench_h2d_async(N, t);
        bench_h2d_strided_2d(N, t);
        bench_d2h_contiguous_sync(N, t);
        bench_d2h_async(N, t);
        bench_d2h_strided_2d(N, t);
    }
    t.print();
    return 0;
}

} // namespace gemm_y

int main() {
    return gemm_y::run_memcpy_microbench_main();
}
