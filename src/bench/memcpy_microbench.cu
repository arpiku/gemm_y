// memcpy_microbench.cu — Chunk 2.2.
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

#include <algorithm>
#include <cstdio>
#include <vector>

#include "CudaCheck.h"
#include "CudaTimer.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Space.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace {

constexpr int kMaxN = 4096;
constexpr int kWarmup = 20;
constexpr int kTimed = 50;

struct Stats { double min_ns; double median_ns; };
Stats summarize(std::vector<float>& ms) {
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.min_ns = static_cast<double>(ms.front()) * 1e6;
    const std::size_t n = ms.size();
    s.median_ns = (n % 2 == 0)
        ? static_cast<double>(ms[n/2 - 1] + ms[n/2]) * 0.5 * 1e6
        : static_cast<double>(ms[n/2]) * 1e6;
    return s;
}

void bench_h2d_contiguous_sync(int N) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.bytes(), cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < kTimed; ++i) {
        t.start();
        CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.bytes(), cudaMemcpyHostToDevice));
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const Stats s = summarize(ms);
    std::printf("h2d,contiguous_sync,%d,%.0f,%.0f\n", N, s.min_ns, s.median_ns);
}

void bench_h2d_async(int N) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.bytes(),
                                   cudaMemcpyHostToDevice, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    for (int i = 0; i < kTimed; ++i) {
        t.start();
        CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.bytes(),
                                   cudaMemcpyHostToDevice, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const Stats s = summarize(ms);
    std::printf("h2d,async_default_stream,%d,%.0f,%.0f\n", N, s.min_ns, s.median_ns);
}

void bench_h2d_strided_2d(int N) {
    // 4096x4096 buffer, copy N x N submatrix with ld=4096.
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(kMaxN, kMaxN);
    auto hsub = h.view().block(0, 0, N, N);
    auto dsub = d.view().block(0, 0, N, N);
    const std::size_t width = static_cast<std::size_t>(N) * sizeof(float);
    const std::size_t spitch = static_cast<std::size_t>(kMaxN) * sizeof(float);
    const std::size_t dpitch = spitch;
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy2D(dsub.ptr, dpitch, hsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < kTimed; ++i) {
        t.start();
        CUDA_CHECK(cudaMemcpy2D(dsub.ptr, dpitch, hsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyHostToDevice));
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const Stats s = summarize(ms);
    std::printf("h2d,strided_2d,%d,%.0f,%.0f\n", N, s.min_ns, s.median_ns);
}

void bench_d2h_contiguous_sync(int N) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.bytes(), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < kTimed; ++i) {
        t.start();
        CUDA_CHECK(cudaMemcpy(h.data(), d.data(), h.bytes(), cudaMemcpyDeviceToHost));
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const Stats s = summarize(ms);
    std::printf("d2h,contiguous_sync,%d,%.0f,%.0f\n", N, s.min_ns, s.median_ns);
}

void bench_d2h_async(int N) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(N, N);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(N, N);
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.bytes(),
                                   cudaMemcpyDeviceToHost, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    for (int i = 0; i < kTimed; ++i) {
        t.start();
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.bytes(),
                                   cudaMemcpyDeviceToHost, nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const Stats s = summarize(ms);
    std::printf("d2h,async_default_stream,%d,%.0f,%.0f\n", N, s.min_ns, s.median_ns);
}

void bench_d2h_strided_2d(int N) {
    Matrix<float, Space::Host> h = Matrix<float, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<float, Space::Device> d = Matrix<float, Space::Device>::alloc(kMaxN, kMaxN);
    auto hsub = h.view().block(0, 0, N, N);
    auto dsub = d.view().block(0, 0, N, N);
    const std::size_t width = static_cast<std::size_t>(N) * sizeof(float);
    const std::size_t spitch = static_cast<std::size_t>(kMaxN) * sizeof(float);
    const std::size_t dpitch = spitch;
    std::vector<float> ms; ms.reserve(kTimed);
    CudaTimer t;
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpy2D(hsub.ptr, dpitch, dsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < kTimed; ++i) {
        t.start();
        CUDA_CHECK(cudaMemcpy2D(hsub.ptr, dpitch, dsub.ptr, spitch,
                               width, static_cast<std::size_t>(N),
                               cudaMemcpyDeviceToHost));
        t.stop();
        ms.push_back(t.elapsed_ms());
    }
    const Stats s = summarize(ms);
    std::printf("d2h,strided_2d,%d,%.0f,%.0f\n", N, s.min_ns, s.median_ns);
}

} // namespace

int run_memcpy_microbench_main() {
    std::printf("direction,variant,N,min_ns,median_ns\n");
    const int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (int N : sizes) {
        bench_h2d_contiguous_sync(N);
        bench_h2d_async(N);
        bench_h2d_strided_2d(N);
        bench_d2h_contiguous_sync(N);
        bench_d2h_async(N);
        bench_d2h_strided_2d(N);
    }
    return 0;
}

} // namespace gemm_y

int main() {
    return gemm_y::run_memcpy_microbench_main();
}
