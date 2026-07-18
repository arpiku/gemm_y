// tests/test.cu — unit tests for Buffer / MatrixView / Matrix / CudaTimer /
// CublasHandle / cublas_gemm / Copy.
//
// No external test framework (AGENTS.md spirit: no external deps). A tiny
// hand-rolled assert macro prints failures and counts them. Returns nonzero
// from main() if any check failed.
//
// Also serves as the build-verification target (smoke test) — the existing
// trivial kernel round-trip is preserved at the bottom.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bench/Accuracy.h"
#include "Buffer.h"
#include "Copy.h"
#include "CudaCheck.h"
#include "CudaTimer.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "bench/Profiler.h"
#include "Space.h"
#include "cuda_compat.h"
#include "cublas/CublasHandle.h"
#include "cublas/cublas_gemm.h"

#if defined(CUDA_ARCH_SM_90)
    #define GEMM_Y_TARGET_ARCH_NAME "sm_90"
#elif defined(CUDA_ARCH_SM_120)
    #define GEMM_Y_TARGET_ARCH_NAME "sm_120"
#else
    #error "Neither CUDA_ARCH_SM_90 nor CUDA_ARCH_SM_120 is defined."
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            ++g_failures;                                                       \
        }                                                                      \
    } while (0)

#define CHECK_APPROX_EQ(a, b, tol)                                            \
    do {                                                                      \
        ++g_checks;                                                           \
        const double _a = static_cast<double>(a);                              \
        const double _b = static_cast<double>(b);                             \
        if (std::fabs(_a - _b) > (tol)) {                                     \
            std::fprintf(stderr,                                               \
                "FAIL %s:%d: |%.6g - %.6g| > %.6g\n",                          \
                __FILE__, __LINE__, _a, _b, static_cast<double>(tol));         \
            ++g_failures;                                                      \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Buffer tests
// ---------------------------------------------------------------------------
void test_buffer_host() {
    gemm_y::Buffer<float, gemm_y::Space::Host> b(64);
    CHECK(b.size() == 64);
    CHECK(b.bytes() == 64 * sizeof(float));
    CHECK(b.data() != nullptr);
    for (std::size_t i = 0; i < b.size(); ++i) b.data()[i] = static_cast<float>(i);
    for (std::size_t i = 0; i < b.size(); ++i)
        CHECK_APPROX_EQ(b.data()[i], static_cast<float>(i), 1e-9);
}

void test_buffer_device() {
    gemm_y::Buffer<float, gemm_y::Space::Device> b(64);
    CHECK(b.size() == 64);
    CHECK(b.bytes() == 64 * sizeof(float));
    CHECK(b.data() != nullptr);
    // Round-trip: fill host, copy to device, copy back, verify.
    std::vector<float> host_in(64), host_out(64, 0.0f);
    for (std::size_t i = 0; i < host_in.size(); ++i)
        host_in[i] = static_cast<float>(i * 2);
    CUDA_CHECK(cudaMemcpy(b.data(), host_in.data(), b.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(host_out.data(), b.data(), b.bytes(), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < host_out.size(); ++i)
        CHECK_APPROX_EQ(host_out[i], host_in[i], 1e-9);
}

// ---------------------------------------------------------------------------
// MatrixView tests
// ---------------------------------------------------------------------------
void test_matrixview_block() {
    // 8x8 host buffer, ld=8 (contiguous ColMajor).
    gemm_y::Buffer<float, gemm_y::Space::Host> b(64);
    for (int i = 0; i < 64; ++i) b.data()[i] = static_cast<float>(i);
    gemm_y::MatrixView<float, gemm_y::Space::Host> v(b.data(), 8, 8, 8);
    CHECK(v.rows == 8);
    CHECK(v.cols == 8);
    CHECK(v.ld == 8);
    CHECK(v.is_contiguous());

    // Submatrix at (2,3) of size 3x4. ld unchanged (8). ptr offset = 2 + 3*8 = 26.
    auto sub = v.block(2, 3, 3, 4);
    CHECK(sub.rows == 3);
    CHECK(sub.cols == 4);
    CHECK(sub.ld == 8);
    CHECK(!sub.is_contiguous());
    CHECK(sub.ptr == v.ptr + 2 + 3 * 8);
    // Element (0,0) of sub == element (2,3) of v == v.ptr[2 + 3*8] = 26.
    CHECK_APPROX_EQ(sub(0, 0), 26.0f, 1e-9);
    // Element (1,2) of sub == element (3,5) of v == v.ptr[3 + 5*8] = 43.
    CHECK_APPROX_EQ(sub(1, 2), 43.0f, 1e-9);
}

void test_matrixview_is_contiguous() {
    gemm_y::Buffer<float, gemm_y::Space::Host> b(64);
    gemm_y::MatrixView<float, gemm_y::Space::Host> full(b.data(), 8, 8, 8);
    CHECK(full.is_contiguous());

    auto sub = full.block(0, 0, 4, 4);  // ld=8, rows=4 -> not contiguous
    CHECK(!sub.is_contiguous());

    // A 4x4 view with ld=4 IS contiguous.
    gemm_y::MatrixView<float, gemm_y::Space::Host> c(b.data(), 4, 4, 4);
    CHECK(c.is_contiguous());
}

// ---------------------------------------------------------------------------
// Copy tests (round-trip preserves data, full + submatrix)
// ---------------------------------------------------------------------------
void test_copy_roundtrip_full() {
    gemm_y::Matrix<float, gemm_y::Space::Host> h =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(16, 16);
    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 16; ++i)
            h.view()(i, j) = static_cast<float>(i * 16 + j);

    gemm_y::Matrix<float, gemm_y::Space::Device> d =
        gemm_y::Matrix<float, gemm_y::Space::Device>::alloc(16, 16);
    gemm_y::Matrix<float, gemm_y::Space::Host> h2 =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(16, 16);

    gemm_y::copy_h2d(d.view(), h.view());
    gemm_y::copy_d2h(h2.view(), d.view());

    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 16; ++i)
            CHECK_APPROX_EQ(h2.view()(i, j), h.view()(i, j), 1e-9);
}

void test_copy_roundtrip_submatrix() {
    // 64x64 host source, copy a 16x16 submatrix (ld=64) to a 16x16 device
    // buffer (ld=16, contiguous), then back to a 64x64 host sink, and
    // verify the submatrix region matches.
    constexpr int N = 64;
    constexpr int S = 16;
    gemm_y::Matrix<float, gemm_y::Space::Host> hsrc =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(N, N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            hsrc.view()(i, j) = static_cast<float>(i * N + j);

    gemm_y::Matrix<float, gemm_y::Space::Device> dsrc =
        gemm_y::Matrix<float, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::copy_h2d(dsrc.view(), hsrc.view());

    // Device submatrix (ld=N, strided).
    auto dsub = dsrc.view().block(4, 5, S, S);

    // D2H the strided device submatrix into a contiguous host buffer (S x S,
    // ld=S). Exercises the strided cudaMemcpy2D path on the D2H side.
    gemm_y::Matrix<float, gemm_y::Space::Host> hsub =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(S, S);
    gemm_y::copy_d2h(hsub.view(), dsub);

    // Verify hsub matches the (4,5) block of hsrc.
    for (int j = 0; j < S; ++j)
        for (int i = 0; i < S; ++i)
            CHECK_APPROX_EQ(hsub.view()(i, j),
                            hsrc.view()(4 + i, 5 + j), 1e-9);

    // Now H2D the contiguous host buffer into a contiguous device buffer
    // (exercises the contiguous H2D path), then D2H back, verify round-trip.
    gemm_y::Matrix<float, gemm_y::Space::Device> ddst =
        gemm_y::Matrix<float, gemm_y::Space::Device>::alloc(S, S);
    gemm_y::copy_h2d(ddst.view(), hsub.view());
    gemm_y::Matrix<float, gemm_y::Space::Host> hsub2 =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(S, S);
    gemm_y::copy_d2h(hsub2.view(), ddst.view());
    for (int j = 0; j < S; ++j)
        for (int i = 0; i < S; ++i)
            CHECK_APPROX_EQ(hsub2.view()(i, j), hsub.view()(i, j), 1e-9);
}

// ---------------------------------------------------------------------------
// CudaTimer test
// ---------------------------------------------------------------------------
void test_cuda_timer() {
    gemm_y::CudaTimer t;
    t.start();
    // Sleep on the device via a no-op kernel + sync. We just measure that
    // elapsed_ms() returns a non-negative value after a sync.
    CUDA_CHECK(cudaDeviceSynchronize());
    t.stop();
    const float ms = t.elapsed_ms();
    CHECK(ms >= 0.0f);
}

// ---------------------------------------------------------------------------
// CublasHandle test
// ---------------------------------------------------------------------------
void test_cublas_handle() {
    gemm_y::CublasHandle h;
    CHECK(h.get() != nullptr);
}

// ---------------------------------------------------------------------------
// cublas_gemm test (small N=64 bf16 vs host fp32 reference)
// ---------------------------------------------------------------------------
void test_cublas_gemm_bf16() {
    constexpr int N = 64;
    // Host A, B in bf16; reference computed in fp32.
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);

    // Deterministic small-int fill (matches Profiler's pattern).
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            hA.view()(i, j) = static_cast<gemm_y::dtypes::bf16>(((i + j) & 7) - 3);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            hB.view()(i, j) = static_cast<gemm_y::dtypes::bf16>(((i - j) & 7) - 3);

    // Host fp32 reference: naive triple loop, accumulate in fp32.
    std::vector<float> ref(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0f);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                const float a = static_cast<float>(hA.view()(i, k));
                const float b = static_cast<float>(hB.view()(k, j));
                acc += a * b;
            }
            ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N] = acc;
        }
    }

    // Device buffers + run cuBLAS.
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);

    gemm_y::copy_h2d(dA.view(), hA.view());
    gemm_y::copy_h2d(dB.view(), hB.view());

    gemm_y::CublasHandle handle;
    gemm_y::cublas_gemm(handle, dA.view(), dB.view(), dC.view());

    gemm_y::copy_d2h(hC.view(), dC.view());

    // Compare cuBLAS output vs host fp32 reference. Tolerance 1e-3 (cuBLAS
    // is the ground truth here — see TODO 3.3).
    double max_abs = 0.0, max_rel = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const double g = static_cast<double>(hC.view()(i, j));
            const double r = static_cast<double>(
                ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N]);
            const double abs_err = std::fabs(g - r);
            const double denom = std::fmax(std::fabs(r), 1e-9);
            const double rel_err = abs_err / denom;
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
        }
    }
    std::printf("[test_cublas_gemm_bf16] max_abs=%.3e max_rel=%.3e\n", max_abs, max_rel);
    CHECK(max_rel <= 1e-3);
}

// ---------------------------------------------------------------------------
// Smoke test (preserved from the original test.cu)
// ---------------------------------------------------------------------------
__global__ void fill_kernel(int* data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

void test_smoke() {
    constexpr int kN = 256;
    std::vector<int> host(kN, 0);
    int* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, kN * sizeof(int)));
    fill_kernel<<<(kN + 63) / 64, 64>>>(device, kN, 42);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), device, kN * sizeof(int),
                          cudaMemcpyDeviceToHost));
    cudaFree(device);
    for (int i = 0; i < kN; ++i) CHECK(host[i] == 42);
}

} // namespace

int main() {
    std::printf("test_cuda: compiled for %s\n", GEMM_Y_TARGET_ARCH_NAME);

    test_buffer_host();
    test_buffer_device();
    test_matrixview_block();
    test_matrixview_is_contiguous();
    test_copy_roundtrip_full();
    test_copy_roundtrip_submatrix();
    test_cuda_timer();
    test_cublas_handle();
    test_cublas_gemm_bf16();
    test_smoke();

    std::printf("test_cuda: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
