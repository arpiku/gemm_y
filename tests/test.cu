// tests/test.cu — unit tests for Buffer / MatrixView / Matrix / CudaTimer /
// CublasHandle / cublas_gemm / NaiveGemm / Profiler.
//
// No external test framework (AGENTS.md spirit: no external deps). A tiny
// hand-rolled assert macro prints failures and counts them. Returns nonzero
// from main() if any check failed.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Arch.h"
#include "bench/Accuracy.h"
#include "Buffer.h"
#include "CudaCheck.h"
#include "CudaTimer.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Space.h"
#include "bench/Fill.h"
#include "bench/Profiler.h"
#include "dtypes.h"
#include "cuda_compat.h"
#include "cublas/CublasHandle.h"
#include "cublas/cublas_gemm.h"

#if defined(CUDA_ARCH_SM_90)
    #include "sm90/gemm_naive.cuh"
#elif defined(CUDA_ARCH_SM_120)
    #include "sm120/gemm_naive.cuh"
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

// R7: trimmed to invariants only. Round-trip is covered by Copy tests.
void test_buffer_device() {
    gemm_y::Buffer<float, gemm_y::Space::Device> b(64);
    CHECK(b.size() == 64);
    CHECK(b.bytes() == 64 * sizeof(float));
    CHECK(b.data() != nullptr);
}

// ---------------------------------------------------------------------------
// MatrixView tests
// ---------------------------------------------------------------------------
// R8: T -> const T compiles. The reverse (const T -> T) is a compile error
// by design (the SFINAE constraint in the converting constructor). We
// document this via a comment — uncommenting the reverse line should fail
// to compile.
void test_matrixview_const_conversion() {
    gemm_y::Buffer<float, gemm_y::Space::Host> b(16);
    gemm_y::MatrixView<float, gemm_y::Space::Host> mut(b.data(), 4, 4, 4);
    gemm_y::MatrixView<const float, gemm_y::Space::Host> ro = mut;  // T -> const T: OK
    CHECK(ro.ptr == mut.ptr);
    CHECK(ro.rows == mut.rows);
    CHECK(ro.ld == mut.ld);

    // Reverse (const T -> T) is a compile error — the SFINAE constraint
    // in MatrixView's converting constructor only allows adding const,
    // never stripping it. Uncomment to verify:
    //   gemm_y::MatrixView<float, gemm_y::Space::Host> bad = ro;  // ERROR
}

// R8: non-const Matrix::view() returns a mutable view; const Matrix::view()
// returns a read-only view.
void test_matrix_view_from_matrix() {
    gemm_y::Matrix<float, gemm_y::Space::Host> m =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(4, 4);

    // Non-const: mutable view.
    gemm_y::MatrixView<float, gemm_y::Space::Host> mv = m.view();
    mv(0, 0) = 42.0f;
    CHECK_APPROX_EQ(m.view()(0, 0), 42.0f, 1e-9);

    // Const: read-only view.
    const gemm_y::Matrix<float, gemm_y::Space::Host>& cm = m;
    gemm_y::MatrixView<const float, gemm_y::Space::Host> ro = cm.view();
    CHECK_APPROX_EQ(ro(0, 0), 42.0f, 1e-9);
    // ro(0, 0) = 1.0f;  // would be a compile error (assignment to const T&)
}

// ---------------------------------------------------------------------------
// Copy tests (round-trip preserves data, contiguous cudaMemcpy)
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

    CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h2.data(), d.data(), d.bytes(), cudaMemcpyDeviceToHost));

    for (int j = 0; j < 16; ++j)
        for (int i = 0; i < 16; ++i)
            CHECK_APPROX_EQ(h2.view()(i, j), h.view()(i, j), 1e-9);
}

// ---------------------------------------------------------------------------
// CudaTimer test — assert 0 < ms < 100 after empty kernel
// ---------------------------------------------------------------------------
void test_cuda_timer() {
    gemm_y::CudaTimer t;
    t.start();
    CUDA_CHECK(cudaDeviceSynchronize());
    t.stop();
    const float ms = t.elapsed_ms();
    CHECK(ms > 0.0f);
    // An empty sync should take well under 100 ms; if it's more, something
    // is wrong with the timer or the device is hung.
    CHECK(ms < 100.0f);
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
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);

    gemm_y::bench::fill_sequential<gemm_y::dtypes::bf16>(hA.view(), hB.view());

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

    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);

    CUDA_CHECK(cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));

    gemm_y::CublasHandle handle;
    gemm_y::cublas_gemm(handle, dA.view(), dB.view(), dC.view());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC.data(), dC.bytes(), cudaMemcpyDeviceToHost));

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

// R8: NaiveGemm<bf16> on N=32 vs host fp32 reference, max_rel_err <= 1e-3.
void test_naive_gemm_bf16() {
    constexpr int N = 32;
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);

    gemm_y::bench::fill_sequential<gemm_y::dtypes::bf16>(hA.view(), hB.view());

    // Host fp32 reference.
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

    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(N, N);

    CUDA_CHECK(cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));

    gemm_y::NaiveGemm<gemm_y::dtypes::bf16> kernel;
    kernel(gemm_y::GemmArgs<gemm_y::dtypes::bf16>{dA.view(), dB.view(), dC.view()}, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC.data(), dC.bytes(), cudaMemcpyDeviceToHost));

    double max_rel = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const double g = static_cast<double>(hC.view()(i, j));
            const double r = static_cast<double>(
                ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N]);
            const double abs_err = std::fabs(g - r);
            const double denom = std::fmax(std::fabs(r), 1e-9);
            const double rel_err = abs_err / denom;
            if (rel_err > max_rel) max_rel = rel_err;
        }
    }
    std::printf("[test_naive_gemm_bf16] max_rel=%.3e\n", max_rel);
    CHECK(max_rel <= 1e-3);
}

// fp16 cuBLAS vs host fp64 reference. fp16 has a 10-bit mantissa; cuBLAS
// fp16 (tensor cores, fp32 accum) should agree with fp64 to ~1e-4.
void test_cublas_gemm_fp16() {
    using T = gemm_y::dtypes::fp16;
    constexpr int N = 64;
    gemm_y::Matrix<T, gemm_y::Space::Host> hA = gemm_y::Matrix<T, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Host> hB = gemm_y::Matrix<T, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Host> hC = gemm_y::Matrix<T, gemm_y::Space::Host>::alloc(N, N);

    gemm_y::bench::fill_sequential<T>(hA.view(), hB.view());

    // Host fp64 reference: naive triple loop, accumulate in double.
    std::vector<double> ref(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) {
                const double a = static_cast<double>(hA.view()(i, k));
                const double b = static_cast<double>(hB.view()(k, j));
                acc += a * b;
            }
            ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N] = acc;
        }
    }

    gemm_y::Matrix<T, gemm_y::Space::Device> dA = gemm_y::Matrix<T, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Device> dB = gemm_y::Matrix<T, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Device> dC = gemm_y::Matrix<T, gemm_y::Space::Device>::alloc(N, N);

    CUDA_CHECK(cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));

    gemm_y::CublasHandle handle;
    gemm_y::cublas_gemm(handle, dA.view(), dB.view(), dC.view());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC.data(), dC.bytes(), cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const double g = static_cast<double>(hC.view()(i, j));
            const double r = ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N];
            const double abs_err = std::fabs(g - r);
            const double denom = std::fmax(std::fabs(r), 1e-9);
            const double rel_err = abs_err / denom;
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
        }
    }
    std::printf("[test_cublas_gemm_fp16] max_abs=%.3e max_rel=%.3e\n", max_abs, max_rel);
    CHECK(max_rel <= gemm_y::kRelErrTol<T>());
}

// tfloat (tf32) cuBLAS vs host fp64 reference. tf32 truncates the fp32
// mantissa to 10 bits; cuBLAS tf32 (tensor cores, fp32 accum) should agree
// with fp64 to ~1e-4. Also verifies the math mode is restored after the
// call (CublasMathModeGuard must restore CUBLAS_DEFAULT_MATH).
void test_cublas_gemm_tfloat() {
    using T = gemm_y::dtypes::tfloat;  // tfloat = tf32 path (TC), not pedantic fp32 (CUDA cores).
    constexpr int N = 64;
    gemm_y::Matrix<T, gemm_y::Space::Host> hA = gemm_y::Matrix<T, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Host> hB = gemm_y::Matrix<T, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Host> hC = gemm_y::Matrix<T, gemm_y::Space::Host>::alloc(N, N);

    gemm_y::bench::fill_sequential<T>(hA.view(), hB.view());

    // Host fp64 reference: naive triple loop, accumulate in double.
    std::vector<double> ref(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) {
                const double a = static_cast<double>(hA.view()(i, k));
                const double b = static_cast<double>(hB.view()(k, j));
                acc += a * b;
            }
            ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N] = acc;
        }
    }

    gemm_y::Matrix<T, gemm_y::Space::Device> dA = gemm_y::Matrix<T, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Device> dB = gemm_y::Matrix<T, gemm_y::Space::Device>::alloc(N, N);
    gemm_y::Matrix<T, gemm_y::Space::Device> dC = gemm_y::Matrix<T, gemm_y::Space::Device>::alloc(N, N);

    CUDA_CHECK(cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));

    gemm_y::CublasHandle handle;

    // Math-mode restore: capture before, run, capture after, assert unchanged.
    cublasMath_t mode_before = CUBLAS_DEFAULT_MATH;
    CUBLAS_CHECK(cublasGetMathMode(handle.get(), &mode_before));
    gemm_y::cublas_gemm(handle, dA.view(), dB.view(), dC.view());
    cublasMath_t mode_after = CUBLAS_DEFAULT_MATH;
    CUBLAS_CHECK(cublasGetMathMode(handle.get(), &mode_after));
    CHECK(mode_after == mode_before);
    CHECK(mode_after == CUBLAS_DEFAULT_MATH);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC.data(), dC.bytes(), cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const double g = static_cast<double>(hC.view()(i, j));
            const double r = ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N];
            const double abs_err = std::fabs(g - r);
            const double denom = std::fmax(std::fabs(r), 1e-9);
            const double rel_err = abs_err / denom;
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
        }
    }
    std::printf("[test_cublas_gemm_tfloat] max_abs=%.3e max_rel=%.3e\n", max_abs, max_rel);
    CHECK(max_rel <= gemm_y::kRelErrTol<T>());
}

// R8: small Profiler sweep. Register NaiveGemm<bf16>, sweep {32, 64},
// assert 4 rows (2 sizes x 2 kernels: cuBLAS + naive), all max_rel_err
// <= kRelErrTol. Exercises the R3 decoupling (run_sweep returns a result,
// no CSV written).
void test_profiler_run_sweep_small() {
    gemm_y::Profiler<gemm_y::dtypes::bf16> prof;
    prof.register_kernel<gemm_y::NaiveGemm<gemm_y::dtypes::bf16>>();
    const std::vector<int> sizes = {32, 64};
    const gemm_y::SweepResult result = prof.run_sweep(sizes);

    // 2 sizes x (1 cuBLAS + 1 naive) = 4 rows.
    CHECK(result.rows.size() == 4);

    // All accuracy checks must pass.
    for (const auto& r : result.rows) {
        CHECK(r.max_rel_err <= gemm_y::kRelErrTol<gemm_y::dtypes::bf16>());
    }

    // First row per N should be cuBLAS; second should be naive.
    CHECK(result.rows[0].kernel_name == "cublas");
    CHECK(result.rows[1].kernel_name == "naive");
    CHECK(result.rows[2].kernel_name == "cublas");
    CHECK(result.rows[3].kernel_name == "naive");

    // ref_* on the naive row should equal kernel_* on the cuBLAS row at the same N.
    CHECK_APPROX_EQ(result.rows[1].ref_kernel_median_ns,
                    result.rows[0].kernel_median_ns, 1e-9);
    CHECK_APPROX_EQ(result.rows[3].ref_kernel_median_ns,
                    result.rows[2].kernel_median_ns, 1e-9);
}

} // namespace

int main() {
    std::printf("test_cuda: compiled for %s\n", gemm_y::kArchName);

    test_buffer_host();
    test_buffer_device();
    test_matrixview_const_conversion();
    test_matrix_view_from_matrix();
    test_copy_roundtrip_full();
    test_cuda_timer();
    test_cublas_handle();
    test_cublas_gemm_bf16();
    test_cublas_gemm_fp16();
    test_cublas_gemm_tfloat();
    test_naive_gemm_bf16();
    test_profiler_run_sweep_small();

    std::printf("test_cuda: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
