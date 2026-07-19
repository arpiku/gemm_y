// tests/test.cu — unit tests for Buffer / MatrixView / Matrix / CudaTimer /
// CublasHandle / cublas_gemm / Copy / NaiveGemm / Profiler.
//
// No external test framework (AGENTS.md spirit: no external deps). A tiny
// hand-rolled assert macro prints failures and counts them. Returns nonzero
// from main() if any check failed.
//
// Phase 1.5 (R7): removed test_smoke (RAII-violating, redundant — the
//   round-trip it tested is covered by Copy tests). Trimmed test_buffer_device
//   to Buffer invariants only (round-trip covered by Copy tests).
// Phase 1.5 (R8): strengthened test_cuda_timer; added
//   test_matrixview_const_conversion, test_matrix_view_from_matrix,
//   test_cublas_gemm_bf16_strided, test_naive_gemm_bf16,
//   test_profiler_run_sweep_small.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Arch.h"
#include "bench/Accuracy.h"
#include "Buffer.h"
#include "Copy.h"
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
    #include "sm90/gemm_bf16_naive.cuh"
#elif defined(CUDA_ARCH_SM_120)
    #include "sm120/gemm_bf16_naive.cuh"
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
void test_matrixview_block() {
    gemm_y::Buffer<float, gemm_y::Space::Host> b(64);
    for (int i = 0; i < 64; ++i) b.data()[i] = static_cast<float>(i);
    gemm_y::MatrixView<float, gemm_y::Space::Host> v(b.data(), 8, 8, 8);
    CHECK(v.rows == 8);
    CHECK(v.cols == 8);
    CHECK(v.ld == 8);
    CHECK(v.is_contiguous());

    auto sub = v.block(2, 3, 3, 4);
    CHECK(sub.rows == 3);
    CHECK(sub.cols == 4);
    CHECK(sub.ld == 8);
    CHECK(!sub.is_contiguous());
    CHECK(sub.ptr == v.ptr + 2 + 3 * 8);
    CHECK_APPROX_EQ(sub(0, 0), 26.0f, 1e-9);
    CHECK_APPROX_EQ(sub(1, 2), 43.0f, 1e-9);
}

void test_matrixview_is_contiguous() {
    gemm_y::Buffer<float, gemm_y::Space::Host> b(64);
    gemm_y::MatrixView<float, gemm_y::Space::Host> full(b.data(), 8, 8, 8);
    CHECK(full.is_contiguous());

    auto sub = full.block(0, 0, 4, 4);
    CHECK(!sub.is_contiguous());

    gemm_y::MatrixView<float, gemm_y::Space::Host> c(b.data(), 4, 4, 4);
    CHECK(c.is_contiguous());
}

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

    auto dsub = dsrc.view().block(4, 5, S, S);

    // D2H the strided device submatrix into a contiguous host buffer.
    gemm_y::Matrix<float, gemm_y::Space::Host> hsub =
        gemm_y::Matrix<float, gemm_y::Space::Host>::alloc(S, S);
    gemm_y::copy_d2h(hsub.view(), dsub);

    for (int j = 0; j < S; ++j)
        for (int i = 0; i < S; ++i)
            CHECK_APPROX_EQ(hsub.view()(i, j),
                            hsrc.view()(4 + i, 5 + j), 1e-9);

    // H2D the contiguous host buffer into a contiguous device buffer, then
    // D2H back, verify round-trip.
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
// Compile-time guarantees for detail::copy_kind_v (Phase 1.6.11)
// ---------------------------------------------------------------------------
// Wrong-direction instantiations (Host->Host, Device->Device) are rejected
// by the static_assert inside detail::copy; the poison primary template
// of copy_kind_v is the defensive secondary net. These static_asserts pin
// the valid specializations so a future refactor that flips the mapping
// fails to compile here.
void test_copy_kind_compile_time() {
    using gemm_y::Space;
    using gemm_y::detail::copy_kind_v;

    static_assert(copy_kind_v<Space::Device, Space::Host> == cudaMemcpyHostToDevice,
                  "H2D must map to cudaMemcpyHostToDevice");
    static_assert(copy_kind_v<Space::Host, Space::Device> == cudaMemcpyDeviceToHost,
                  "D2H must map to cudaMemcpyDeviceToHost");

    // Sanity: the kind enum values are distinct, so the table is not
    // accidentally uniform.
    static_assert(copy_kind_v<Space::Device, Space::Host> !=
                  copy_kind_v<Space::Host, Space::Device>,
                  "H2D and D2H kinds must differ");

    ++g_checks;  // count the test as one exercised check
}

// ---------------------------------------------------------------------------
// CudaTimer test (R8: strengthened — assert 0 < ms < 100 after empty kernel)
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

    gemm_y::copy_h2d(dA.view(), hA.view());
    gemm_y::copy_h2d(dB.view(), hB.view());

    gemm_y::CublasHandle handle;
    gemm_y::cublas_gemm(handle, dA.view(), dB.view(), dC.view());

    gemm_y::copy_d2h(hC.view(), dC.view());

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

// R8: cuBLAS on a strided submatrix (N=32 of 64x64, ld=64). Catches the
// bench runner's exact dependency — ld-aware cuBLAS calls.
void test_cublas_gemm_bf16_strided() {
    constexpr int N = 32;
    constexpr int BIG = 64;
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hA =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(BIG, BIG);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(BIG, BIG);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(BIG, BIG);

    gemm_y::bench::fill_sequential<gemm_y::dtypes::bf16>(hA.view(), hB.view());

    // Host fp32 reference for the N x N top-left block.
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
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(BIG, BIG);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dB =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(BIG, BIG);
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device> dC =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Device>::alloc(BIG, BIG);

    gemm_y::copy_h2d(dA.view(), hA.view());
    gemm_y::copy_h2d(dB.view(), hB.view());

    // Strided sub-views: N x N with ld = BIG.
    auto dA_sub = dA.view().block(0, 0, N, N);
    auto dB_sub = dB.view().block(0, 0, N, N);
    auto dC_sub = dC.view().block(0, 0, N, N);

    gemm_y::CublasHandle handle;
    gemm_y::cublas_gemm(handle, dA_sub, dB_sub, dC_sub);

    // D2H the strided C submatrix into a contiguous host buffer.
    gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host> hC_sub =
        gemm_y::Matrix<gemm_y::dtypes::bf16, gemm_y::Space::Host>::alloc(N, N);
    gemm_y::copy_d2h(hC_sub.view(), dC_sub);

    double max_rel = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            const double g = static_cast<double>(hC_sub.view()(i, j));
            const double r = static_cast<double>(
                ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N]);
            const double abs_err = std::fabs(g - r);
            const double denom = std::fmax(std::fabs(r), 1e-9);
            const double rel_err = abs_err / denom;
            if (rel_err > max_rel) max_rel = rel_err;
        }
    }
    std::printf("[test_cublas_gemm_bf16_strided] max_rel=%.3e\n", max_rel);
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

    gemm_y::copy_h2d(dA.view(), hA.view());
    gemm_y::copy_h2d(dB.view(), hB.view());

    gemm_y::NaiveGemm<gemm_y::dtypes::bf16> kernel;
    kernel(gemm_y::GemmArgs<gemm_y::dtypes::bf16>{dA.view(), dB.view(), dC.view()}, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    gemm_y::copy_d2h(hC.view(), dC.view());

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
        CHECK(r.max_rel_err <= gemm_y::kRelErrTol);
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
    test_matrixview_block();
    test_matrixview_is_contiguous();
    test_matrixview_const_conversion();
    test_matrix_view_from_matrix();
    test_copy_roundtrip_full();
    test_copy_roundtrip_submatrix();
    test_copy_kind_compile_time();
    test_cuda_timer();
    test_cublas_handle();
    test_cublas_gemm_bf16();
    test_cublas_gemm_bf16_strided();
    test_naive_gemm_bf16();
    test_profiler_run_sweep_small();

    std::printf("test_cuda: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
