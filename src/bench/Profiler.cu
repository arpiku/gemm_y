// Profiler.cu — implementation of Profiler<T>::run_sweep.
//
// Compiled per arch (alongside the arch-specific kernel directory). The
// sweep logic is arch-agnostic; only the kernels it dispatches to differ.
// run_sweep returns SweepResult (decoupled from CSV I/O); cuBLAS is
// measured once per N and reused as ref_* for all kernels.

#include "Profiler.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Arch.h"
#include "Accuracy.h"
#include "Copy.h"
#include "CudaTimer.h"
#include "Fill.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Stats.h"
#include "Tracer.h"
#include "dtypes.h"

namespace gemm_y {

namespace {

constexpr int kMaxN = 4096;
constexpr int kWarmup = 20;
constexpr int kTimed = 50;

} // namespace

template <typename T>
SweepResult Profiler<T>::run_sweep(const std::vector<int>& sizes) {
    SweepResult result;

    // Pre-allocate 4096x4096 device buffers: A, B, C_max, C_ref_max.
    // Four separate cudaMalloc allocations (CUDA guarantees disjoint
    // address ranges — no aliasing risk; see ARD.md §5).
    Matrix<T, Space::Device> A_max = Matrix<T, Space::Device>::alloc(kMaxN, kMaxN);
    Matrix<T, Space::Device> B_max = Matrix<T, Space::Device>::alloc(kMaxN, kMaxN);
    Matrix<T, Space::Device> C_max = Matrix<T, Space::Device>::alloc(kMaxN, kMaxN);
    Matrix<T, Space::Device> C_ref_max = Matrix<T, Space::Device>::alloc(kMaxN, kMaxN);

    // Host buffers for H2D source + D2H sink (full 4096x4096).
    Matrix<T, Space::Host> hA_max = Matrix<T, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<T, Space::Host> hB_max = Matrix<T, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<T, Space::Host> hC_max = Matrix<T, Space::Host>::alloc(kMaxN, kMaxN);
    Matrix<T, Space::Host> hC_ref_max = Matrix<T, Space::Host>::alloc(kMaxN, kMaxN);

    // Fill host A, B once with the deterministic pattern.
    bench::fill_sequential<T>(hA_max.view(), hB_max.view());

    // H2D A, B once (timed; reported in every CSV row).
    CudaTimer h2d_timer;
    h2d_timer.start();
    copy_h2d(A_max.view(), hA_max.view());
    copy_h2d(B_max.view(), hB_max.view());
    h2d_timer.stop();
    const double h2d_total_ns = static_cast<double>(h2d_timer.elapsed_ms()) * 1e6;

    std::printf("[Profiler] arch=%s dtype=%s kernels=%zu sizes=%zu  h2d(A+B)=%.2f ms\n",
                kArchName, dtypes::name<T>().data(), kernels_.size(), sizes.size(),
                h2d_total_ns / 1e6);

    tracer::Timer<> sweep_timer;
    (void)sweep_timer.mark("sweep_start");

    for (int N : sizes) {
        if (N > kMaxN) {
            std::fprintf(stderr, "Profiler: N=%d exceeds kMaxN=%d; skipping\n", N, kMaxN);
            continue;
        }

        // Sub-views (ld = kMaxN, strided for N < kMaxN).
        auto dA = A_max.view().block(0, 0, N, N);
        auto dB = B_max.view().block(0, 0, N, N);
        auto dC = C_max.view().block(0, 0, N, N);
        auto dC_ref = C_ref_max.view().block(0, 0, N, N);

        // 1) cuBLAS reference -> dC_ref. Warmup + timed. Measured ONCE per N;
        //    its kernel_min_ns / kernel_median_ns are reused as ref_* for every
        //    subsequent kernel row at this N.
        bench::TimedStats ref_stats;
        double ref_d2h_ns = 0.0;
        {
            for (int i = 0; i < kWarmup; ++i) {
                cublas_gemm(cublas_, dA, dB, dC_ref);
            }
            std::vector<float> ms; ms.reserve(kTimed);
            CudaTimer t;
            for (int i = 0; i < kTimed; ++i) {
                t.start();
                cublas_gemm(cublas_, dA, dB, dC_ref);
                t.stop();
                ms.push_back(t.elapsed_ms());
            }
            ref_stats = bench::summarize_ns(ms);

            // D2H C_ref once (timed).
            CudaTimer d2h_t;
            d2h_t.start();
            copy_d2h(hC_ref_max.view().block(0, 0, N, N), dC_ref);
            d2h_t.stop();
            ref_d2h_ns = static_cast<double>(d2h_t.elapsed_ms()) * 1e6;
        }

        // Push the cuBLAS row first per N. ref_* == kernel_* (self-referential).
        // Accuracy is 0 (ground truth vs itself).
        {
            SweepRow row;
            row.arch = kArchName;
            row.dtype = std::string(dtypes::name<T>());
            row.N = N;
            row.kernel_name = "cublas";
            row.kernel_desc = "cublasGemmEx reference (fp32 accum)";
            row.h2d_ns = h2d_total_ns;
            row.kernel_min_ns = ref_stats.min_ns;
            row.kernel_median_ns = ref_stats.median_ns;
            row.d2h_ns = ref_d2h_ns;
            row.ref_kernel_min_ns = ref_stats.min_ns;
            row.ref_kernel_median_ns = ref_stats.median_ns;
            row.max_abs_err = 0.0;
            row.max_rel_err = 0.0;
            result.rows.push_back(std::move(row));
        }

        // 2) Per registered kernel: warmup + timed + accuracy vs C_ref.
        for (const auto& k : kernels_) {
            GemmArgs<T> args{dA, dB, dC};

            // Debug-build OOB check: snapshot C_ref's N x N block on the
            // host before the custom kernel runs; verify unchanged after.
            // Catches out-of-bounds writes that would otherwise silently
            // corrupt the reference (see ARD.md §5.1).
            std::vector<T> cref_snapshot;
        #ifndef NDEBUG
            {
                cref_snapshot.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), T{0});
                for (int j = 0; j < N; ++j) {
                    for (int i = 0; i < N; ++i) {
                        cref_snapshot[static_cast<std::size_t>(i) +
                                      static_cast<std::size_t>(j) * static_cast<std::size_t>(N)] =
                            hC_ref_max.view()(i, j);
                    }
                }
            }
        #endif

            // Warmup (untimed).
            for (int i = 0; i < kWarmup; ++i) {
                k.run(args, nullptr);
            }
            CUDA_CHECK_LAST_ERROR();

            // Timed.
            std::vector<float> ms; ms.reserve(kTimed);
            CudaTimer t;
            for (int i = 0; i < kTimed; ++i) {
                t.start();
                k.run(args, nullptr);
                t.stop();
                ms.push_back(t.elapsed_ms());
            }
            CUDA_CHECK_LAST_ERROR();
            const bench::TimedStats s = bench::summarize_ns(ms);

            // D2H C once (timed).
            CudaTimer d2h_t;
            d2h_t.start();
            copy_d2h(hC_max.view().block(0, 0, N, N), dC);
            d2h_t.stop();
            const double d2h_ns = static_cast<double>(d2h_t.elapsed_ms()) * 1e6;

            // Accuracy vs C_ref (host-side, fp64).
            auto hgot = hC_max.view().block(0, 0, N, N);
            auto href = hC_ref_max.view().block(0, 0, N, N);
            const ErrReport<T> err = compare<T>(hgot, href);

        #ifndef NDEBUG
            {
                // Verify C_ref on device wasn't corrupted by the custom kernel.
                std::vector<T> now(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), T{0});
                copy_d2h(hC_ref_max.view().block(0, 0, N, N), dC_ref);
                for (int j = 0; j < N; ++j) {
                    for (int i = 0; i < N; ++i) {
                        now[static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(j) * static_cast<std::size_t>(N)] =
                            hC_ref_max.view()(i, j);
                    }
                }
                if (std::memcmp(now.data(), cref_snapshot.data(),
                                now.size() * sizeof(T)) != 0) {
                    std::fprintf(stderr,
                                 "[OOB] kernel '%s' (N=%d) corrupted C_ref_max!\n",
                                 k.name.c_str(), N);
                    std::abort();
                }
            }
        #endif

            // Push the kernel row. ref_* reuse the cuBLAS row's kernel_*.
            {
                SweepRow row;
                row.arch = kArchName;
                row.dtype = std::string(dtypes::name<T>());
                row.N = N;
                row.kernel_name = k.name;
                row.kernel_desc = k.description;
                row.h2d_ns = h2d_total_ns;
                row.kernel_min_ns = s.min_ns;
                row.kernel_median_ns = s.median_ns;
                row.d2h_ns = d2h_ns;
                row.ref_kernel_min_ns = ref_stats.min_ns;
                row.ref_kernel_median_ns = ref_stats.median_ns;
                row.max_abs_err = err.max_abs;
                row.max_rel_err = err.max_rel;
                result.rows.push_back(std::move(row));
            }

            const char* verdict = (err.max_rel <= kRelErrTol) ? "PASS" : "FAIL";
            std::printf("[Profiler] N=%4d  %-12s  kernel=%.1f us  ref=%.1f us  "
                        "rel_err=%.3e  %s\n",
                        N, k.name.c_str(),
                        s.median_ns / 1e3, ref_stats.median_ns / 1e3,
                        err.max_rel, verdict);
        }
    }

    (void)sweep_timer.mark("sweep_end");
    // Print total sweep wall time (host-side steady_clock; includes launch
    // overhead — for orchestration context only, not kernel timing).
    // Indices: [0]=origin (ctor), [1]=sweep_start, [2]=sweep_end.
    {
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            sweep_timer[2].timestamp - sweep_timer[1].timestamp);
        std::printf("[Profiler] sweep wall time: %ld ms\n", static_cast<long>(dt.count()));
    }

    return result;
}

// Explicit instantiations for the dtypes we support. bf16 is wired now;
// fp16/fp32 are listed so the harness compiles when their kernels land.
template class Profiler<__nv_bfloat16>;
template class Profiler<__half>;
template class Profiler<float>;

} // namespace gemm_y
