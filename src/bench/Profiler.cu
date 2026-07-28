// Profiler.cu — implementation of Profiler<T>::run_sweep.
//
// Compiled per arch (alongside the arch-specific kernel directory). The
// sweep logic is arch-agnostic; only the kernels it dispatches to differ.
// run_sweep returns SweepResult (decoupled from CSV I/O); cuBLAS is
// measured once per N and reused as ref_* for all kernels.
//
// Per-N allocation (ARD §5 revised): A/B/C/C_ref (device + host) are
// allocated N×N inside the per-N loop. ld == N for every kernel launch
// (was ld == 4096 under the pre-alloc strategy). H2D A+B is timed per N
// and reported in the h2d_ns column for every row at that N. cudaMemcpy
// is called directly (no Copy.h wrapper); buffers are contiguous.

#include "Profiler.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Arch.h"
#include "Accuracy.h"
#include "CudaCheck.h"
#include "CudaTimer.h"
#include "Fill.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Stats.h"
#include "dtypes.h"

namespace gemm_y {

namespace {

constexpr int kWarmup = 20;
constexpr int kTimed = 50;

} // namespace

template <typename T>
SweepResult Profiler<T>::run_sweep(const std::vector<int>& sizes) {
    SweepResult result;

    std::printf("[Profiler] arch=%s dtype=%s kernels=%zu sizes=%zu  "
                "h2d=per-N (in CSV)\n",
                kArchName, dtypes::name<T>().data(), kernels_.size(), sizes.size());

    const auto sweep_start = std::chrono::steady_clock::now();

    for (int N : sizes) {
        // Per-N allocation: A, B, C, C_ref on device + host. RAII —
        // constructed and destroyed per iteration, no manual cudaFree.
        // ld == N (ColMajor default in Matrix::alloc).
        Matrix<T, Space::Device> dA = Matrix<T, Space::Device>::alloc(N, N);
        Matrix<T, Space::Device> dB = Matrix<T, Space::Device>::alloc(N, N);
        Matrix<T, Space::Device> dC = Matrix<T, Space::Device>::alloc(N, N);
        Matrix<T, Space::Device> dC_ref = Matrix<T, Space::Device>::alloc(N, N);

        Matrix<T, Space::Host> hA = Matrix<T, Space::Host>::alloc(N, N);
        Matrix<T, Space::Host> hB = Matrix<T, Space::Host>::alloc(N, N);
        Matrix<T, Space::Host> hC = Matrix<T, Space::Host>::alloc(N, N);
        Matrix<T, Space::Host> hC_ref = Matrix<T, Space::Host>::alloc(N, N);

        // Fill host A, B with the deterministic pattern (N×N directly).
        bench::fill_sequential<T>(hA.view(), hB.view());

        // H2D A+B (timed per N; reported in every CSV row at this N).
        CudaTimer h2d_timer;
        h2d_timer.start();
        CUDA_CHECK(cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));
        h2d_timer.stop();
        const double h2d_ns = static_cast<double>(h2d_timer.elapsed_ms()) * 1e6;

        auto dA_v = dA.view();
        auto dB_v = dB.view();
        auto dC_v = dC.view();
        auto dC_ref_v = dC_ref.view();

        // 1) cuBLAS reference -> dC_ref. Warmup + timed. Measured ONCE per N;
        //    its kernel_min_ns / kernel_median_ns are reused as ref_* for every
        //    subsequent kernel row at this N.
        bench::TimedStats ref_stats;
        double ref_d2h_ns = 0.0;
        {
            for (int i = 0; i < kWarmup; ++i) {
                cublas_gemm(cublas_, dA_v, dB_v, dC_ref_v);
            }
            std::vector<float> ms; ms.reserve(kTimed);
            CudaTimer t;
            for (int i = 0; i < kTimed; ++i) {
                t.start();
                cublas_gemm(cublas_, dA_v, dB_v, dC_ref_v);
                t.stop();
                ms.push_back(t.elapsed_ms());
            }
            ref_stats = bench::summarize_ns(ms);

            // D2H C_ref once (timed).
            CudaTimer d2h_t;
            d2h_t.start();
            CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref.data(), dC_ref.bytes(),
                                  cudaMemcpyDeviceToHost));
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
            row.h2d_ns = h2d_ns;
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
            GemmArgs<T> args{dA_v, dB_v, dC_v};

            // Debug-build OOB check: snapshot C_ref's N×N buffer on the host
            // before the custom kernel runs; verify unchanged after. Catches
            // out-of-bounds writes that would otherwise silently corrupt the
            // reference (see ARD.md §5.1). Contiguous cudaMemcpy + memcmp.
            std::vector<T> cref_snapshot;
        #ifndef NDEBUG
            {
                cref_snapshot.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), T{0});
                CUDA_CHECK(cudaMemcpy(cref_snapshot.data(), dC_ref.data(),
                                      dC_ref.bytes(), cudaMemcpyDeviceToHost));
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
            CUDA_CHECK(cudaMemcpy(hC.data(), dC.data(), dC.bytes(),
                                  cudaMemcpyDeviceToHost));
            d2h_t.stop();
            const double d2h_ns = static_cast<double>(d2h_t.elapsed_ms()) * 1e6;

            // Accuracy vs C_ref (host-side, fp64).
            const ErrReport<T> err = compare<T>(hC.view(), hC_ref.view());

        #ifndef NDEBUG
            {
                // Verify C_ref on device wasn't corrupted by the custom kernel.
                std::vector<T> now(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), T{0});
                CUDA_CHECK(cudaMemcpy(now.data(), dC_ref.data(), dC_ref.bytes(),
                                      cudaMemcpyDeviceToHost));
                if (std::memcmp(now.data(), cref_snapshot.data(),
                                now.size() * sizeof(T)) != 0) {
                    std::fprintf(stderr,
                                 "[OOB] kernel '%s' (N=%d) corrupted C_ref!\n",
                                 k.name.c_str(), N);
                    std::abort();
                }
            }
        #endif

            // Skip-on-fail: timing of mathematically invalid kernels is
            // meaningless (see ARD.md §6). Print stderr FAIL and continue
            // without writing the row. cuBLAS reference rows are always
            // written (they're pushed before this loop, err == 0).
            constexpr double tol = kRelErrTol<T>();
            if (err.max_rel > tol) {
                std::fprintf(stderr,
                             "[Profiler] FAIL N=%4d %-12s rel_err=%.3e tol=%.3e "
                             "(skipping row)\n",
                             N, k.name.c_str(), err.max_rel, tol);
                continue;
            }

            // Push the kernel row. ref_* reuse the cuBLAS row's kernel_*.
            {
                SweepRow row;
                row.arch = kArchName;
                row.dtype = std::string(dtypes::name<T>());
                row.N = N;
                row.kernel_name = k.name;
                row.kernel_desc = k.description;
                row.h2d_ns = h2d_ns;
                row.kernel_min_ns = s.min_ns;
                row.kernel_median_ns = s.median_ns;
                row.d2h_ns = d2h_ns;
                row.ref_kernel_min_ns = ref_stats.min_ns;
                row.ref_kernel_median_ns = ref_stats.median_ns;
                row.max_abs_err = err.max_abs;
                row.max_rel_err = err.max_rel;
                result.rows.push_back(std::move(row));
            }

            std::printf("[Profiler] N=%4d  %-12s  kernel=%.1f us  ref=%.1f us  "
                        "rel_err=%.3e  PASS\n",
                        N, k.name.c_str(),
                        s.median_ns / 1e3, ref_stats.median_ns / 1e3,
                        err.max_rel);
        }
    }

    // Total sweep wall time (host-side steady_clock; includes launch
    // overhead — for orchestration context only, not kernel timing).
    const auto sweep_end = std::chrono::steady_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
        sweep_end - sweep_start);
    std::printf("[Profiler] sweep wall time: %ld ms\n", static_cast<long>(dt.count()));

    return result;
}

// Explicit instantiations for the dtypes we support. bf16 is wired now;
// fp16/tfloat are listed so the harness compiles when their kernels land.
template class Profiler<__nv_bfloat16>;
template class Profiler<__half>;
template class Profiler<float>;

} // namespace gemm_y
