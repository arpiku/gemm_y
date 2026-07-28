// Profiler.cu — implementation of Profiler<T>::run_sweep.
//
// Compiled per arch (alongside the arch-specific kernel directory). The
// sweep logic is arch-agnostic; only the kernels it dispatches to differ.
// run_sweep returns SweepResult (decoupled from CSV I/O); cuBLAS is
// measured once per N and reused as ref_* for all kernels.
//
// Per-N allocation (ARD §5 revised): A/B/C/C_ref (device + host) are
// allocated N×N inside the per-N loop. ld == N for every kernel launch.
// cudaMemcpy is called directly (no Copy.h wrapper); buffers are contiguous.
//
// Measurement (ARD §7, §19): all kernels and cuBLAS run on the owned
// stream_ (one stream per Profiler). A cudaStreamSynchronize after each
// warmup loop ensures the GPU is idle before the timed loop begins. The
// timer_ member is reused across the entire sweep (one event pair). h2d_ns
// and d2h_ns are not reported — harness overhead is not kernel performance.

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

// Serialize the kTimed elapsed-ms samples (converted to ns) as a
// semicolon-separated string for the CSV `kernel_samples_ns` column.
// Parsed by ingest.py into the `samples` table (ARD §20).
std::string serialize_samples_ns(const std::vector<float>& ms) {
    std::string out;
    out.reserve(ms.size() * 12);  // ~12 chars per "1234567.89;"
    for (std::size_t i = 0; i < ms.size(); ++i) {
        if (i != 0) out.push_back(';');
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", static_cast<double>(ms[i]) * 1e6);
        out += buf;
    }
    return out;
}

} // namespace

template <typename T>
SweepResult Profiler<T>::run_sweep(const std::vector<int>& sizes) {
    SweepResult result;

    std::printf("[Profiler] arch=%s dtype=%s kernels=%zu sizes=%zu  "
                "warmup=%d timed=%d\n",
                kArchName, dtypes::name<T>().data(), kernels_.size(), sizes.size(),
                kWarmup, kTimed);

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

        // H2D A+B (untimed; harness overhead, not kernel performance).
        CUDA_CHECK(cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));

        auto dA_v = dA.view();
        auto dB_v = dB.view();
        auto dC_v = dC.view();
        auto dC_ref_v = dC_ref.view();

        // 1) cuBLAS reference -> dC_ref. Warmup + timed. Measured ONCE per N;
        //    its stats are reused as ref_* for every subsequent kernel row at this N.
        bench::TimedStats ref_stats;
        std::string ref_samples_ns;
        {
            for (int i = 0; i < kWarmup; ++i) {
                cublas_gemm(cublas_, dA_v, dB_v, dC_ref_v, stream_);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            std::vector<float> ms; ms.reserve(kTimed);
            for (int i = 0; i < kTimed; ++i) {
                timer_.start(stream_);
                cublas_gemm(cublas_, dA_v, dB_v, dC_ref_v, stream_);
                timer_.stop(stream_);
                ms.push_back(timer_.elapsed_ms());
            }
            ref_stats = bench::summarize_ns(ms);
            ref_samples_ns = serialize_samples_ns(ms);

            // D2H C_ref once (untimed).
            CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref.data(), dC_ref.bytes(),
                                  cudaMemcpyDeviceToHost));
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
            row.kernel_min_ns = ref_stats.min_ns;
            row.kernel_median_ns = ref_stats.median_ns;
            row.kernel_std_ns = ref_stats.std_ns;
            row.kernel_p95_ns = ref_stats.p95_ns;
            row.kernel_ci_low_ns = ref_stats.ci_low_ns;
            row.kernel_ci_high_ns = ref_stats.ci_high_ns;
            row.ref_kernel_min_ns = ref_stats.min_ns;
            row.ref_kernel_median_ns = ref_stats.median_ns;
            row.ref_kernel_std_ns = ref_stats.std_ns;
            row.ref_kernel_p95_ns = ref_stats.p95_ns;
            row.ref_kernel_ci_low_ns = ref_stats.ci_low_ns;
            row.ref_kernel_ci_high_ns = ref_stats.ci_high_ns;
            row.max_abs_err = 0.0;
            row.max_rel_err = 0.0;
            row.kernel_samples_ns = ref_samples_ns;
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

            // Warmup (untimed). Sync after warmup so the timed loop starts
            // from an idle GPU (no warmup launches still in flight).
            for (int i = 0; i < kWarmup; ++i) {
                k.run(args, stream_);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            CUDA_CHECK_LAST_ERROR();

            // Timed.
            std::vector<float> ms; ms.reserve(kTimed);
            for (int i = 0; i < kTimed; ++i) {
                timer_.start(stream_);
                k.run(args, stream_);
                timer_.stop(stream_);
                ms.push_back(timer_.elapsed_ms());
            }
            CUDA_CHECK_LAST_ERROR();
            const bench::TimedStats s = bench::summarize_ns(ms);
            const std::string samples_ns = serialize_samples_ns(ms);

            // D2H C once (untimed).
            CUDA_CHECK(cudaMemcpy(hC.data(), dC.data(), dC.bytes(),
                                  cudaMemcpyDeviceToHost));

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

            // Push the kernel row. ref_* reuse the cuBLAS row's stats.
            {
                SweepRow row;
                row.arch = kArchName;
                row.dtype = std::string(dtypes::name<T>());
                row.N = N;
                row.kernel_name = k.name;
                row.kernel_desc = k.description;
                row.kernel_min_ns = s.min_ns;
                row.kernel_median_ns = s.median_ns;
                row.kernel_std_ns = s.std_ns;
                row.kernel_p95_ns = s.p95_ns;
                row.kernel_ci_low_ns = s.ci_low_ns;
                row.kernel_ci_high_ns = s.ci_high_ns;
                row.ref_kernel_min_ns = ref_stats.min_ns;
                row.ref_kernel_median_ns = ref_stats.median_ns;
                row.ref_kernel_std_ns = ref_stats.std_ns;
                row.ref_kernel_p95_ns = ref_stats.p95_ns;
                row.ref_kernel_ci_low_ns = ref_stats.ci_low_ns;
                row.ref_kernel_ci_high_ns = ref_stats.ci_high_ns;
                row.max_abs_err = err.max_abs;
                row.max_rel_err = err.max_rel;
                row.kernel_samples_ns = samples_ns;
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
