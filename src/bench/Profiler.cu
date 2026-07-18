// Profiler.cu — implementation of Profiler<T>::run_sweep.
//
// Compiled per arch (alongside the arch-specific kernel directory). The
// sweep logic is arch-agnostic; only the kernels it dispatches to differ.

#include "Profiler.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "Accuracy.h"
#include "Copy.h"
#include "CudaTimer.h"
#include "CsvWriter.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Tracer.h"

namespace gemm_y {

namespace {

constexpr int kMaxN = 4096;
constexpr int kWarmup = 20;
constexpr int kTimed = 50;

// Deterministic host fill for A and B. Distinct patterns so A and B don't
// alias in cache; values kept small to keep bf16/fp32 accumulation in a
// reasonable range for accuracy comparison.
template <typename T>
void fill_host_ab(MatrixView<T, Space::Host> A, MatrixView<T, Space::Host> B) {
    for (int j = 0; j < A.cols; ++j) {
        for (int i = 0; i < A.rows; ++i) {
            A(i, j) = static_cast<T>(((i + j) & 7) - 3);   // small ints in [-3, 4]
        }
    }
    for (int j = 0; j < B.cols; ++j) {
        for (int i = 0; i < B.rows; ++i) {
            B(i, j) = static_cast<T>(((i - j) & 7) - 3);
        }
    }
}

template <typename T>
const char* dtype_name();
template <> const char* dtype_name<__nv_bfloat16>() { return "bf16"; }
template <> const char* dtype_name<__half>()          { return "fp16"; }
template <> const char* dtype_name<float>()          { return "fp32"; }

#if defined(CUDA_ARCH_SM_90)
    constexpr const char* kArchName = "sm_90";
#elif defined(CUDA_ARCH_SM_120)
    constexpr const char* kArchName = "sm_120";
#else
    #error "Neither CUDA_ARCH_SM_90 nor CUDA_ARCH_SM_120 is defined."
#endif

// Compute min and median of a sorted copy of `samples` (in ns).
struct TimedStats { double min_ns; double median_ns; };
TimedStats summarize_ns(std::vector<float>& ms_samples) {
    std::sort(ms_samples.begin(), ms_samples.end());
    TimedStats s;
    s.min_ns = static_cast<double>(ms_samples.front()) * 1e6;
    const std::size_t n = ms_samples.size();
    s.median_ns = (n % 2 == 0)
        ? static_cast<double>(ms_samples[n / 2 - 1] + ms_samples[n / 2]) * 0.5 * 1e6
        : static_cast<double>(ms_samples[n / 2]) * 1e6;
    return s;
}

} // namespace

template <typename T>
void Profiler<T>::run_sweep(const std::vector<int>& sizes,
                            const std::string& out_csv_path) {
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
    fill_host_ab<T>(hA_max.view(), hB_max.view());

    // H2D A, B once (timed; reported in CSV).
    CudaTimer h2d_timer;
    h2d_timer.start();
    copy_h2d(A_max.view(), hA_max.view());
    copy_h2d(B_max.view(), hB_max.view());
    h2d_timer.stop();
    const double h2d_total_ns = static_cast<double>(h2d_timer.elapsed_ms()) * 1e6;

    std::printf("[Profiler] arch=%s dtype=%s kernels=%zu sizes=%zu  h2d(A+B)=%.2f ms\n",
                kArchName, dtype_name<T>(), kernels_.size(), sizes.size(),
                h2d_total_ns / 1e6);

    CsvWriter csv;
    if (!csv.open(out_csv_path)) {
        std::fprintf(stderr, "Profiler: failed to open %s for write\n",
                     out_csv_path.c_str());
        std::abort();
    }
    csv.write_header(
        "arch,dtype,N,kernel_name,kernel_desc,h2d_ns,kernel_min_ns,kernel_median_ns,"
        "d2h_ns,ref_kernel_min_ns,ref_kernel_median_ns,max_abs_err,max_rel_err");

    tracer::Timer<4096> sweep_timer;
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

        // 1) cuBLAS reference -> dC_ref. Warmup + timed.
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
            const TimedStats s = summarize_ns(ms);

            // D2H C_ref once (timed).
            CudaTimer d2h_t;
            d2h_t.start();
            copy_d2h(hC_ref_max.view().block(0, 0, N, N), dC_ref);
            d2h_t.stop();
            const double ref_d2h_ns = static_cast<double>(d2h_t.elapsed_ms()) * 1e6;

            // We'll write the cuBLAS row first (kernel_name = "cublas",
            // kernel_desc = "cublasGemmEx reference"). ref_* columns are
            // self-referential here (== kernel_*).
            csv.append_row(std::string(kArchName), std::string(dtype_name<T>()), N,
                           std::string("cublas"),
                           std::string("cublasGemmEx reference (fp32 accum)"),
                           0.0,                       // h2d_ns (already done globally)
                           s.min_ns, s.median_ns,
                           ref_d2h_ns,
                           s.min_ns, s.median_ns,     // ref == self for the cuBLAS row
                           0.0, 0.0);                 // accuracy: ground truth vs itself
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
                // Copy current C_ref (already on host as hC_ref_max) into a
                // flat snapshot. hC_ref_max is ColMajor with ld=kMaxN.
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
            const TimedStats s = summarize_ns(ms);

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
                // Re-D2H the current C_ref and compare against the snapshot.
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

            // cuBLAS reference time at this N (re-read from the cuBLAS row
            // we just wrote would be circular; instead we re-time cuBLAS
            // briefly here for the ref_* columns). To keep the sweep cheap,
            // we re-run a small timed loop (kTimed iters) for cuBLAS at
            // this N. This is the same cost as the cuBLAS row above; we
            // accept the duplication for simplicity in Phase 1.
            std::vector<float> ref_ms; ref_ms.reserve(kTimed);
            CudaTimer rt;
            for (int i = 0; i < kTimed; ++i) {
                rt.start();
                cublas_gemm(cublas_, dA, dB, dC_ref);
                rt.stop();
                ref_ms.push_back(rt.elapsed_ms());
            }
            const TimedStats ref_s = summarize_ns(ref_ms);

            csv.append_row(std::string(kArchName), std::string(dtype_name<T>()), N,
                           k.name, k.description,
                           0.0,                       // h2d_ns (global; not per-kernel)
                           s.min_ns, s.median_ns,
                           d2h_ns,
                           ref_s.min_ns, ref_s.median_ns,
                           err.max_abs, err.max_rel);

            const char* verdict = (err.max_rel <= kRelErrTol) ? "PASS" : "FAIL";
            std::printf("[Profiler] N=%4d  %-12s  kernel=%.1f us  ref=%.1f us  "
                        "rel_err=%.3e  %s\n",
                        N, k.name.c_str(),
                        s.median_ns / 1e3, ref_s.median_ns / 1e3,
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
}

// Explicit instantiations for the dtypes we support in Phase 1+.
// Phase 1 wires bf16 only; fp16/fp32 are listed so the harness compiles
// when their kernels are added in Phase 2+.
template class Profiler<__nv_bfloat16>;
template class Profiler<__half>;
template class Profiler<float>;

} // namespace gemm_y
