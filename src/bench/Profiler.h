// Profiler.h — type-erased kernel registry + sweep runner.
//
// One Profiler<T> instance per dtype. cuBLAS is the implicit reference: run
// first per N, cached on device (C_ref), and used as the accuracy ground
// truth. Custom kernels write to a separate C buffer (see ARD.md §4, §5).
//
// register_kernel<K>() type-erases into std::function once at registration.
// Stateless K (the common case) fits SBO; stateful K pays one heap alloc,
// amortized across the full sweep (see ARD.md §4 overhead analysis).
//
// run_sweep returns SweepResult — a vector of SweepRow — instead of writing
// CSV directly. main.cpp owns CsvWriter and iterates result.rows. This
// decouples bench logic from I/O and makes run_sweep testable in isolation.
// cuBLAS is measured once per N, stored as the first row per N in
// SweepResult, and its kernel_min_ns / kernel_median_ns are reused as the
// ref_* columns for every subsequent kernel row at that N.
//
// Measurement (ARD §7, §19): all kernels and cuBLAS run on a single owned
// CUDA stream (`stream_`). A `cudaStreamSynchronize` after each warmup loop
// ensures the GPU is idle before the timed loop begins. One CudaTimer
// (`timer_`) is reused across the entire sweep — one cudaEventCreate/Destroy
// pair instead of ~224. h2d_ns/d2h_ns are no longer reported (harness
// overhead, not kernel performance — see ARD §2E).

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "CudaCheck.h"
#include "CudaTimer.h"
#include "GemmArgs.h"
#include "KernelTraits.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Space.h"
#include "cuda_compat.h"
#include "cublas/CublasHandle.h"
#include "cublas/cublas_gemm.h"

namespace gemm_y {

// One row of the sweep result. Maps 1:1 to a CSV row. POD; pass-by-value.
struct SweepRow {
    std::string arch;
    std::string dtype;
    int N;
    std::string kernel_name;
    std::string kernel_desc;
    double kernel_min_ns;
    double kernel_median_ns;
    double kernel_std_ns;        // sample std dev (n-1)
    double kernel_p95_ns;        // 95th percentile
    double kernel_ci_low_ns;     // 95% CI lower bound for the median
    double kernel_ci_high_ns;    // 95% CI upper bound for the median
    double ref_kernel_min_ns;    // cuBLAS min at this N (== kernel_* for the cuBLAS row)
    double ref_kernel_median_ns;
    double ref_kernel_std_ns;
    double ref_kernel_p95_ns;
    double ref_kernel_ci_low_ns;
    double ref_kernel_ci_high_ns;
    double max_abs_err;
    double max_rel_err;
    // Raw timed samples (ns) as a semicolon-separated string, e.g.
    // "1234.5;1235.1;...". Populated from the kTimed (50) cudaEvent samples.
    // Empty for rows where samples weren't collected. Parsed by ingest.py
    // into the `samples` table for distribution / diff views (ARD §20).
    std::string kernel_samples_ns;
};

// Result of a full sweep. main.cpp iterates rows to write CSV.
struct SweepResult {
    std::vector<SweepRow> rows;
};

template <typename T>
class Profiler {
public:
    struct Entry {
        std::string name;
        std::string description;
        std::function<bool(const GemmArgs<T>&)> supports;
        std::function<void(GemmArgs<T>, cudaStream_t)> run;
    };

    Profiler() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~Profiler() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(Profiler&&) = delete;

    // Register a kernel functor. Static_assert enforces the KernelTraits
    // contract at the call site so misuse fails loudly at compile time.
    template <typename K>
    void register_kernel() {
        static_assert(KernelTraits_v<K, T>,
                      "K does not satisfy KernelTraits<T> (needs name(), "
                      "description(), and operator()(GemmArgs<T>, cudaStream_t) const).");
        Entry e;
        e.name = std::string(K::name());
        e.description = std::string(K::description());
        e.supports = [](const GemmArgs<T>&) { return true; };
        e.run = [](GemmArgs<T> args, cudaStream_t s) {
            K{}(args, s);
        };
        kernels_.push_back(std::move(e));
    }

    // Register a kernel with a domain predicate. Unsupported cases are
    // reported as skips and are not timed or emitted as result rows.
    template <typename K>
    void register_kernel_if(std::function<bool(const GemmArgs<T>&)> supports) {
        static_assert(KernelTraits_v<K, T>,
                      "K does not satisfy KernelTraits<T> (needs name(), "
                      "description(), and operator()(GemmArgs<T>, cudaStream_t) const).");
        Entry e;
        e.name = std::string(K::name());
        e.description = std::string(K::description());
        e.supports = std::move(supports);
        e.run = [](GemmArgs<T> args, cudaStream_t s) {
            K{}(args, s);
        };
        kernels_.push_back(std::move(e));
    }

    // Run the full sweep. Per N: allocates N×N device + host buffers
    // (A/B/C/C_ref), fills A/B with a deterministic pattern, copies A/B to
    // device, runs cuBLAS reference once, then each registered kernel with
    // warmup=20 / timed=50 (with a post-warmup sync), computes accuracy vs
    // cuBLAS, appends a SweepRow to the result. Returns the SweepResult;
    // does NOT write CSV. Implementation lives in Profiler.cu (compiled per arch).
    [[nodiscard]] SweepResult run_sweep(const std::vector<int>& sizes);

private:
    std::vector<Entry> kernels_;
    CublasHandle cublas_;
    cudaStream_t stream_ = nullptr;
    CudaTimer timer_;  // reused across the entire sweep
};

} // namespace gemm_y
