// Profiler.h — type-erased kernel registry + sweep runner.
//
// One Profiler<T> instance per dtype. cuBLAS is the implicit reference: run
// first per N, cached on device (C_ref_max), and used as the accuracy ground
// truth. Custom kernels write to a separate C_max buffer (see ARD.md §4, §5).
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

#pragma once

#include <functional>
#include <string>
#include <vector>

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
    double h2d_ns;               // global H2D time (A+B), repeated per row
    double kernel_min_ns;
    double kernel_median_ns;
    double d2h_ns;
    double ref_kernel_min_ns;   // cuBLAS min at this N (== kernel_* for the cuBLAS row)
    double ref_kernel_median_ns;
    double max_abs_err;
    double max_rel_err;
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
        std::function<void(GemmArgs<T>, cudaStream_t)> run;
    };

    Profiler() = default;

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
        e.run = [](GemmArgs<T> args, cudaStream_t s) {
            K{}(args, s);
        };
        kernels_.push_back(std::move(e));
    }

    // Run the full sweep. Pre-allocates 4096x4096 device + host buffers,
    // fills A/B with a deterministic pattern, copies A/B to device once,
    // then per N: runs cuBLAS reference once, then each registered kernel
    // with warmup=20 / timed=50, computes accuracy vs cuBLAS, appends a
    // SweepRow to the result. Returns the SweepResult; does NOT write CSV.
    // Implementation lives in Profiler.cu (compiled per arch).
    [[nodiscard]] SweepResult run_sweep(const std::vector<int>& sizes);

private:
    std::vector<Entry> kernels_;
    CublasHandle cublas_;
};

} // namespace gemm_y
