// Profiler.h — type-erased kernel registry + sweep runner.
//
// One Profiler<T> instance per dtype. cuBLAS is the implicit reference: run
// first per N, cached on device (C_ref_max), and used as the accuracy ground
// truth. Custom kernels write to a separate C_max buffer (see ARD.md §4, §5).
//
// register_kernel<K>() type-erases into std::function once at registration.
// Stateless K (the common case) fits SBO; stateful K pays one heap alloc,
// amortized across the full sweep (see ARD.md §4 overhead analysis).

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "CudaTimer.h"
#include "cublas/CublasHandle.h"
#include "GemmArgs.h"
#include "KernelTraits.h"
#include "Matrix.h"
#include "MatrixView.h"
#include "Space.h"
#include "cuda_compat.h"
#include "cublas/cublas_gemm.h"

namespace gemm_y {

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
    // then per N: runs cuBLAS reference, then each registered kernel with
    // warmup=20 / timed=50, computes accuracy vs cuBLAS, writes a CSV row.
    // Implementation lives in Profiler.cu (compiled per arch).
    void run_sweep(const std::vector<int>& sizes, const std::string& out_csv_path);

private:
    std::vector<Entry> kernels_;
    CublasHandle cublas_;
};

} // namespace gemm_y
