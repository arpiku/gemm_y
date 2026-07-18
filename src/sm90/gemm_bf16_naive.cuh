// gemm_bf16_naive.cuh — declaration of the Hopper (sm_90) naive bf16 GEMM
// kernel functor. Identical to src/sm120/gemm_bf16_naive.cuh in Phase 1;
// divergence begins with tensor-core kernels in Phase 2 (see ARD.md §8).

#pragma once

#include <string_view>

#include "bench/GemmArgs.h"
#include "bench/KernelTraits.h"
#include "cuda_compat.h"

namespace gemm_y {

template <typename T>
struct NaiveGemm {
    static constexpr std::string_view name()        { return "naive"; }
    static constexpr std::string_view description() {
        return "naive triple-loop; 1 thread/element; fp32 accum";
    }

    // Defined in gemm_bf16_naive.cu (CUDA translation unit).
    void operator()(GemmArgs<T> args, cudaStream_t stream) const;
};

} // namespace gemm_y
