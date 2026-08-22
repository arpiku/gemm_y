// gemm_naive.cuh — declaration of the Blackwell (sm_120) naive GEMM kernel
// functor. Dtype-agnostic (instantiated for bf16 / fp16 / tfloat from one
// template; casts to fp32 for accumulation, back to T on the write).
//
// The struct definition (with name()/description()) lives here so main.cpp
// (a C++ translation unit) can register it with the Profiler. operator()
// is only declared here and defined in gemm_naive.cu (a CUDA TU), because
// it contains the <<<>>> launch syntax that nvcc must compile.
//
// Perf-irrelevant — exists only to validate the harness end-to-end.

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

    // Defined in gemm_naive.cu (CUDA translation unit).
    void operator()(GemmArgs<T> args, cudaStream_t stream) const;
};

} // namespace gemm_y
