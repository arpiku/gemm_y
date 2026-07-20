// gemm_bf16.cuh — shared declaration header for all bf16-specific custom
// kernels on sm_90 (Hopper). Mirror of src/sm120/gemm_bf16.cuh; divergence
// between arches begins with tensor-core kernels (see ARD §8, §16).
//
// Each struct here declares name(), description(), and
// operator()(GemmArgs<__nv_bfloat16>, cudaStream_t) const (no template
// parameter — tiled TC kernels are dtype-specific by nature). Each
// kernel's operator() definition lives in its own gemm_bf16_k<n>.cu file
// (compile isolation — a one-line edit to k5 should not recompile k0–k4).
// CMake's file(GLOB _gemm_y_arch_sources CONFIGURE_DEPENDS "${arch_dir}/*.cu")
// auto-picks new .cu files; zero CMake changes per kernel.
//
// NaiveGemm<T> (the dtype-agnostic sanity baseline) lives in
// gemm_naive.{cuh,cu}, not here. See ARD §16.

#pragma once

#include <string_view>

#include "bench/GemmArgs.h"
#include "bench/KernelTraits.h"
#include "cuda_compat.h"

namespace gemm_y {

// k0 — dummy custom kernel (verbatim naive body). Workflow development
// only; no optimization. Perf ≈ NaiveGemm. Rename to a descriptive name
// when a real strategy lands (see ARD §16).
struct k0 {
    static constexpr std::string_view name()        { return "k0"; }
    static constexpr std::string_view description() {
        return "k0: dummy = naive kernel body (workflow development)";
    }
    void operator()(GemmArgs<__nv_bfloat16> args,
                    cudaStream_t stream) const;
};

} // namespace gemm_y
