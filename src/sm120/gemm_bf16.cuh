// gemm_bf16.cuh — shared declaration header for all bf16-specific custom
// kernels on sm_120 (Blackwell).
//
// Each declaration here provides name(), description(), and
// operator()(GemmArgs<__nv_bfloat16>, cudaStream_t) const. Kernel families may
// be ordinary structs or class templates when compile-time configuration is
// needed. Each kernel's operator() definition lives in its own
// gemm_bf16_k<n>.cu file
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

// k0_ldg — scalar baseline with explicit read-only-cache loads.
struct k0_ldg {
  static constexpr std::string_view name() { return "k0_ldg"; }
  static constexpr std::string_view description() {
    return "bf16 scalar GEMM; one thread/element; fp32 accum; __ldg A/B loads";
  }
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

// k0_coalesced — scalar baseline with an A-favorable 32x8 thread mapping.
struct k0_coalesced {
  static constexpr std::string_view name() { return "k0_coalesced"; }
  static constexpr std::string_view description() {
    return "bf16 scalar GEMM; one thread/element; fp32 accum; 32x8 A-coalesced "
           "mapping";
  }
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

} // namespace gemm_y
