// gemm_bf16.cuh — shared declaration header for all bf16-specific custom
// kernels on sm_90 (Hopper). Mirror of src/sm120/gemm_bf16.cuh; divergence
// between arches begins with tensor-core kernels (see ARD §8, §16).
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

// k0 — dummy custom kernel (verbatim naive body). Workflow development
// only; no optimization. Perf ≈ NaiveGemm. Rename to a descriptive name
// when a real strategy lands (see ARD §16).
struct k0 {
  static constexpr std::string_view name() { return "k0"; }
  static constexpr std::string_view description() {
    return "k0: dummy = naive kernel body (workflow development)";
  }
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

// k4a_tma_wgmma_pipe — proven Hopper TMA/WGMMA baseline.
struct k4a_tma_wgmma_pipe {
  static constexpr std::string_view name() { return "k4a_tma_wgmma_pipe"; }
  static constexpr std::string_view description() {
    return "sm90a BF16 TMA/WGMMA; 128x128x64 CTA; 3-stage pipeline; 1 producer + 2 consumer warp-groups";
  }
  static bool supports(const GemmArgs<__nv_bfloat16> &args);
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

// k4b_tma_wgmma_pipe — wider-N Hopper TMA/WGMMA candidate.
struct k4b_tma_wgmma_pipe {
  static constexpr std::string_view name() { return "k4b_tma_wgmma_pipe"; }
  static constexpr std::string_view description() {
    return "sm90a BF16 TMA/WGMMA; 128x256x64 CTA; 2-stage pipeline; 1 producer + 2 consumer warp-groups; m64n256k16";
  }
  static bool supports(const GemmArgs<__nv_bfloat16> &args);
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

// k4_t — templated Hopper TMA/WGMMA pipeline (generalizes k4a/k4b).
//
// Exposes the CTA tile and pipeline depth as compile-time knobs for the
// tuning round; the WGMMA instruction shape is derived from TileN
// (m64n128k16 or m64n256k16). TileM=128 (one producer + two consumer
// warp-groups) and TileK=64 (128B-swizzled TMA boxes) are fixed this round;
// widening them requires new WGMMA shapes / TMA box encodings.
template <int TileM, int TileN, int TileK, int Stages> struct k4_t {
  static_assert(TileM == 128,
                "k4_t currently requires TileM=128 (two consumer warp-groups)");
  static_assert(TileN == 128 || TileN == 256,
                "k4_t supports TileN=128 (m64n128k16) or TileN=256 (m64n256k16)");
  static_assert(TileK == 64,
                "k4_t currently requires TileK=64 (128B-swizzled TMA boxes)");
  static_assert(Stages >= 2, "k4_t needs at least two pipeline stages");

  static constexpr std::string_view name() {
    if constexpr (TileM == 128 && TileN == 128 && TileK == 64 && Stages == 3) {
      return "k4_t_128x128x64_s3";
    } else if constexpr (TileM == 128 && TileN == 128 && TileK == 64 &&
                         Stages == 2) {
      return "k4_t_128x128x64_s2";
    } else if constexpr (TileM == 128 && TileN == 256 && TileK == 64 &&
                         Stages == 2) {
      return "k4_t_128x256x64_s2";
    } else {
      return "k4_t_custom";
    }
  }

  static constexpr std::string_view description() {
    if constexpr (TileM == 128 && TileN == 128 && TileK == 64 && Stages == 3) {
      return "sm90a BF16 TMA/WGMMA (templated); 128x128x64 CTA; 3-stage "
             "pipeline; m64n128k16";
    } else if constexpr (TileM == 128 && TileN == 128 && TileK == 64 &&
                         Stages == 2) {
      return "sm90a BF16 TMA/WGMMA (templated); 128x128x64 CTA; 2-stage "
             "pipeline; m64n128k16";
    } else if constexpr (TileM == 128 && TileN == 256 && TileK == 64 &&
                         Stages == 2) {
      return "sm90a BF16 TMA/WGMMA (templated); 128x256x64 CTA; 2-stage "
             "pipeline; m64n256k16";
    } else {
      return "sm90a BF16 TMA/WGMMA (templated); custom tile/stage config";
    }
  }

  static bool supports(const GemmArgs<__nv_bfloat16> &args);
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

// k1_smem — single-buffered cooperative shared-memory tile.
struct k1_smem {
  static constexpr std::string_view name() { return "k1_smem"; }
  static constexpr std::string_view description() {
    return "bf16 shared-memory GEMM; 64x64x16 tile; 16x16 block; 4x4/thread; "
           "single-buffered";
  }
  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

template <int TileM, int TileN, int TileK> struct k1_t {
  static_assert(TileM > 0 && TileN > 0 && TileK > 0,
                "k1_t tile dimensions must be positive");
  static_assert(TileM % 16 == 0 && TileN % 16 == 0,
                "k1_t M/N tiles must be divisible by the fixed 16x16 block");

  static constexpr std::string_view name() {
    if constexpr (TileM == 64 && TileN == 64 && TileK == 16) {
      return "k1_t";
    } else if constexpr (TileM == 64 && TileN == 64 && TileK == 8) {
      return "k1_t_64x64x8";
    } else if constexpr (TileM == 32 && TileN == 64 && TileK == 16) {
      return "k1_t_32x64x16";
    } else if constexpr (TileM == 64 && TileN == 32 && TileK == 16) {
      return "k1_t_64x32x16";
    } else {
      return "k1_t_custom";
    }
  }

  static constexpr std::string_view description() {
    if constexpr (TileM == 64 && TileN == 64 && TileK == 16) {
      return "bf16 dynamic-smem GEMM; 64x64x16 tile; 16x16 block; "
             "4x4/thread; single-buffered";
    } else if constexpr (TileM == 64 && TileN == 64 && TileK == 8) {
      return "bf16 dynamic-smem GEMM; 64x64x8 tile; 16x16 block; "
             "4x4/thread; single-buffered";
    } else if constexpr (TileM == 32 && TileN == 64 && TileK == 16) {
      return "bf16 dynamic-smem GEMM; 32x64x16 tile; 16x16 block; "
             "2x4/thread; single-buffered";
    } else if constexpr (TileM == 64 && TileN == 32 && TileK == 16) {
      return "bf16 dynamic-smem GEMM; 64x32x16 tile; 16x16 block; "
             "4x2/thread; single-buffered";
    } else {
      return "bf16 dynamic-smem GEMM; custom compile-time tile; 16x16 block; "
             "single-buffered";
    }
  }

  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const;
};

} // namespace gemm_y
