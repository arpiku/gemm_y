// k1_dispatch.cuh — production sm120 bf16 dispatch policy.
//
// The switch is the source-level result of the completed k1_ tuning round.
// Candidate selection is compile-time and keyed by the supported square N;
// k1_smem remains the deterministic fallback for other sizes.

#pragma once

#include "gemm_bf16.cuh"

namespace gemm_y {

struct k1_dispatch {
  static constexpr std::string_view name() { return "k1_dispatch"; }

  static constexpr std::string_view selected_kernel_name(int n) {
    switch (n) {
    case 32:
    case 64:
      return "k1_t_32x64x16";
    case 96:
    case 128:
    case 192:
    case 256:
    case 384:
    case 512:
      return "k1_t_64x32x16";
    case 768:
      return "k1_t_64x64x8";
    case 1024:
    case 1536:
    case 2048:
    case 3072:
    case 4096:
      return "k1_t";
    default:
      return "k1_smem";
    }
  }

  static constexpr std::string_view description() {
    return "source-level k1_ switch dispatch; tuned sm120 bf16; fallback "
           "k1_smem";
  }

  void operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const {
    switch (args.C.cols) {
    case 32:
    case 64:
      k1_t<32, 64, 16>{}(args, stream);
      return;
    case 96:
    case 128:
    case 192:
    case 256:
    case 384:
    case 512:
      k1_t<64, 32, 16>{}(args, stream);
      return;
    case 768:
      k1_t<64, 64, 8>{}(args, stream);
      return;
    case 1024:
    case 1536:
    case 2048:
    case 3072:
    case 4096:
      k1_t<64, 64, 16>{}(args, stream);
      return;
    default:
      k1_smem{}(args, stream);
      return;
    }
  }
};

} // namespace gemm_y
