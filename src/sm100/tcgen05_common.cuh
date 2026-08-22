// tcgen05_common.cuh — minimal sm100a TMA/tcgen05 helpers.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cuda_compat.h"

namespace gemm_y::sm100_detail {

constexpr int kWarpSize = 32;

__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t"
               "}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

__device__ inline void mbarrier_init(int address, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(address),
               "r"(count));
}

__device__ inline void mbarrier_wait(int address, int phase) {
  constexpr uint32_t kTimeoutTicks = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
               "%1, %2;\n\t"
               "@P1 bra.uni DONE;\n\t"
               "bra.uni LAB_WAIT;\n\t"
               "DONE:\n\t"
               "}" ::"r"(address),
               "r"(phase), "r"(kTimeoutTicks));
}

template <int CtaGroup = 1>
__device__ inline void tma_3d_gmem2smem(int destination, const void *tensor_map,
                                        int x, int y, int z, int mbarrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
      "complete_tx::bytes.cta_group::%6 [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
          destination),
      "l"(tensor_map), "r"(x), "r"(y), "r"(z), "r"(mbarrier), "n"(CtaGroup)
      : "memory");
}

template <int CtaGroup = 1>
__device__ inline void
tcgen05_mma_f16(int tensor_address, uint64_t a_desc, uint64_t b_desc,
                uint32_t instruction_desc, int enable_input_d) {
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
               "}" ::"r"(tensor_address),
               "l"(a_desc), "l"(b_desc), "r"(instruction_desc),
               "r"(enable_input_d), "n"(CtaGroup));
}

__device__ inline constexpr uint64_t desc_encode(uint64_t address) {
  return (address & 0x3FFFFULL) >> 4ULL;
}

} // namespace gemm_y::sm100_detail

namespace gemm_y {

inline void check_driver(CUresult result, const char *expression,
                         const char *file, int line) {
  if (result == CUDA_SUCCESS)
    return;
  const char *message = nullptr;
  if (cuGetErrorString(result, &message) != CUDA_SUCCESS ||
      message == nullptr) {
    message = "unknown CUDA driver error";
  }
  std::fprintf(stderr, "CUDA driver error at %s:%d (%s): %s (code %d)\n", file,
               line, expression, message, static_cast<int>(result));
  std::abort();
}

} // namespace gemm_y

#define GEMM_Y_CU_CHECK(expr)                                                  \
  do {                                                                         \
    const CUresult result_ = (expr);                                           \
    if (result_ != CUDA_SUCCESS)                                               \
      ::gemm_y::check_driver(result_, #expr, __FILE__, __LINE__);              \
  } while (0)
