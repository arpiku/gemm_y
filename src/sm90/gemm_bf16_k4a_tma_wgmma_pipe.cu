// gemm_bf16_k4a_tma_wgmma_pipe.cu — Hopper BF16 TMA + WGMMA GEMM.
//
// A CTA computes a 128x128 output tile with a three-stage 128x64 / 64x128
// TMA pipeline. One warp-group produces TMA stages while two warp-groups each
// accumulate one 64-row output band using m64n128k16 WGMMA instructions.

#include "gemm_bf16.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 64;
constexpr int kStages = 3;
constexpr int kWarpGroupThreads = 128;
constexpr int kConsumerWarpGroups = 2;
constexpr int kThreads = (1 + kConsumerWarpGroups) * kWarpGroupThreads;
constexpr int kWgmmaM = 64;
constexpr int kWgmmaN = 128;
constexpr int kWgmmaK = 16;
constexpr int kWgmmaKPerTile = kTileK / kWgmmaK;
// A's minor dimension is split into two 64-row boxes because a 128B
// swizzle cannot encode a 128-element BF16 minor dimension (256 bytes).
constexpr int kTmaABoxRows = kTileM / 2;
constexpr int kTmaBytes =
    (kTileM * kTileK + kTileK * kTileN) * sizeof(__nv_bfloat16);
constexpr int kTmaStrideAlignmentBytes = 16;

struct alignas(128) SharedStorage {
  // TMA writes the 128B-swizzled layouts described by the tensor maps below.
  alignas(128) __nv_bfloat16 A[kStages][kTileM * kTileK];
  alignas(128) __nv_bfloat16 B[kStages][kTileK * kTileN];
  alignas(8) uint64_t full[kStages];
  alignas(8) uint64_t empty[kStages];
};

static_assert(sizeof(SharedStorage) <= 99 * 1024,
              "k4a shared-memory footprint exceeds Hopper's 99 KiB limit");

__device__ __forceinline__ uint32_t smem_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ uint64_t encode_descriptor_field(uint64_t value) {
  return (value & 0x3ffffULL) >> 4ULL;
}

// Descriptor for a 128B-swizzled TMA tile. The 16-byte leading offset and
// 1024-byte stride match the swizzled 64x16 WGMMA matrix core.
__device__ __forceinline__ uint64_t make_wgmma_descriptor(
    const __nv_bfloat16 *pointer) {
  uint64_t descriptor = encode_descriptor_field(smem_address(pointer));
  descriptor |= encode_descriptor_field(16) << 16ULL;
  descriptor |= encode_descriptor_field(1024) << 32ULL;
  descriptor |= 1ULL << 62ULL;
  return descriptor;
}

__device__ __forceinline__ void warpgroup_fence() {
  asm volatile("wgmma.fence.sync.aligned;" : : : "memory");
}

__device__ __forceinline__ void warpgroup_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" : : : "memory");
}

template <int Groups>
__device__ __forceinline__ void warpgroup_wait() {
  static_assert(Groups >= 0 && Groups <= 7,
                "WGMMA wait-group count must be in [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(Groups) : "memory");
}

template <uint32_t Count>
__device__ __forceinline__ void warpgroup_allocate_registers() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" : : "n"(Count));
}

template <uint32_t Count>
__device__ __forceinline__ void warpgroup_release_registers() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" : : "n"(Count));
}

__device__ __forceinline__ void mbarrier_init(uint64_t *barrier, int arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(
                   smem_address(barrier)),
               "r"(arrivals));
}

__device__ __forceinline__ void mbarrier_expect_bytes(uint64_t *barrier,
                                                       uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
               "[%0], %1;"
               :
               : "r"(smem_address(barrier)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t *barrier) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
               :
               : "r"(smem_address(barrier))
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t *barrier, int phase) {
  constexpr uint32_t kWaitHint = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred complete;\n\t"
               "wait_loop:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, "
               "[%0], %1, %2;\n\t"
               "@complete bra.uni wait_done;\n\t"
               "bra.uni wait_loop;\n\t"
               "wait_done:\n\t"
               "}"
               :
               : "r"(smem_address(barrier)), "r"(phase), "r"(kWaitHint)
               : "memory");
}

__device__ __forceinline__ void tma_load_2d(void *destination,
                                            const CUtensorMap *tensor_map,
                                            int x, int y,
                                            uint64_t *barrier) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile."
               "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :
               : "r"(smem_address(destination)), "l"(tensor_map), "r"(x),
                 "r"(y), "r"(smem_address(barrier))
               : "memory");
}

// Every WGMMA thread owns eight FP32 registers for each 16-column output
// fragment of m64n128. The descriptor layout is intentionally fixed with the
// tile geometry above; this is not a generic WGMMA wrapper.
template <int ScaleD>
__device__ __forceinline__ void wgmma_m64n128k16(
    float (&d)[kWgmmaN / 16][8], const __nv_bfloat16 *a,
    const __nv_bfloat16 *b) {
  static_assert(ScaleD == 0 || ScaleD == 1, "ScaleD must be 0 or 1");
  const uint64_t descriptor_a = make_wgmma_descriptor(a);
  const uint64_t descriptor_b = make_wgmma_descriptor(b);
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      " %8, %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31, "
      " %32, %33, %34, %35, %36, %37, %38, %39, "
      " %40, %41, %42, %43, %44, %45, %46, %47, "
      " %48, %49, %50, %51, %52, %53, %54, %55, "
      " %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, %66, %67, %68, %69, %70;"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
        "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
        "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
        "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
        "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
        "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
        "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
        "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
      : "l"(descriptor_a), "l"(descriptor_b), "n"(ScaleD), "n"(1),
        "n"(1), "n"(0), "n"(0));
}

__global__ __launch_bounds__(kThreads) void k4a_tma_wgmma_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, __nv_bfloat16 *C, int M, int N,
    int K, int ldC) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto &shared = *reinterpret_cast<SharedStorage *>(storage);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_group = tid / kWarpGroupThreads;
  const int lane_in_group = tid % kWarpGroupThreads;
  const int block_row = static_cast<int>(blockIdx.x) * kTileM;
  const int block_col = static_cast<int>(blockIdx.y) * kTileN;
  const int k_tiles = K / kTileK;

  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      mbarrier_init(&shared.full[stage], 1);
      mbarrier_init(&shared.empty[stage], kConsumerWarpGroups);
    }
  }
  __syncthreads();

  if (warp_group == 0) {
    warpgroup_release_registers<24>();
    if (lane_in_group == 0) {
      for (int tile = 0; tile < k_tiles; ++tile) {
        const int stage = tile % kStages;
        const int phase = (tile / kStages) & 1;
        mbarrier_wait(&shared.empty[stage], phase);
        mbarrier_expect_bytes(&shared.full[stage], kTmaBytes);
        tma_load_2d(shared.A[stage], &a_map, block_row, tile * kTileK,
                    &shared.full[stage]);
        tma_load_2d(shared.A[stage] + kTmaABoxRows * kTileK, &a_map,
                    block_row + kTmaABoxRows, tile * kTileK,
                    &shared.full[stage]);
        tma_load_2d(shared.B[stage], &b_map, tile * kTileK, block_col,
                    &shared.full[stage]);
      }
    }
    return;
  }

  warpgroup_allocate_registers<240>();
  float accum[kWgmmaN / 16][8] = {};
  const int consumer = warp_group - 1;
  const int consumer_row = consumer * kWgmmaM;

  // Mark every stage reusable before the producer issues its initial preload.
  if (lane_in_group == 0) {
    for (int stage = 0; stage < kStages; ++stage)
      mbarrier_arrive(&shared.empty[stage]);
  }

  for (int tile = 0; tile < k_tiles; ++tile) {
    const int stage = tile % kStages;
    const int phase = (tile / kStages) & 1;
    mbarrier_wait(&shared.full[stage], phase);

    const __nv_bfloat16 *a = shared.A[stage] + consumer_row * kTileK;
    const __nv_bfloat16 *b = shared.B[stage];
    warpgroup_fence();
    if (tile == 0) {
      wgmma_m64n128k16<0>(accum, a, b);
    } else {
      wgmma_m64n128k16<1>(accum, a, b);
    }
#pragma unroll
    for (int k_step = 1; k_step < kWgmmaKPerTile; ++k_step) {
      wgmma_m64n128k16<1>(accum, a + k_step * kWgmmaK,
                           b + k_step * kWgmmaK);
    }
    warpgroup_commit();
    warpgroup_wait<0>();
    if (lane_in_group == 0)
      mbarrier_arrive(&shared.empty[stage]);
  }

  const int lane = lane_in_group & 31;
  const int warp = lane_in_group >> 5;
  const int row = block_row + consumer_row + warp * 16 + lane / 4;
#pragma unroll
  for (int fragment = 0; fragment < kWgmmaN / 16; ++fragment) {
    const int col = block_col + fragment * 16 + 2 * (lane & 3);
    if (row + 8 < M && col + 9 < N) {
      C[(row + 8) + col * ldC] = static_cast<__nv_bfloat16>(accum[fragment][2]);
      C[row + col * ldC] = static_cast<__nv_bfloat16>(accum[fragment][0]);
      C[(row + 8) + (col + 1) * ldC] =
          static_cast<__nv_bfloat16>(accum[fragment][3]);
      C[row + (col + 1) * ldC] = static_cast<__nv_bfloat16>(accum[fragment][1]);
      C[(row + 8) + (col + 8) * ldC] =
          static_cast<__nv_bfloat16>(accum[fragment][6]);
      C[row + (col + 8) * ldC] = static_cast<__nv_bfloat16>(accum[fragment][4]);
      C[(row + 8) + (col + 9) * ldC] =
          static_cast<__nv_bfloat16>(accum[fragment][7]);
      C[row + (col + 9) * ldC] = static_cast<__nv_bfloat16>(accum[fragment][5]);
    }
  }
}

void encode_tma_map(CUtensorMap *map, const __nv_bfloat16 *pointer, int rows,
                    int cols, int ld, int box_rows, int box_cols) {
  constexpr uint32_t kRank = 2;
  const uint64_t dimensions[kRank] = {static_cast<uint64_t>(rows),
                                      static_cast<uint64_t>(cols)};
  const uint64_t strides[kRank - 1] = {
      static_cast<uint64_t>(ld) * sizeof(__nv_bfloat16)};
  const uint32_t box[kRank] = {static_cast<uint32_t>(box_rows),
                               static_cast<uint32_t>(box_cols)};
  const uint32_t element_strides[kRank] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kRank,
      const_cast<__nv_bfloat16 *>(pointer), dimensions, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    const char *message = nullptr;
    (void)cuGetErrorString(status, &message);
    std::fprintf(stderr, "k4a_tma_wgmma_pipe: cuTensorMapEncodeTiled failed: %s\n",
                 message == nullptr ? "unknown error" : message);
    std::abort();
  }
}

} // namespace

bool k4a_tma_wgmma_pipe::supports(const GemmArgs<__nv_bfloat16> &args) {
  const bool dimensions =
      args.C.rows % kTileM == 0 && args.C.cols % kTileN == 0 &&
      args.A.cols % kTileK == 0 && args.A.rows == args.C.rows &&
      args.B.rows == args.A.cols && args.B.cols == args.C.cols;
  const bool strides = args.A.ld >= args.A.rows && args.B.ld >= args.B.rows &&
                       args.C.ld >= args.C.rows &&
                       (args.A.ld * static_cast<int>(sizeof(__nv_bfloat16))) %
                               kTmaStrideAlignmentBytes ==
                           0 &&
                       (args.B.ld * static_cast<int>(sizeof(__nv_bfloat16))) %
                               kTmaStrideAlignmentBytes ==
                           0;
  const bool pointers =
      (reinterpret_cast<uintptr_t>(args.A.ptr) % kTmaStrideAlignmentBytes) ==
          0 &&
      (reinterpret_cast<uintptr_t>(args.B.ptr) % kTmaStrideAlignmentBytes) ==
          0;
  return dimensions && strides && pointers;
}

void k4a_tma_wgmma_pipe::operator()(GemmArgs<__nv_bfloat16> args,
                                     cudaStream_t stream) const {
  if (!supports(args)) {
    std::fprintf(
        stderr, "k4a_tma_wgmma_pipe: unsupported shape, stride, or alignment\n");
    std::abort();
  }

  CUtensorMap a_map{};
  CUtensorMap b_map{};
  encode_tma_map(&a_map, args.A.ptr, args.A.rows, args.A.cols, args.A.ld,
                 kTmaABoxRows, kTileK);
  encode_tma_map(&b_map, args.B.ptr, args.B.rows, args.B.cols, args.B.ld,
                 kTileK, kTileN);

  const dim3 grid(static_cast<unsigned>(args.C.rows / kTileM),
                  static_cast<unsigned>(args.C.cols / kTileN), 1);
  const dim3 block(kThreads, 1, 1);
  constexpr int kSharedBytes = static_cast<int>(sizeof(SharedStorage));
  CUDA_CHECK(cudaFuncSetAttribute(k4a_tma_wgmma_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kSharedBytes));
  k4a_tma_wgmma_kernel<<<grid, block, kSharedBytes, stream>>>(
      a_map, b_map, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
      args.C.ld);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace gemm_y
