// gemm_bf16_k4_t.cu — templated Hopper BF16 TMA + WGMMA GEMM.
//
// Generalizes the k4a/k4b pipeline into a single template with the CTA tile
// and pipeline depth as compile-time knobs. One producer warp-group fills a
// Stages-deep TMA pipeline while TileM/64 consumer warp-groups each
// accumulate one 64-row output band; the WGMMA instruction shape is derived
// from TileN (m64n128k16 or m64n256k16). TileM=128 and TileK=64 are fixed
// this round: the epilogue assumes two consumer warp-groups and the TMA box
// encoding assumes 128B-swizzled 64-element minor dimensions.

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

constexpr int kWarpGroupThreads = 128;
constexpr int kWgmmaM = 64;
constexpr int kWgmmaK = 16;
constexpr int kTmaStrideAlignmentBytes = 16;

template <int TileM, int TileN, int TileK, int Stages>
struct alignas(128) SharedStorage {
  // TMA writes the 128B-swizzled layouts described by the tensor maps below.
  alignas(128) __nv_bfloat16 A[Stages][TileM * TileK];
  alignas(128) __nv_bfloat16 B[Stages][TileK * TileN];
  alignas(8) uint64_t full[Stages];
  alignas(8) uint64_t empty[Stages];
};

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
    float (&d)[128 / 16][8], const __nv_bfloat16 *a, const __nv_bfloat16 *b) {
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

// Every WGMMA thread owns eight FP32 registers for each 16-column output
// fragment of m64n256. The descriptor layout is fixed for the 128B-swizzled
// 64-element-minor-dimension TMA tiles used by this kernel.
template <int ScaleD>
__device__ __forceinline__ void wgmma_m64n256k16(
    float (&d)[256 / 16][8], const __nv_bfloat16 *a, const __nv_bfloat16 *b) {
  static_assert(ScaleD == 0 || ScaleD == 1, "ScaleD must be 0 or 1");
  const uint64_t descriptor_a = make_wgmma_descriptor(a);
  const uint64_t descriptor_b = make_wgmma_descriptor(b);
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      " %8, %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31, "
      " %32, %33, %34, %35, %36, %37, %38, %39, "
      " %40, %41, %42, %43, %44, %45, %46, %47, "
      " %48, %49, %50, %51, %52, %53, %54, %55, "
      " %56, %57, %58, %59, %60, %61, %62, %63, "
      " %64, %65, %66, %67, %68, %69, %70, %71, "
      " %72, %73, %74, %75, %76, %77, %78, %79, "
      " %80, %81, %82, %83, %84, %85, %86, %87, "
      " %88, %89, %90, %91, %92, %93, %94, %95, "
      " %96, %97, %98, %99, %100, %101, %102, %103, "
      " %104, %105, %106, %107, %108, %109, %110, %111, "
      " %112, %113, %114, %115, %116, %117, %118, %119, "
      " %120, %121, %122, %123, %124, %125, %126, %127}, "
      "%128, %129, %130, %131, %132, %133, %134;"
      :
        "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
        "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
        "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
        "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
        "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
        "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
        "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
        "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
        "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
        "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
        "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
        "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
        "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
        "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
      : "l"(descriptor_a), "l"(descriptor_b), "n"(ScaleD), "n"(1),
        "n"(1), "n"(0), "n"(0));
}

// Selects the WGMMA instruction matching the template's TileN.
template <int N, int ScaleD>
__device__ __forceinline__ void wgmma_m64n_k16(float (&d)[N / 16][8],
                                               const __nv_bfloat16 *a,
                                               const __nv_bfloat16 *b) {
  static_assert(N == 128 || N == 256,
                "k4_t supports m64n128k16 or m64n256k16 only");
  if constexpr (N == 128) {
    wgmma_m64n128k16<ScaleD>(d, a, b);
  } else {
    wgmma_m64n256k16<ScaleD>(d, a, b);
  }
}

template <int TileM, int TileN, int TileK, int Stages>
__global__ __launch_bounds__((1 + TileM / kWgmmaM) * kWarpGroupThreads) void
k4_t_gemm_kernel(const __grid_constant__ CUtensorMap a_map,
                 const __grid_constant__ CUtensorMap b_map, __nv_bfloat16 *C,
                 int M, int N, int K, int ldC) {
  constexpr int kConsumerWarpGroups = TileM / kWgmmaM;
  constexpr int kWgmmaKPerTile = TileK / kWgmmaK;
  // A's minor dimension is split into two 64-row boxes because a 128B
  // swizzle cannot encode a 128-element BF16 minor dimension (256 bytes).
  constexpr int kTmaABoxRows = TileM / 2;
  constexpr int kTmaBytes =
      (TileM * TileK + TileK * TileN) * static_cast<int>(sizeof(__nv_bfloat16));

  extern __shared__ __align__(128) unsigned char storage[];
  auto &shared =
      *reinterpret_cast<SharedStorage<TileM, TileN, TileK, Stages> *>(storage);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_group = tid / kWarpGroupThreads;
  const int lane_in_group = tid % kWarpGroupThreads;
  const int block_row = static_cast<int>(blockIdx.x) * TileM;
  const int block_col = static_cast<int>(blockIdx.y) * TileN;
  const int k_tiles = K / TileK;

  if (tid == 0) {
    for (int stage = 0; stage < Stages; ++stage) {
      mbarrier_init(&shared.full[stage], 1);
      mbarrier_init(&shared.empty[stage], kConsumerWarpGroups);
    }
  }
  __syncthreads();

  if (warp_group == 0) {
    warpgroup_release_registers<24>();
    if (lane_in_group == 0) {
      for (int tile = 0; tile < k_tiles; ++tile) {
        const int stage = tile % Stages;
        const int phase = (tile / Stages) & 1;
        mbarrier_wait(&shared.empty[stage], phase);
        mbarrier_expect_bytes(&shared.full[stage], kTmaBytes);
        tma_load_2d(shared.A[stage], &a_map, block_row, tile * TileK,
                    &shared.full[stage]);
        tma_load_2d(shared.A[stage] + kTmaABoxRows * TileK, &a_map,
                    block_row + kTmaABoxRows, tile * TileK, &shared.full[stage]);
        tma_load_2d(shared.B[stage], &b_map, tile * TileK, block_col,
                    &shared.full[stage]);
      }
    }
    return;
  }

  warpgroup_allocate_registers<240>();
  float accum[TileN / 16][8] = {};
  const int consumer = warp_group - 1;
  const int consumer_row = consumer * kWgmmaM;

  // Mark every stage reusable before the producer issues its initial preload.
  if (lane_in_group == 0) {
    for (int stage = 0; stage < Stages; ++stage)
      mbarrier_arrive(&shared.empty[stage]);
  }

  for (int tile = 0; tile < k_tiles; ++tile) {
    const int stage = tile % Stages;
    const int phase = (tile / Stages) & 1;
    mbarrier_wait(&shared.full[stage], phase);

    const __nv_bfloat16 *a = shared.A[stage] + consumer_row * TileK;
    const __nv_bfloat16 *b = shared.B[stage];
    warpgroup_fence();
    if (tile == 0) {
      wgmma_m64n_k16<TileN, 0>(accum, a, b);
    } else {
      wgmma_m64n_k16<TileN, 1>(accum, a, b);
    }
#pragma unroll
    for (int k_step = 1; k_step < kWgmmaKPerTile; ++k_step) {
      wgmma_m64n_k16<TileN, 1>(accum, a + k_step * kWgmmaK,
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
  for (int fragment = 0; fragment < TileN / 16; ++fragment) {
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
    std::fprintf(stderr, "k4_t: cuTensorMapEncodeTiled failed: %s\n",
                 message == nullptr ? "unknown error" : message);
    std::abort();
  }
}

} // namespace

template <int TileM, int TileN, int TileK, int Stages>
bool k4_t<TileM, TileN, TileK, Stages>::supports(
    const GemmArgs<__nv_bfloat16> &args) {
  const bool dimensions =
      args.C.rows % TileM == 0 && args.C.cols % TileN == 0 &&
      args.A.cols % TileK == 0 && args.A.rows == args.C.rows &&
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

template <int TileM, int TileN, int TileK, int Stages>
void k4_t<TileM, TileN, TileK, Stages>::operator()(
    GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const {
  static_assert(sizeof(SharedStorage<TileM, TileN, TileK, Stages>) <= 99 * 1024,
                "k4_t shared-memory footprint exceeds Hopper's 99 KiB limit");
  if (!supports(args)) {
    std::fprintf(stderr,
                 "k4_t: unsupported shape, stride, or alignment\n");
    std::abort();
  }

  CUtensorMap a_map{};
  CUtensorMap b_map{};
  encode_tma_map(&a_map, args.A.ptr, args.A.rows, args.A.cols, args.A.ld,
                 TileM / 2, TileK);
  encode_tma_map(&b_map, args.B.ptr, args.B.rows, args.B.cols, args.B.ld,
                 TileK, TileN);

  const dim3 grid(static_cast<unsigned>(args.C.rows / TileM),
                  static_cast<unsigned>(args.C.cols / TileN), 1);
  const dim3 block((1 + TileM / kWgmmaM) * kWarpGroupThreads, 1, 1);
  constexpr int kSharedBytes =
      static_cast<int>(sizeof(SharedStorage<TileM, TileN, TileK, Stages>));
  CUDA_CHECK(cudaFuncSetAttribute(k4_t_gemm_kernel<TileM, TileN, TileK, Stages>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kSharedBytes));
  k4_t_gemm_kernel<TileM, TileN, TileK, Stages>
      <<<grid, block, kSharedBytes, stream>>>(
          a_map, b_map, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
          args.C.ld);
  CUDA_CHECK_LAST_ERROR();
}

template struct k4_t<128, 128, 64, 3>;
template struct k4_t<128, 128, 64, 2>;
template struct k4_t<128, 256, 64, 2>;

} // namespace gemm_y
