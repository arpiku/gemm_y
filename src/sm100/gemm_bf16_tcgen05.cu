// gemm_bf16_tcgen05.cu — first sm100 tcgen05 BF16 GEMM experiment.
//
// two-stage TMA staging, one warp issuing tcgen05.mma, and tensor-memory loads
// in the epilogue.
// For this square-only
// experiment, the current column-major A/B buffers are supplied as B/A to the
// row-major formulation, producing (B^T A^T) = (AB)^T in the same C buffer.
// This preserves the repository's column-major contract without packing.

#include "gemm_bf16.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"
#include "tcgen05_common.cuh"

namespace gemm_y {
namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 128;
constexpr int kMmaK = 16;
constexpr int kNumStages = 2;
constexpr int kNumWarps = 4;
constexpr int kThreads = kNumWarps * sm100_detail::kWarpSize;

static_assert(kBlockK % 64 == 0);
static_assert(kBlockN % kMmaK == 0);
static_assert(kBlockM % 16 == 0);

__global__ __launch_bounds__(kThreads) void tcgen05_gemm_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map, __nv_bfloat16 *c, int n) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / sm100_detail::kWarpSize;
  const int lane_id = tid % sm100_detail::kWarpSize;
  const int bid = static_cast<int>(blockIdx.x);
  const int grid_n = n / kBlockN;
  const int block_row = (bid / grid_n) * kBlockM;
  const int block_col = (bid % grid_n) * kBlockN;

  extern __shared__ __align__(1024) char smem_storage[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_storage));
  constexpr int a_size = kBlockM * kBlockK * sizeof(__nv_bfloat16);
  constexpr int b_size = kBlockN * kBlockK * sizeof(__nv_bfloat16);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t tma_mbarriers[kNumStages];
  __shared__ uint64_t mma_mbarriers[1];
  __shared__ int tensor_memory_address[1];

  const int tma_mbarrier =
      static_cast<int>(__cvta_generic_to_shared(tma_mbarriers));
  const int mma_mbarrier =
      static_cast<int>(__cvta_generic_to_shared(mma_mbarriers));

  if (warp_id == 0 && sm100_detail::elect_sync()) {
    for (int stage = 0; stage < kNumStages; ++stage) {
      sm100_detail::mbarrier_init(tma_mbarrier + stage * 8, 1);
    }
    sm100_detail::mbarrier_init(mma_mbarrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    // The .sync.aligned allocation is collective across this warp.
    const int address =
        static_cast<int>(__cvta_generic_to_shared(tensor_memory_address));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(address),
        "r"(kBlockN));
  }

  __syncthreads();
  const int tensor_address = tensor_memory_address[0];
  int tma_phase = 0;
  int mma_phase = 0;

  constexpr uint32_t instruction_desc =
      (1U << 4U) |  // output dtype FP32
      (1U << 7U) |  // A dtype BF16
      (1U << 10U) | // B dtype BF16
      ((static_cast<uint32_t>(kBlockN) >> 3U) << 17U) |
      ((static_cast<uint32_t>(kBlockM) >> 4U) << 24U);

  auto make_desc = [](int address) -> uint64_t {
    constexpr int kSharedByteOffset = 8 * 128;
    return sm100_detail::desc_encode(static_cast<uint64_t>(address)) |
           (sm100_detail::desc_encode(static_cast<uint64_t>(kSharedByteOffset))
            << 32ULL) |
           (1ULL << 46ULL) | (2ULL << 61ULL);
  };

  auto load = [&](int iteration) {
    if (warp_id == 0 && sm100_detail::elect_sync()) {
      const int stage = iteration % kNumStages;
      const int barrier = tma_mbarrier + stage * 8;
      const int a_smem = smem + stage * (a_size + b_size);
      const int b_smem = a_smem + a_size;
      const int offset_k = iteration * kBlockK;

      sm100_detail::tma_3d_gmem2smem<1>(a_smem, &a_map, 0, block_row,
                                        offset_k / 64, barrier);
      sm100_detail::tma_3d_gmem2smem<1>(b_smem, &b_map, 0, block_col,
                                        offset_k / 64, barrier);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
                   "[%0], %1;" ::"r"(barrier),
                   "r"(a_size + b_size)
                   : "memory");
    }
  };

  auto compute = [&](int iteration) {
    const int stage = iteration % kNumStages;
    const int barrier = tma_mbarrier + stage * 8;
    sm100_detail::mbarrier_wait(barrier, tma_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int a_smem = smem + stage * (a_size + b_size);
    const int b_smem = a_smem + a_size;
    if (stage == kNumStages - 1)
      tma_phase ^= 1;

    if (warp_id == 0 && sm100_detail::elect_sync()) {
      sm100_detail::tcgen05_mma_f16<1>(tensor_address, make_desc(a_smem),
                                       make_desc(b_smem), instruction_desc,
                                       iteration);
      for (int k2 = 1; k2 < 64 / kMmaK; ++k2) {
        sm100_detail::tcgen05_mma_f16<1>(
            tensor_address, make_desc(a_smem + k2 * 32),
            make_desc(b_smem + k2 * 32), instruction_desc, 1);
      }
      for (int k1 = 1; k1 < kBlockK / 64; ++k1) {
        for (int k2 = 0; k2 < 64 / kMmaK; ++k2) {
          sm100_detail::tcgen05_mma_f16<1>(
              tensor_address, make_desc(a_smem + k1 * kBlockM * 128 + k2 * 32),
              make_desc(b_smem + k1 * kBlockN * 128 + k2 * 32),
              instruction_desc, 1);
        }
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                   "cluster.b64 "
                   "[%0];" ::"r"(mma_mbarrier)
                   : "memory");
    }
  };

  const int iterations = n / kBlockK;
  for (int i = 0; i < kNumStages - 1; ++i)
    load(i);
  for (int iteration = 0; iteration < iterations - kNumStages + 1;
       ++iteration) {
    load(iteration + kNumStages - 1);
    compute(iteration);
    sm100_detail::mbarrier_wait(mma_mbarrier, mma_phase);
    mma_phase ^= 1;
  }
  for (int iteration = iterations - kNumStages + 1; iteration < iterations;
       ++iteration) {
    compute(iteration);
    sm100_detail::mbarrier_wait(mma_mbarrier, mma_phase);
    mma_phase ^= 1;
  }

  asm volatile("tcgen05.fence::after_thread_sync;");
  for (int output_tile = 0; output_tile < kBlockN / 8; ++output_tile) {
    float values[8];
    const int address = tensor_address +
                        ((warp_id * sm100_detail::kWarpSize) << 16) +
                        output_tile * 8;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]),
                   "=f"(values[3]), "=f"(values[4]), "=f"(values[5]),
                   "=f"(values[6]), "=f"(values[7])
                 : "r"(address));
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    __nv_bfloat162 converted[4];
    for (int i = 0; i < 4; ++i) {
      converted[i] = __float22bfloat162_rn({values[i * 2], values[i * 2 + 1]});
    }
    // Row-major view of C^T maps directly to this column-major C buffer.
    __nv_bfloat16 *output =
        c + (block_row + tid) * n + block_col + output_tile * 8;
    reinterpret_cast<int4 *>(output)[0] =
        reinterpret_cast<int4 *>(converted)[0];
  }

  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(
                     tensor_address),
                 "r"(kBlockN));
  }
  (void)lane_id;
}

void encode_map(CUtensorMap *map, const __nv_bfloat16 *pointer, int n,
                int shared_height) {
  constexpr uint32_t rank = 3;
  const uint64_t global_dimensions[rank] = {64, static_cast<uint64_t>(n),
                                            static_cast<uint64_t>(n / 64)};
  const uint64_t global_strides[rank - 1] = {
      static_cast<uint64_t>(n) * sizeof(__nv_bfloat16), 128};
  const uint32_t box_dimensions[rank] = {64,
                                         static_cast<uint32_t>(shared_height),
                                         static_cast<uint32_t>(kBlockK / 64)};
  const uint32_t element_strides[rank] = {1, 1, 1};
  GEMM_Y_CU_CHECK(cuTensorMapEncodeTiled(
      map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
      const_cast<__nv_bfloat16 *>(pointer), global_dimensions, global_strides,
      box_dimensions, element_strides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

} // namespace

void k_tcgen05x::operator()(GemmArgs<__nv_bfloat16> args,
                            cudaStream_t stream) const {
  const int n = args.C.rows;
  if (args.C.rows != args.C.cols || args.A.rows != n || args.A.cols != n ||
      args.B.rows != n || args.B.cols != n || args.A.ld != n ||
      args.B.ld != n || args.C.ld != n || n < kBlockN || n % kBlockM != 0 ||
      n % kBlockN != 0 || n % kBlockK != 0) {
    std::fprintf(
        stderr,
        "k_tcgen05x requires aligned square contiguous matrices; N=%d\n", n);
    std::abort();
  }

  CUtensorMap a_map{};
  CUtensorMap b_map{};
  // A row-major view is current B^T; B row-major view is current A^T.
  encode_map(&a_map, args.B.ptr, n, kBlockM);
  encode_map(&b_map, args.A.ptr, n, kBlockN);

  constexpr std::size_t shared_bytes = static_cast<std::size_t>(
      (kBlockM + kBlockN) * kBlockK * kNumStages * sizeof(__nv_bfloat16));
  const int grid = (n / kBlockM) * (n / kBlockN);
  auto kernel = tcgen05_gemm_kernel;
  if (shared_bytes > 48'000) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_bytes)));
  }
  kernel<<<grid, kThreads, shared_bytes, stream>>>(a_map, b_map, args.C.ptr, n);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace gemm_y
