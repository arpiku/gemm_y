// gemm_bf16_k2_tma_mma.cu — sm120 BF16 TMA + explicit MMA candidate.
//
// TMA stages a 64x16 A tile and a 16x64 B tile. Each warp computes a
// 16x32 output tile using four explicit m16n8k16 BF16 MMA operations.

#include "gemm_bf16.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsM = 4;

constexpr int kWarpTileM = 16;
constexpr int kWarpTileN = 32;
constexpr int kMmaN = 8;
constexpr int kMmaCountN = kWarpTileN / kMmaN;

constexpr int kBytesPerElement = sizeof(__nv_bfloat16);
constexpr int kBytesPerStage =
    (kTileM * kTileK + kTileK * kTileN) * kBytesPerElement;

__device__ inline void mbarrier_init(uint64_t *barrier) {
  const unsigned address = __cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(address));
}

__device__ inline void mbarrier_wait(uint64_t *barrier, int phase) {
  const unsigned address = __cvta_generic_to_shared(barrier);
  constexpr unsigned kWaitHint = 0x989680;
  asm volatile(
      "{\n"
      " .reg .pred complete;\n"
      "wait_loop:\n"
      " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, "
      "[%0], %1, %2;\n"
      " @complete bra.uni wait_done;\n"
      " bra.uni wait_loop;\n"
      "wait_done:\n"
      "}\n"
      :
      : "r"(address), "r"(phase), "r"(kWaitHint));
}

__device__ inline void tma_load_2d(void *destination, const CUtensorMap *map,
                                   int x, int y, uint64_t *barrier) {
  const unsigned destination_address = __cvta_generic_to_shared(destination);
  const unsigned barrier_address = __cvta_generic_to_shared(barrier);
  const int coordinates[2] = {x, y};
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile."
               "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :
               : "r"(destination_address), "l"(map), "r"(coordinates[0]),
                 "r"(coordinates[1]), "r"(barrier_address)
               : "memory");
}

// Load a 16x16 BF16 A tile and a 16x8 BF16 B tile into the register layout
// consumed by mma.sync.aligned.m16n8k16. The shared tiles are column-major.
__device__ inline void load_mma_operands(const __nv_bfloat16 *As,
                                         const __nv_bfloat16 *Bs, int warp_m,
                                         int column, unsigned (&a)[4],
                                         unsigned (&b)[2]) {
  const int tid =
      static_cast<int>(threadIdx.y) * kThreadsX + static_cast<int>(threadIdx.x);
  const int lane = tid & (kWarpSize - 1);
  const int lane_row = lane & 7;
  const int lane_group = lane >> 3;

  // Each ldmatrix lane supplies a 16-byte-aligned row address. For A, the
  // transpose form converts the column-major shared tile into the row-major
  // MMA operand. B is already column-major for the row.col MMA instruction.
  const auto *a_address = As + warp_m * kWarpTileM +
                          (lane_row + (lane_group >> 1) * 8) * kTileM +
                          (lane_group & 1) * 8;
  const auto *b_address = Bs + (column + lane_row) * kTileK + lane_group * 8;
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(static_cast<unsigned>(__cvta_generic_to_shared(a_address))));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(b[0]), "=r"(b[1])
      : "r"(static_cast<unsigned>(__cvta_generic_to_shared(b_address))));
}

__device__ inline void mma_m16n8k16(const unsigned (&a)[4],
                                    const unsigned (&b)[2], float (&d)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

__global__ void
k2_tma_mma_gemm_kernel(const __grid_constant__ CUtensorMap a_map,
                       const __grid_constant__ CUtensorMap b_map,
                       __nv_bfloat16 *C, int M, int N, int K, int ldC) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto *const shared = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *const As = shared;
  auto *const Bs = As + kTileM * kTileK;
  __shared__ uint64_t barrier;

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * kThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id % kWarpsM;
  const int warp_n = warp_id / kWarpsM;
  const int block_row = static_cast<int>(blockIdx.x) * kTileM;
  const int block_col = static_cast<int>(blockIdx.y) * kTileN;

  float accum[kMmaCountN][4] = {};
  if (tx == 0 && ty == 0) {
    mbarrier_init(&barrier);
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
  }
  __syncthreads();

  int phase = 0;
  for (int k0 = 0; k0 < K; k0 += kTileK) {
    if (tx == 0 && ty == 0) {
      const unsigned address = __cvta_generic_to_shared(&barrier);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
                   "[%0], %1;"
                   :
                   : "r"(address), "r"(kBytesPerStage)
                   : "memory");
      tma_load_2d(As, &a_map, block_row, k0, &barrier);
      tma_load_2d(Bs, &b_map, k0, block_col, &barrier);
    }

    mbarrier_wait(&barrier, phase);
    phase ^= 1;

    unsigned a[4];
    for (int mma_n = 0; mma_n < kMmaCountN; ++mma_n) {
      unsigned b[2];
      load_mma_operands(As, Bs, warp_m, warp_n * kWarpTileN + mma_n * kMmaN, a,
                        b);
      mma_m16n8k16(a, b, accum[mma_n]);
    }
    __syncthreads();
  }

  const int lane = tid & (kWarpSize - 1);
  const int row_base = lane / 4;
  const int col_base = (lane % 4) * 2;
  for (int mma_n = 0; mma_n < kMmaCountN; ++mma_n) {
    const int row = block_row + warp_m * kWarpTileM + row_base;
    const int col = block_col + warp_n * kWarpTileN + mma_n * kMmaN + col_base;
    if (row < M && col + 1 < N) {
      C[row + col * ldC] = static_cast<__nv_bfloat16>(accum[mma_n][0]);
      C[row + (col + 1) * ldC] = static_cast<__nv_bfloat16>(accum[mma_n][1]);
      if (row + 8 < M) {
        C[row + 8 + col * ldC] = static_cast<__nv_bfloat16>(accum[mma_n][2]);
        C[row + 8 + (col + 1) * ldC] =
            static_cast<__nv_bfloat16>(accum[mma_n][3]);
      }
    }
  }
}

void encode_map(CUtensorMap *map, const __nv_bfloat16 *pointer, int rows,
                int cols, int ld, int box_rows, int box_cols) {
  constexpr uint32_t rank = 2;
  const uint64_t dimensions[rank] = {static_cast<uint64_t>(rows),
                                     static_cast<uint64_t>(cols)};
  const uint64_t strides[rank - 1] = {static_cast<uint64_t>(ld) *
                                      sizeof(__nv_bfloat16)};
  const uint32_t box[rank] = {static_cast<uint32_t>(box_rows),
                              static_cast<uint32_t>(box_cols)};
  const uint32_t element_strides[rank] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank,
      const_cast<__nv_bfloat16 *>(pointer), dimensions, strides, box,
      element_strides, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) {
    const char *message = nullptr;
    (void)cuGetErrorString(status, &message);
    std::fprintf(stderr, "k2_tma_mma: cuTensorMapEncodeTiled failed: %s\n",
                 message == nullptr ? "unknown error" : message);
    std::abort();
  }
}

} // namespace

bool k2_tma_mma::supports(const GemmArgs<__nv_bfloat16> &args) {
  constexpr int kTmaStrideAlignmentBytes = 16;
  const bool aligned = (args.A.ld * static_cast<int>(sizeof(__nv_bfloat16))) %
                               kTmaStrideAlignmentBytes ==
                           0 &&
                       (args.B.ld * static_cast<int>(sizeof(__nv_bfloat16))) %
                               kTmaStrideAlignmentBytes ==
                           0;
  return args.C.rows % kTileM == 0 && args.C.cols % kTileN == 0 &&
         args.A.cols % kTileK == 0 && args.A.rows == args.C.rows &&
         args.B.rows == args.A.cols && args.B.cols == args.C.cols && aligned;
}

void k2_tma_mma::operator()(GemmArgs<__nv_bfloat16> args,
                            cudaStream_t stream) const {
  if (!supports(args)) {
    std::fprintf(stderr,
                 "k2_tma_mma: unsupported shape or stride; profiler should "
                 "skip this invocation\n");
    std::abort();
  }

  CUtensorMap a_map{};
  CUtensorMap b_map{};
  encode_map(&a_map, args.A.ptr, args.A.rows, args.A.cols, args.A.ld, kTileM,
             kTileK);
  encode_map(&b_map, args.B.ptr, args.B.rows, args.B.cols, args.B.ld, kTileK,
             kTileN);

  constexpr std::size_t shared_bytes =
      static_cast<std::size_t>(kTileM * kTileK + kTileK * kTileN) *
          sizeof(__nv_bfloat16) +
      static_cast<std::size_t>(kTileM * kTileN) * sizeof(float);
  const dim3 grid((args.C.rows + kTileM - 1) / kTileM,
                  (args.C.cols + kTileN - 1) / kTileN, 1);
  const dim3 block(kThreadsX, kThreadsY, 1);

  auto kernel = k2_tma_mma_gemm_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_bytes)));
  kernel<<<grid, block, shared_bytes, stream>>>(a_map, b_map, args.C.ptr,
                                                args.C.rows, args.C.cols,
                                                args.A.cols, args.C.ld);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace gemm_y
