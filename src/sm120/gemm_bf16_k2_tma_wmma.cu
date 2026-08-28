// gemm_bf16_k2_tma_wmma.cu — sm120 BF16 TMA + WMMA kernel.
//
// TMA moves one 64x16 A tile and one 16x64 B tile into shared memory. WMMA
// loads 16x16 warp fragments and performs BF16 MMA with FP32 accumulation.
// WMMA stores the FP32 fragments to shared memory before BF16 conversion to C.

#include "gemm_bf16.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <mma.h>

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
constexpr int kWarpsN = 2;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpTileN = kWarpsN * kWmmaN;

constexpr int kBytesPerElement = sizeof(__nv_bfloat16);
constexpr int kBytesPerStage =
    (kTileM * kTileK + kTileK * kTileN) * kBytesPerElement;

__device__ inline void mbarrier_init(uint64_t *barrier) {
  const unsigned address = __cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address));
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
      "}\n" ::"r"(address),
      "r"(phase), "r"(kWaitHint));
}

__device__ inline void tma_load_2d(void *destination, const CUtensorMap *map,
                                   int x, int y, uint64_t *barrier) {
  const unsigned destination_address = __cvta_generic_to_shared(destination);
  const unsigned barrier_address = __cvta_generic_to_shared(barrier);
  const int coordinates[2] = {x, y};
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile."
               "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
                   destination_address),
               "l"(map), "r"(coordinates[0]), "r"(coordinates[1]),
               "r"(barrier_address)
               : "memory");
}

__global__ void
k2_tma_wmma_gemm_kernel(const __grid_constant__ CUtensorMap a_map,
                        const __grid_constant__ CUtensorMap b_map,
                        __nv_bfloat16 *C, int M, int N, int K, int ldC) {

  extern __shared__ __align__(128) unsigned char storage[];
  auto *const shared = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *const As = shared;
  auto *const Bs = As + kTileM * kTileK;
  auto *const Cs = reinterpret_cast<float *>(Bs + kTileK * kTileN);
  __shared__ uint64_t barrier;

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * kThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id % kWarpsM;
  const int warp_n = warp_id / kWarpsM;
  const int block_row = static_cast<int>(blockIdx.x) * kTileM;
  const int block_col = static_cast<int>(blockIdx.y) * kTileN;

  using AFragment =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                             __nv_bfloat16, nvcuda::wmma::col_major>;
  using BFragment =
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                             __nv_bfloat16, nvcuda::wmma::col_major>;
  using CFragment = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM,
                                           kWmmaN, kWmmaK, float>;
  AFragment a_fragment;
  BFragment b_fragments[2];
  CFragment c_fragments[2];
  for (auto &fragment : c_fragments)
    nvcuda::wmma::fill_fragment(fragment, 0.0f);

  if (tx == 0 && ty == 0) {
    mbarrier_init(&barrier);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  int phase = 0;
  for (int k0 = 0; k0 < K; k0 += kTileK) {
    if (tx == 0 && ty == 0) {
      const unsigned address = __cvta_generic_to_shared(&barrier);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
                   "[%0], %1;" ::"r"(address),
                   "r"(kBytesPerStage)
                   : "memory");
      tma_load_2d(As, &a_map, block_row, k0, &barrier);
      tma_load_2d(Bs, &b_map, k0, block_col, &barrier);
    }

    mbarrier_wait(&barrier, phase);
    phase ^= 1;

    nvcuda::wmma::load_matrix_sync(a_fragment, As + warp_m * kWmmaM, kTileM);
    for (int fragment_n = 0; fragment_n < 2; ++fragment_n) {
      const int col = warp_n * kWarpTileN + fragment_n * kWmmaN;
      nvcuda::wmma::load_matrix_sync(b_fragments[fragment_n], Bs + col * kTileK,
                                     kTileK);
      nvcuda::wmma::mma_sync(c_fragments[fragment_n], a_fragment,
                             b_fragments[fragment_n], c_fragments[fragment_n]);
    }
    __syncthreads();
  }

  for (int fragment_n = 0; fragment_n < 2; ++fragment_n) {
    const int row = warp_m * kWmmaM;
    const int col = warp_n * kWarpTileN + fragment_n * kWmmaN;
    nvcuda::wmma::store_matrix_sync(Cs + row + col * kTileM,
                                    c_fragments[fragment_n], kTileM,
                                    nvcuda::wmma::mem_col_major);
  }
  __syncthreads();

  for (int index = tid; index < kTileM * kTileN;
       index += kThreadsX * kThreadsY) {
    const int row = index % kTileM;
    const int col = index / kTileM;
    C[block_row + row + (block_col + col) * ldC] =
        static_cast<__nv_bfloat16>(Cs[index]);
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
    std::fprintf(stderr, "k2_tma_wmma: cuTensorMapEncodeTiled failed: %s\n",
                 message == nullptr ? "unknown error" : message);
    std::abort();
  }
}

} // namespace

bool k2_tma_wmma::supports(const GemmArgs<__nv_bfloat16> &args) {
  constexpr int kTmaStrideAlignmentBytes = 16;
  const bool tma_stride_aligned =
      (args.A.ld * static_cast<int>(sizeof(__nv_bfloat16))) %
              kTmaStrideAlignmentBytes ==
          0 &&
      (args.B.ld * static_cast<int>(sizeof(__nv_bfloat16))) %
              kTmaStrideAlignmentBytes ==
          0;
  return args.C.rows % kTileM == 0 && args.C.cols % kTileN == 0 &&
         args.A.cols % kTileK == 0 && args.A.rows == args.C.rows &&
         args.B.rows == args.A.cols && args.B.cols == args.C.cols &&
         tma_stride_aligned;
}

void k2_tma_wmma::operator()(GemmArgs<__nv_bfloat16> args,
                             cudaStream_t stream) const {
  if (!supports(args)) {
    std::fprintf(stderr,
                 "k2_tma_wmma: unsupported shape or stride; profiler should "
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

  auto kernel = k2_tma_wmma_gemm_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(shared_bytes)));
  kernel<<<grid, block, shared_bytes, stream>>>(a_map, b_map, args.C.ptr,
                                                args.C.rows, args.C.cols,
                                                args.A.cols, args.C.ld);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace gemm_y
