// gemm_bf16_k1_t_x.cu — isolated k1_t code-generation experiment.
//
// This keeps k1_t<64,64,16>'s tile, mapping, and shared-memory layout. The
// This version uses explicit __fmaf_rn accumulation, targeted loop
// unrolling, and compile-time shift/mask index lowering for the fixed tile.
//
// The focused CUDA-event comparison passed correctness, but did not improve
// the baseline: k1_t_x was 6.2 us versus 6.0 us at N=32, tied at 14.4 us
// versus 14.4 us at N=96, and tied at 18.5 us versus 18.5 us at N=128.
// cuobjdump reported 40 registers for k1_t_x versus 38 for k1_t, with both
// using 1024 bytes of shared memory. It remains preserved as a non-production
// reference experiment; it is not part of the smoke or production registry.

#include "gemm_bf16.cuh"

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
constexpr int kThreads = kThreadsX * kThreadsY;
constexpr int kPerThreadM = kTileM / kThreadsX;
constexpr int kPerThreadN = kTileN / kThreadsY;

static_assert(kTileM > 0 && kTileN > 0 && kTileK > 0);
static_assert((kTileM & (kTileM - 1)) == 0 && (kTileN & (kTileN - 1)) == 0 &&
                  (kTileK & (kTileK - 1)) == 0,
              "k1_t_x requires power-of-two tile dimensions");
static_assert(kTileM % kThreadsX == 0 && kTileN % kThreadsY == 0);

__global__ void k1_t_x_gemm_kernel(const __nv_bfloat16 *A,
                                   const __nv_bfloat16 *B, __nv_bfloat16 *C,
                                   int M, int N, int K, int ldA, int ldB,
                                   int ldC) {
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16 *const As = smem;
  __nv_bfloat16 *const Bs = As + kTileM * kTileK;

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * kThreadsX + tx;
  const int block_row = static_cast<int>(blockIdx.x) * kTileM;
  const int block_col = static_cast<int>(blockIdx.y) * kTileN;
  float accum[kPerThreadM][kPerThreadN] = {};

  for (int k0 = 0; k0 < K; k0 += kTileK) {
    for (int index = tid; index < kTileM * kTileK; index += kThreads) {
      const int kk = index >> 6;
      const int row = index & 63;
      const int global_row = block_row + row;
      const int global_k = k0 + kk;
      As[(kk << 6) + row] = (global_row < M && global_k < K)
                                ? A[global_row + global_k * ldA]
                                : __nv_bfloat16(0.0f);
    }

    for (int index = tid; index < kTileN * kTileK; index += kThreads) {
      const int col = index >> 4;
      const int kk = index & 15;
      const int global_col = block_col + col;
      const int global_k = k0 + kk;
      Bs[(col << 4) + kk] = (global_col < N && global_k < K)
                                ? B[global_k + global_col * ldB]
                                : __nv_bfloat16(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK && k0 + kk < K; ++kk) {
      float a[kPerThreadM];
      float b[kPerThreadN];
#pragma unroll
      for (int mi = 0; mi < kPerThreadM; ++mi) {
        a[mi] = static_cast<float>(As[(kk << 6) + (tx << 2) + mi]);
      }
#pragma unroll
      for (int ni = 0; ni < kPerThreadN; ++ni) {
        b[ni] = static_cast<float>(Bs[((ty << 2) + ni) * kTileK + kk]);
      }
#pragma unroll
      for (int mi = 0; mi < kPerThreadM; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kPerThreadN; ++ni) {
          accum[mi][ni] = __fmaf_rn(a[mi], b[ni], accum[mi][ni]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < kPerThreadM; ++mi) {
    const int row = block_row + (tx << 2) + mi;
#pragma unroll
    for (int ni = 0; ni < kPerThreadN; ++ni) {
      const int col = block_col + (ty << 2) + ni;
      if (row < M && col < N) {
        C[row + col * ldC] = static_cast<__nv_bfloat16>(accum[mi][ni]);
      }
    }
  }
}

} // namespace

void k1_t_x::operator()(GemmArgs<__nv_bfloat16> args,
                        cudaStream_t stream) const {
  constexpr std::size_t shared_bytes =
      static_cast<std::size_t>(kTileM * kTileK + kTileN * kTileK) *
      sizeof(__nv_bfloat16);
  const int grid_x = (args.C.rows + kTileM - 1) / kTileM;
  const int grid_y = (args.C.cols + kTileN - 1) / kTileN;
  k1_t_x_gemm_kernel<<<dim3(grid_x, grid_y, 1), dim3(kThreadsX, kThreadsY, 1),
                       shared_bytes, stream>>>(
      args.A.ptr, args.B.ptr, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
      args.A.ld, args.B.ld, args.C.ld);
  CUDA_CHECK_LAST_ERROR();
}

} // namespace gemm_y
