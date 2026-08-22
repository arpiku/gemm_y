// gemm_bf16_k1_t.cu — templated single-buffered shared-memory bf16 GEMM.
//
// The initial family keeps the 16x16 (256-thread) block fixed while exposing
// the cooperative M/N/K tile as compile-time parameters. A is laid out as
// [k][row] and B as [column][k], preserving the coalesced loads of k1_smem.

#include "gemm_bf16.cuh"

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;

template <int TileM, int TileN, int TileK>
__global__ void k1_t_gemm_kernel(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                                 __nv_bfloat16 *C, int M, int N, int K, int ldA,
                                 int ldB, int ldC) {
  static_assert(TileM > 0 && TileN > 0 && TileK > 0);
  static_assert(TileM % kThreadsX == 0 && TileN % kThreadsY == 0);

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16 *const As = smem;
  __nv_bfloat16 *const Bs = As + TileM * TileK;

  constexpr int kPerThreadM = TileM / kThreadsX;
  constexpr int kPerThreadN = TileN / kThreadsY;
  constexpr int kThreads = kThreadsX * kThreadsY;

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * kThreadsX + tx;
  const int block_row = static_cast<int>(blockIdx.x) * TileM;
  const int block_col = static_cast<int>(blockIdx.y) * TileN;
  float accum[kPerThreadM][kPerThreadN] = {};

  for (int k0 = 0; k0 < K; k0 += TileK) {
    for (int index = tid; index < TileM * TileK; index += kThreads) {
      const int kk = index / TileM;
      const int row = index % TileM;
      const int global_row = block_row + row;
      const int global_k = k0 + kk;
      As[kk * TileM + row] = (global_row < M && global_k < K)
                                 ? A[global_row + global_k * ldA]
                                 : __nv_bfloat16(0.0f);
    }

    for (int index = tid; index < TileN * TileK; index += kThreads) {
      const int col = index / TileK;
      const int kk = index % TileK;
      const int global_col = block_col + col;
      const int global_k = k0 + kk;
      Bs[col * TileK + kk] = (global_col < N && global_k < K)
                                 ? B[global_k + global_col * ldB]
                                 : __nv_bfloat16(0.0f);
    }
    __syncthreads();

    for (int kk = 0; kk < TileK && k0 + kk < K; ++kk) {
      float a[kPerThreadM];
      float b[kPerThreadN];
      for (int mi = 0; mi < kPerThreadM; ++mi) {
        a[mi] = static_cast<float>(As[kk * TileM + tx * kPerThreadM + mi]);
      }
      for (int ni = 0; ni < kPerThreadN; ++ni) {
        b[ni] = static_cast<float>(Bs[(ty * kPerThreadN + ni) * TileK + kk]);
      }
      for (int mi = 0; mi < kPerThreadM; ++mi) {
        for (int ni = 0; ni < kPerThreadN; ++ni) {
          accum[mi][ni] += a[mi] * b[ni];
        }
      }
    }
    __syncthreads();
  }

  for (int mi = 0; mi < kPerThreadM; ++mi) {
    const int row = block_row + tx * kPerThreadM + mi;
    for (int ni = 0; ni < kPerThreadN; ++ni) {
      const int col = block_col + ty * kPerThreadN + ni;
      if (row < M && col < N) {
        C[row + col * ldC] = static_cast<__nv_bfloat16>(accum[mi][ni]);
      }
    }
  }
}

} // namespace

template <int TileM, int TileN, int TileK>
void k1_t<TileM, TileN, TileK>::operator()(GemmArgs<__nv_bfloat16> args,
                                           cudaStream_t stream) const {
  static_assert(TileM > 0 && TileN > 0 && TileK > 0);
  static_assert(TileM % kThreadsX == 0 && TileN % kThreadsY == 0);
  constexpr std::size_t shared_bytes =
      static_cast<std::size_t>(TileM * TileK + TileN * TileK) *
      sizeof(__nv_bfloat16);
  const int grid_x = (args.C.rows + TileM - 1) / TileM;
  const int grid_y = (args.C.cols + TileN - 1) / TileN;
  k1_t_gemm_kernel<TileM, TileN, TileK>
      <<<dim3(grid_x, grid_y, 1), dim3(kThreadsX, kThreadsY, 1), shared_bytes,
         stream>>>(args.A.ptr, args.B.ptr, args.C.ptr, args.C.rows, args.C.cols,
                   args.A.cols, args.A.ld, args.B.ld, args.C.ld);
  CUDA_CHECK_LAST_ERROR();
}

template struct k1_t<64, 64, 16>;
template struct k1_t<32, 64, 16>;
template struct k1_t<64, 32, 16>;
template struct k1_t<64, 64, 8>;

} // namespace gemm_y
