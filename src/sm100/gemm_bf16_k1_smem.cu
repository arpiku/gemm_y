// gemm_bf16_k1_smem.cu — Blackwell shared-memory tiled bf16 GEMM.
//
// Fixed configuration: 64x64 output tile, 16-wide K tile, and a 16x16
// (256-thread) block. Each thread computes a 4x4 output tile. Shared memory
// is single-buffered; A is stored as [k][row] and B as [col][k] so cooperative
// loads traverse contiguous global elements for both operands.

#include "gemm_bf16.cuh"

#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr int kPerThreadM = kTileM / kThreadsX;
constexpr int kPerThreadN = kTileN / kThreadsY;

__global__ void k1_smem_gemm_kernel(const __nv_bfloat16 *A,
                                    const __nv_bfloat16 *B, __nv_bfloat16 *C,
                                    int M, int N, int K, int ldA, int ldB,
                                    int ldC) {
  __shared__ __nv_bfloat16 As[kTileK][kTileM];
  __shared__ __nv_bfloat16 Bs[kTileN][kTileK];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int tid = ty * kThreadsX + tx;
  const int block_row = static_cast<int>(blockIdx.x) * kTileM;
  const int block_col = static_cast<int>(blockIdx.y) * kTileN;

  float accum[kPerThreadM][kPerThreadN] = {};

  for (int k0 = 0; k0 < K; k0 += kTileK) {
    // A is transposed in shared memory so the linear cooperative load walks
    // adjacent rows in a column-major global matrix.
    for (int index = tid; index < kTileM * kTileK;
         index += kThreadsX * kThreadsY) {
      const int kk = index / kTileM;
      const int row = index % kTileM;
      const int global_row = block_row + row;
      const int global_k = k0 + kk;
      As[kk][row] = (global_row < M && global_k < K)
                        ? A[global_row + global_k * ldA]
                        : __nv_bfloat16(0.0f);
    }

    // B is stored as [column][k], matching its column-major global layout.
    for (int index = tid; index < kTileN * kTileK;
         index += kThreadsX * kThreadsY) {
      const int col = index / kTileK;
      const int kk = index % kTileK;
      const int global_col = block_col + col;
      const int global_k = k0 + kk;
      Bs[col][kk] = (global_col < N && global_k < K)
                        ? B[global_k + global_col * ldB]
                        : __nv_bfloat16(0.0f);
    }
    __syncthreads();

    for (int kk = 0; kk < kTileK && k0 + kk < K; ++kk) {
      float a[kPerThreadM];
      float b[kPerThreadN];
      for (int mi = 0; mi < kPerThreadM; ++mi) {
        a[mi] = static_cast<float>(As[kk][tx * kPerThreadM + mi]);
      }
      for (int ni = 0; ni < kPerThreadN; ++ni) {
        b[ni] = static_cast<float>(Bs[ty * kPerThreadN + ni][kk]);
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

void k1_smem::operator()(GemmArgs<__nv_bfloat16> args,
                         cudaStream_t stream) const {
  const int grid_x = (args.C.rows + kTileM - 1) / kTileM;
  const int grid_y = (args.C.cols + kTileN - 1) / kTileN;
  const dim3 grid(grid_x, grid_y, 1);
  const dim3 block(kThreadsX, kThreadsY, 1);
  k1_smem_gemm_kernel<<<grid, block, 0, stream>>>(
      args.A.ptr, args.B.ptr, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
      args.A.ld, args.B.ld, args.C.ld);
}

} // namespace gemm_y
