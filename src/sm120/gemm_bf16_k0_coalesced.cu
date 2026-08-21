// gemm_bf16_k0_coalesced.cu — Blackwell (sm_120) mapping experiment.
//
// This starts from the original scalar k0 load path and changes only the
// thread/block mapping to make adjacent lanes access contiguous A elements.

#include "gemm_bf16.cuh"

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

__global__ void k0_coalesced_gemm_kernel(const __nv_bfloat16 *A,
                                         const __nv_bfloat16 *B,
                                         __nv_bfloat16 *C, int M, int N, int K,
                                         int ldA, int ldB, int ldC) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float a = static_cast<float>(A[i + k * ldA]);
    const float b = static_cast<float>(B[k + j * ldB]);
    acc += a * b;
  }
  C[i + j * ldC] = static_cast<__nv_bfloat16>(acc);
}

} // namespace

void k0_coalesced::operator()(GemmArgs<__nv_bfloat16> args,
                              cudaStream_t stream) const {
  constexpr int kBlockX = 32;
  constexpr int kBlockY = 8;
  const int grid_x = (args.C.rows + kBlockX - 1) / kBlockX;
  const int grid_y = (args.C.cols + kBlockY - 1) / kBlockY;
  dim3 grid(grid_x, grid_y, 1);
  dim3 block(kBlockX, kBlockY, 1);
  k0_coalesced_gemm_kernel<<<grid, block, 0, stream>>>(
      args.A.ptr, args.B.ptr, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
      args.A.ld, args.B.ld, args.C.ld);
}

} // namespace gemm_y
