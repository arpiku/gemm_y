// gemm_bf16_k0_ldg.cu — Blackwell (sm_120) explicit-read-only-load experiment.
//
// This keeps the one-thread-per-output baseline unchanged except for explicit
// __ldg loads of the read-only A and B operands.
//
// Layering (ARD §3): the device __global__ function takes raw pointers +
// dimension ints (pure CUDA, no project includes). The operator() functor
// is the thin adapter that unpacks GemmArgs<T> into the raw-pointer call.

#include "gemm_bf16.cuh"

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

// Device kernel: raw pointers + dimension ints. C = A(M×K) × B(K×N) = C(M×N),
// ColMajor.
__global__ void k0_ldg_gemm_kernel(const __nv_bfloat16 *A,
                                   const __nv_bfloat16 *B, __nv_bfloat16 *C,
                                   int M, int N, int K, int ldA, int ldB,
                                   int ldC) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= N)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const __nv_bfloat16 a_bf16 = __ldg(A + i + k * ldA);
    const __nv_bfloat16 b_bf16 = __ldg(B + k + j * ldB);
    const float a = static_cast<float>(a_bf16);
    const float b = static_cast<float>(b_bf16);
    acc += a * b;
  }
  C[i + j * ldC] = static_cast<__nv_bfloat16>(acc);
}

} // namespace

void k0_ldg::operator()(GemmArgs<__nv_bfloat16> args,
                        cudaStream_t stream) const {
  // ColMajor is the only layout (enforced structurally by Matrix::alloc
  // setting ld = rows); the kernel hardcodes ColMajor addressing.
  constexpr int kBlock = 16;
  const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
  const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
  dim3 grid(grid_x, grid_y, 1);
  dim3 block(kBlock, kBlock, 1);
  k0_ldg_gemm_kernel<<<grid, block, 0, stream>>>(
      args.A.ptr, args.B.ptr, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
      args.A.ld, args.B.ld, args.C.ld);
}

} // namespace gemm_y
