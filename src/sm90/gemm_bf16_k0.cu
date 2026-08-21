// gemm_bf16_k0.cu — Hopper (sm_90) k0 dummy kernel definition.
//
// Verbatim copy of NaiveGemm<__nv_bfloat16>::operator() — same triple-loop
// device kernel, same launch config. Workflow development only; no
// optimization. The point is to exercise the registration / measurement loop
// with a real (if perf-identical) kernel before investing in tiled TC kernels.
// See ARD §16 and TODO 2C.1.
//
// Identical to src/sm120/gemm_bf16_k0.cu. Per AGENTS.md and ARD.md §8,
// arch-specific code lives in separate .cu files (no #ifdef branches);
// CMake compiles only the directory matching GEMM_Y_CUDA_ARCH. Divergence
// between arches begins with tensor-core kernels.

#include "gemm_bf16.cuh"

#include "CudaCheck.h"
#include "bench/GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

// Anonymous namespace so this device function does not collide with
// detail::naive_gemm_kernel in gemm_naive.cu (same body, distinct symbol).
// Device kernel: raw pointers + dimension ints. C = A(M×K) × B(K×N) = C(M×N),
// ColMajor.
__global__ void k0_gemm_kernel(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                               __nv_bfloat16 *C, int M, int N, int K, int ldA,
                               int ldB, int ldC) {
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

void k0::operator()(GemmArgs<__nv_bfloat16> args, cudaStream_t stream) const {
  // ColMajor is the only layout (enforced structurally by Matrix::alloc
  // setting ld = rows); the kernel hardcodes ColMajor addressing.
  constexpr int kBlock = 16;
  const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
  const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
  dim3 grid(grid_x, grid_y, 1);
  dim3 block(kBlock, kBlock, 1);
  k0_gemm_kernel<<<grid, block, 0, stream>>>(
      args.A.ptr, args.B.ptr, args.C.ptr, args.C.rows, args.C.cols, args.A.cols,
      args.A.ld, args.B.ld, args.C.ld);
}

} // namespace gemm_y
