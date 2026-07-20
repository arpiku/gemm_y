// gemm_bf16_k0.cu — Hopper (sm_90) k0 dummy kernel definition.
//
// Verbatim copy of NaiveGemm<__nv_bfloat16>::operator() — same triple-loop
// device kernel, same launch config. Workflow development only; no
// optimization. The point is to exercise the registration / measurement
// loop with a real (if perf-identical) kernel before investing in tiled
// TC kernels. See ARD §16 and TODO 2C.1.
//
// Identical to src/sm120/gemm_bf16_k0.cu. Per AGENTS.md and ARD.md §8,
// arch-specific code lives in separate .cu files (no #ifdef branches);
// CMake compiles only the directory matching GEMM_Y_CUDA_ARCH. Divergence
// between arches begins with tensor-core kernels.

#include "gemm_bf16.cuh"

#include "bench/GemmArgs.h"
#include "CudaCheck.h"
#include "MatrixView.h"
#include "cuda_compat.h"

namespace gemm_y {
namespace {

// Anonymous namespace so this device function does not collide with
// detail::naive_gemm_kernel in gemm_naive.cu (same body, distinct symbol).
__global__ void k0_gemm_kernel(MatrixView<const __nv_bfloat16, Space::Device> A,
                               MatrixView<const __nv_bfloat16, Space::Device> B,
                               MatrixView<__nv_bfloat16,       Space::Device> C) {
    // MatrixView used as POD descriptor — only ptr/rows/cols/ld are read.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= C.rows || j >= C.cols) return;

    float acc = 0.0f;
    for (int k = 0; k < A.cols; ++k) {
        const float a = static_cast<float>(A.ptr[i + k * A.ld]);
        const float b = static_cast<float>(B.ptr[k + j * B.ld]);
        acc += a * b;
    }
    C.ptr[i + j * C.ld] = static_cast<__nv_bfloat16>(acc);
}

} // namespace

void k0::operator()(GemmArgs<__nv_bfloat16> args,
                    cudaStream_t /*stream*/) const {
    // Debug-only ColMajor invariant (mirrors NaiveGemm).
    GEMM_Y_ASSERT(args.A.layout == Layout::ColMajor &&
                  args.B.layout == Layout::ColMajor &&
                  args.C.layout == Layout::ColMajor,
                  "k0 assumes ColMajor inputs");
    constexpr int kBlock = 16;
    const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
    const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(kBlock, kBlock, 1);
    k0_gemm_kernel<<<grid, block, 0, nullptr>>>(
        args.A, args.B, args.C);
}

} // namespace gemm_y
