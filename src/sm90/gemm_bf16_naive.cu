// gemm_bf16_naive.cu — Hopper (sm_90) naive bf16 GEMM kernel.
//
// Identical to src/sm120/gemm_bf16_naive.cu in Phase 1. Per AGENTS.md and
// ARD.md §8, arch-specific code lives in separate .cu files (no #ifdef
// branches); CMake compiles only the directory matching GEMM_Y_CUDA_ARCH.
// Divergence between arches begins with tensor-core kernels in Phase 2.

#include "gemm_bf16_naive.cuh"

#include "bench/GemmArgs.h"
#include "MatrixView.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace detail {

template <typename T>
__global__ void naive_gemm_kernel(MatrixView<T, Space::Device> A,
                                  MatrixView<T, Space::Device> B,
                                  MatrixView<T, Space::Device> C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= C.rows || j >= C.cols) return;

    float acc = 0.0f;
    for (int k = 0; k < A.cols; ++k) {
        const float a = static_cast<float>(A.ptr[i + k * A.ld]);
        const float b = static_cast<float>(B.ptr[k + j * B.ld]);
        acc += a * b;
    }
    C.ptr[i + j * C.ld] = static_cast<T>(acc);
}

} // namespace detail

template <typename T>
void NaiveGemm<T>::operator()(GemmArgs<T> args, cudaStream_t /*stream*/) const {
    constexpr int kBlock = 16;
    const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
    const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(kBlock, kBlock, 1);
    detail::naive_gemm_kernel<T><<<grid, block, 0, nullptr>>>(
        args.A, args.B, args.C);
}

template struct NaiveGemm<__nv_bfloat16>;

} // namespace gemm_y
