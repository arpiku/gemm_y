// gemm_bf16_naive.cu — Hopper (sm_90) naive GEMM kernel.
//
// Identical to src/sm120/gemm_bf16_naive.cu. Per AGENTS.md and ARD.md §8,
// arch-specific code lives in separate .cu files (no #ifdef branches);
// CMake compiles only the directory matching GEMM_Y_CUDA_ARCH. Divergence
// between arches begins with tensor-core kernels.

#include "gemm_bf16_naive.cuh"

#include "bench/GemmArgs.h"
#include "CudaCheck.h"
#include "MatrixView.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace detail {

template <typename T>
__global__ void naive_gemm_kernel(MatrixView<const T, Space::Device> A,
                                  MatrixView<const T, Space::Device> B,
                                  MatrixView<T,       Space::Device> C) {
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
    C.ptr[i + j * C.ld] = static_cast<T>(acc);
}

} // namespace detail

template <typename T>
void NaiveGemm<T>::operator()(GemmArgs<T> args, cudaStream_t /*stream*/) const {
    // Debug-only ColMajor invariant. See sm120 variant for rationale;
    // duplicated here per AGENTS.md §8 (separate .cu files, no #ifdef).
    GEMM_Y_ASSERT(args.A.layout == Layout::ColMajor &&
                  args.B.layout == Layout::ColMajor &&
                  args.C.layout == Layout::ColMajor,
                  "NaiveGemm assumes ColMajor inputs");
    constexpr int kBlock = 16;
    const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
    const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(kBlock, kBlock, 1);
    detail::naive_gemm_kernel<T><<<grid, block, 0, nullptr>>>(
        args.A, args.B, args.C);
}

template struct NaiveGemm<__nv_bfloat16>;
template struct NaiveGemm<__half>;
template struct NaiveGemm<float>;

} // namespace gemm_y
