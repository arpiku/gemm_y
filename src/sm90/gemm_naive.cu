// gemm_naive.cu — Hopper (sm_90) naive GEMM kernel.

#include "gemm_naive.cuh"

#include "bench/GemmArgs.h"
#include "CudaCheck.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace detail {

// Device kernel: raw pointers + dimension ints. No MatrixView, no project
// includes. C = A(M×K) × B(K×N) = C(M×N), ColMajor.
template <typename T>
__global__ void naive_gemm_kernel(const T* A, const T* B, T* C,
                                  int M, int N, int K,
                                  int ldA, int ldB, int ldC) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float a = static_cast<float>(A[i + k * ldA]);
        const float b = static_cast<float>(B[k + j * ldB]);
        acc += a * b;
    }
    C[i + j * ldC] = static_cast<T>(acc);
}

} // namespace detail

template <typename T>
void NaiveGemm<T>::operator()(GemmArgs<T> args, cudaStream_t stream) const {
    // ColMajor is the only layout (enforced structurally by Matrix::alloc
    // setting ld = rows); the kernel hardcodes ColMajor addressing.
    constexpr int kBlock = 16;
    const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
    const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(kBlock, kBlock, 1);
    detail::naive_gemm_kernel<T><<<grid, block, 0, stream>>>(
        args.A.ptr, args.B.ptr, args.C.ptr,
        args.C.rows, args.C.cols, args.A.cols,
        args.A.ld, args.B.ld, args.C.ld);
}

template struct NaiveGemm<__nv_bfloat16>;
template struct NaiveGemm<__half>;
template struct NaiveGemm<float>;

} // namespace gemm_y
