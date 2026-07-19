// gemm_bf16_naive.cu — Blackwell (sm_120) naive bf16 GEMM kernel.
//
// Triple-loop, 1 thread per C[i][j], inner k loop. ld-aware (reads via
// ptr + i + k*ld). Accumulate in fp32, cast back to bf16 on write.
//
// Perf-irrelevant — exists only to validate the harness end-to-end.
// The struct declaration lives in the matching .cuh so main.cpp (a C++
// TU) can register it with the Profiler.

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
    // 1 thread per C element. Grid covers (rows, cols).
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= C.rows || j >= C.cols) return;

    float acc = 0.0f;
    // ColMajor: A(i,k) = A.ptr[i + k*A.ld]; B(k,j) = B.ptr[k + j*B.ld].
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
    // Debug-only ColMajor invariant. The kernel hardcodes ColMajor
    // addressing; a RowMajor view would produce wrong results silently.
    // One assert per launch (host-side), zero Release cost.
    GEMM_Y_ASSERT(args.A.layout == Layout::ColMajor &&
                  args.B.layout == Layout::ColMajor &&
                  args.C.layout == Layout::ColMajor,
                  "NaiveGemm assumes ColMajor inputs");
    // 16x16 threads per block. Grid covers the C matrix.
    constexpr int kBlock = 16;
    const int grid_x = (args.C.rows + kBlock - 1) / kBlock;
    const int grid_y = (args.C.cols + kBlock - 1) / kBlock;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(kBlock, kBlock, 1);
    detail::naive_gemm_kernel<T><<<grid, block, 0, nullptr>>>(
        args.A, args.B, args.C);
}

// Explicit instantiation for the bf16 dtype. Instantiating the struct
// emits all member definitions (including operator()).
template struct NaiveGemm<__nv_bfloat16>;

} // namespace gemm_y
