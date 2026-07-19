// cublas_gemm.h — thin wrapper around cublasGemmEx.
//
// Implicit alpha=1, beta=0 (AGENTS.md non-goal: epilogue fusion). computeType
// is CUDA_R_32F for bf16/fp16 inputs (fp32 accumulation, cuBLAS default —
// natural ground truth for our kernels). For fp32 inputs, computeType is
// CUDA_R_32F (pedantic, CUDA cores).
//
// Stream binding is per-call (cublasSetStream), not on the handle — see
// ARD.md §5.5. Phase 1 uses stream = nullptr (legacy default stream).

#pragma once

#include "CudaCheck.h"
#include "CublasHandle.h"
#include "MatrixView.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace detail {

// Maps a host dtype T -> cublas compute type + data type.
// bf16/fp16 accumulate in fp32 (CUDA_R_32F); fp32 is pedantic (CUDA_R_32F).
template <typename T> struct CublasTypeMap;
template <> struct CublasTypeMap<__nv_bfloat16> {
    static constexpr cudaDataType_t data_type = CUDA_R_16BF;
    static constexpr cudaDataType_t compute_type = CUDA_R_32F;
};
template <> struct CublasTypeMap<__half> {
    static constexpr cudaDataType_t data_type = CUDA_R_16F;
    static constexpr cudaDataType_t compute_type = CUDA_R_32F;
};
template <> struct CublasTypeMap<float> {
    static constexpr cudaDataType_t data_type = CUDA_R_32F;
    static constexpr cudaDataType_t compute_type = CUDA_R_32F;
};

} // namespace detail

// C = A * B  (no transpose, no epilogue). ColMajor only in Phase 1.
// A is (m x k), B is (k x n), C is (m x n). All views must be ColMajor.
//
// Note: A and B are inputs (read-only in spirit) but are NOT const-ified
// here. cublas_gemm is a function template, and C++ template argument
// deduction does not consider implicit conversions — so
// MatrixView<T,S> -> MatrixView<const T,S> (via MatrixView's converting
// constructor) would fail deduction at every call site. The const
// contract is enforced at the GemmArgs level (NaiveGemm and future
// kernels take GemmArgs<T> with const A/B); cublas_gemm is the
// reference path and takes writable views for API simplicity.
template <typename T>
void cublas_gemm(CublasHandle& handle,
                 MatrixView<T, Space::Device> A,
                 MatrixView<T, Space::Device> B,
                 MatrixView<T, Space::Device> C,
                 cudaStream_t stream = nullptr) {
    // ColMajor: cuBLAS sees C = A * B with A (m x k), B (k x n), C (m x n).
    // cublasGemmEx args: transa=N, transb=N, m=C.rows, n=C.cols, k=A.cols.
    const int m = C.rows;
    const int n = C.cols;
    const int k = A.cols;

    // Sanity: shapes must be consistent. We abort (no exceptions across the
    // CUDA boundary, per AGENTS.md).
    if (A.rows != m || A.cols != k || B.rows != k || B.cols != n) {
        std::fprintf(stderr,
                     "cublas_gemm: shape mismatch. A(%dx%d) B(%dx%d) C(%dx%d); "
                     "expected A(%dx%d) B(%dx%d) C(%dx%d)\n",
                     A.rows, A.cols, B.rows, B.cols, C.rows, C.cols,
                     m, k, k, n, m, n);
        std::abort();
    }
    if (A.layout != Layout::ColMajor || B.layout != Layout::ColMajor ||
        C.layout != Layout::ColMajor) {
        // Phase 1 invariant: all views are ColMajor. The bench runner
        // guarantees this; a violation is a programming error, not a
        // runtime API contract. Debug-only assert (R19).
        GEMM_Y_ASSERT(A.layout == Layout::ColMajor &&
                      B.layout == Layout::ColMajor &&
                      C.layout == Layout::ColMajor,
                      "cublas_gemm: only ColMajor supported in Phase 1");
    }

    using TM = detail::CublasTypeMap<T>;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle.get(), stream));
    CUBLAS_CHECK(cublasGemmEx(
        handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A.ptr, TM::data_type, A.ld,
        B.ptr, TM::data_type, B.ld,
        &beta,
        C.ptr, TM::data_type, C.ld,
        TM::compute_type,
        CUBLAS_GEMM_DEFAULT));
}

} // namespace gemm_y
