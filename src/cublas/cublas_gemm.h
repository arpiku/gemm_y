// cublas_gemm.h — thin wrapper around cublasGemmEx.
//
// Implicit alpha=1, beta=0 (AGENTS.md non-goal: epilogue fusion). computeType
// is CUDA_R_32F for all paths (fp32 accumulation). The math mode is selected
// per dtype via CublasTypeMap<T>::math_mode: bf16/fp16 use CUBLAS_DEFAULT_MATH
// (tensor cores, fp32 accum); tfloat uses CUBLAS_TF32_CUBLAS_MATH (tf32 TC).
// The pedantic fp32 / CUDA-core path is dropped — see ARD §9.
//
// Stream binding is per-call (cublasSetStream), not on the handle — see
// ARD.md §5.5. Uses stream = nullptr (legacy default stream).

#pragma once

#include "CudaCheck.h"
#include "CublasHandle.h"
#include "MatrixView.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace detail {

// Maps a host dtype T -> cublas data type, compute type, and math mode.
// bf16/fp16: tensor cores, fp32 accum, DEFAULT_MATH.
// tfloat:    tf32 tensor cores, fp32 accum, TF32_CUBLAS_MATH (see ARD §9).
template <typename T> struct CublasTypeMap;
template <> struct CublasTypeMap<__nv_bfloat16> {
    static constexpr cudaDataType_t data_type    = CUDA_R_16BF;
    static constexpr cudaDataType_t compute_type = CUDA_R_32F;
    static constexpr cublasMath_t   math_mode    = CUBLAS_DEFAULT_MATH;
};
template <> struct CublasTypeMap<__half> {
    static constexpr cudaDataType_t data_type    = CUDA_R_16F;
    static constexpr cudaDataType_t compute_type = CUDA_R_32F;
    static constexpr cublasMath_t   math_mode    = CUBLAS_DEFAULT_MATH;
};
template <> struct CublasTypeMap<float> {
    static constexpr cudaDataType_t data_type    = CUDA_R_32F;
    static constexpr cudaDataType_t compute_type = CUDA_R_32F;
    static constexpr cublasMath_t   math_mode    = CUBLAS_TF32_TENSOR_OP_MATH;
};

} // namespace detail

// C = A * B  (no transpose, no epilogue). ColMajor only.
// A is (m x k), B is (k x n), C is (m x n). All views must be ColMajor.
//
// A and B are read-only in spirit but NOT const-ified here: cublas_gemm
// is a function template, and C++ template argument deduction does not
// consider implicit conversions, so MatrixView<T,S> -> MatrixView<const T,S>
// would fail deduction at every call site. The const contract is enforced
// at the GemmArgs level (NaiveGemm and future kernels take GemmArgs<T>
// with const A/B); cublas_gemm is the reference path and takes writable
// views for API simplicity.
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
        // ColMajor invariant: the bench runner guarantees this; a violation
        // is a programming error, not a runtime API contract. Debug-only assert.
        GEMM_Y_ASSERT(A.layout == Layout::ColMajor &&
                      B.layout == Layout::ColMajor &&
                      C.layout == Layout::ColMajor,
                      "cublas_gemm: only ColMajor supported");
    }

    using TM = detail::CublasTypeMap<T>;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSetStream(handle.get(), stream));
    // Apply the per-dtype math mode for this call and restore the previous
    // mode on exit. For bf16/fp16 this is a no-op (DEFAULT_MATH -> DEFAULT_MATH);
    // for tfloat it sets TF32_CUBLAS_MATH and restores DEFAULT_MATH.
    CublasMathModeGuard guard(handle.get(), TM::math_mode);
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
