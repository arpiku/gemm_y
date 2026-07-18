// CudaCheck.h — error-checking macros for CUDA runtime + cuBLAS.
//
// Per AGENTS.md: no exceptions across the CUDA boundary. Every runtime call
// is checked; on failure we print file/line/error-string and abort.
//
//   CUDA_CHECK(expr)            — evaluate `expr` (a cudaError_t), abort on != cudaSuccess.
//   CUDA_CHECK_LAST_ERROR()     — peek at the async error state (post-launch).
//   CUBLAS_CHECK(expr)          — evaluate `expr` (a cublasStatus_t), abort on != CUBLAS_STATUS_SUCCESS.

#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_compat.h"
#include <cublas_v2.h>

#define GEMM_Y_CUDA_CHECK_IMPL_(err_expr, file, line)                                  \
    do {                                                                               \
        const cudaError_t _gemm_y_err = (err_expr);                                   \
        if (_gemm_y_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                                      \
                          "CUDA error at %s:%d: %s (code %d)\n",                       \
                          file, line,                                                 \
                          cudaGetErrorString(_gemm_y_err),                             \
                          static_cast<int>(_gemm_y_err));                             \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

#define CUDA_CHECK(expr) GEMM_Y_CUDA_CHECK_IMPL_(expr, __FILE__, __LINE__)

#define CUDA_CHECK_LAST_ERROR()                                                       \
    do {                                                                              \
        const cudaError_t _gemm_y_err = cudaPeekAtLastError();                         \
        if (_gemm_y_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                                      \
                          "CUDA async error at %s:%d: %s (code %d)\n",                \
                          __FILE__, __LINE__,                                         \
                          cudaGetErrorString(_gemm_y_err),                             \
                          static_cast<int>(_gemm_y_err));                             \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(expr)                                                            \
    do {                                                                              \
        const cublasStatus_t _gemm_y_stat = (expr);                                   \
        if (_gemm_y_stat != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr,                                                      \
                          "cuBLAS error at %s:%d: %s (code %d)\n",                    \
                          __FILE__, __LINE__,                                         \
                          cublasGetStatusString(_gemm_y_stat),                        \
                          static_cast<int>(_gemm_y_stat));                            \
            std::abort();                                                              \
        }                                                                              \
    } while (0)
