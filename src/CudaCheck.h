// CudaCheck.h — error-checking macros for CUDA runtime + cuBLAS + a generic
// debug-only assert.
//
// Per AGENTS.md: no exceptions across the CUDA boundary. Every runtime call
// is checked; on failure we print file/line/error-string and abort.
//
//   CUDA_CHECK(expr)            — evaluate `expr` (a cudaError_t), abort on != cudaSuccess.
//   CUDA_CHECK_LAST_ERROR()    — peek at the async error state (post-launch).
//   CUBLAS_CHECK(expr)         — evaluate `expr` (a cublasStatus_t), abort on != CUBLAS_STATUS_SUCCESS.
//   GEMM_Y_ASSERT(cond, msg)   — debug-only assert (no-op in NDEBUG builds).
//                                 Use for invariants; use CUDA_CHECK/CUBLAS_CHECK
//                                 for runtime API contracts.
//
// Macro-local variables use the suffix-underscore convention (e.g. gemm_y_err_)
// to avoid reserved-identifier edge cases (R20).

#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_compat.h"

#define GEMM_Y_CUDA_CHECK_IMPL_(err_expr, file, line)                                  \
    do {                                                                               \
        const cudaError_t gemm_y_err_ = (err_expr);                                    \
        if (gemm_y_err_ != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                       \
                          "CUDA error at %s:%d: %s (code %d)\n",                        \
                          file, line,                                                  \
                          cudaGetErrorString(gemm_y_err_),                             \
                          static_cast<int>(gemm_y_err_));                              \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

#define CUDA_CHECK(expr) GEMM_Y_CUDA_CHECK_IMPL_(expr, __FILE__, __LINE__)

#define CUDA_CHECK_LAST_ERROR()                                                         \
    do {                                                                               \
        const cudaError_t gemm_y_err_ = cudaPeekAtLastError();                          \
        if (gemm_y_err_ != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                       \
                          "CUDA async error at %s:%d: %s (code %d)\n",                 \
                          __FILE__, __LINE__,                                          \
                          cudaGetErrorString(gemm_y_err_),                             \
                          static_cast<int>(gemm_y_err_));                              \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(expr)                                                              \
    do {                                                                                \
        const cublasStatus_t gemm_y_stat_ = (expr);                                    \
        if (gemm_y_stat_ != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr,                                                       \
                          "cuBLAS error at %s:%d: %s (code %d)\n",                      \
                          __FILE__, __LINE__,                                          \
                          cublasGetStatusString(gemm_y_stat_),                          \
                          static_cast<int>(gemm_y_stat_));                             \
            std::abort();                                                              \
        }                                                                              \
    } while (0)

// Debug-only assert. No-op in NDEBUG builds. Use for invariants that would
// indicate a programming error, not for runtime API contracts (use
// CUDA_CHECK / CUBLAS_CHECK for those).
#ifndef NDEBUG
#  define GEMM_Y_ASSERT(cond, msg)                                                      \
     do {                                                                              \
         if (!(cond)) {                                                                \
             std::fprintf(stderr,                                                      \
                           "Assertion failed at %s:%d: %s\n  condition: %s\n",         \
                           __FILE__, __LINE__, (msg), #cond);                          \
             std::abort();                                                             \
         }                                                                             \
     } while (0)
#else
#  define GEMM_Y_ASSERT(cond, msg) do { } while (0)
#endif
