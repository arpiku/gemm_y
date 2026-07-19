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
// The `fprintf`+`abort` tail lives exactly once in `gemm_y::detail::fail_*`;
// the macros are 5-line wrappers delegating via fully-qualified
// `::gemm_y::detail::fail_*` so they expand correctly inside or outside
// `namespace gemm_y`. Macro-local variables use the suffix-underscore
// convention (e_, s_) to avoid reserved-identifier edge cases.

#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_compat.h"

namespace gemm_y {
namespace detail {

[[noreturn]] inline void fail_cuda(cudaError_t e, const char* tag,
                                   const char* file, int line) noexcept {
    std::fprintf(stderr, "%s at %s:%d: %s (code %d)\n",
                 tag, file, line, cudaGetErrorString(e),
                 static_cast<int>(e));
    std::abort();
}

[[noreturn]] inline void fail_cublas(cublasStatus_t s,
                                     const char* file, int line) noexcept {
    std::fprintf(stderr, "cuBLAS error at %s:%d: %s (code %d)\n",
                 file, line, cublasGetStatusString(s),
                 static_cast<int>(s));
    std::abort();
}

} // namespace detail
} // namespace gemm_y

#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        const cudaError_t e_ = (expr);                                              \
        if (e_ != cudaSuccess)                                                      \
            ::gemm_y::detail::fail_cuda(e_, "CUDA error", __FILE__, __LINE__);      \
    } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                     \
    do {                                                                            \
        const cudaError_t e_ = cudaPeekAtLastError();                               \
        if (e_ != cudaSuccess)                                                      \
            ::gemm_y::detail::fail_cuda(e_, "CUDA async error", __FILE__, __LINE__);\
    } while (0)

#define CUBLAS_CHECK(expr)                                                          \
    do {                                                                            \
        const cublasStatus_t s_ = (expr);                                           \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                            \
            ::gemm_y::detail::fail_cublas(s_, __FILE__, __LINE__);                  \
    } while (0)

// Debug-only assert. No-op in NDEBUG builds. Use for invariants that would
// indicate a programming error, not for runtime API contracts (use
// CUDA_CHECK / CUBLAS_CHECK for those). The NDEBUG branch keeps the
// condition parsed via `(void)sizeof(cond)` so syntax errors in the
// condition are still caught under -DNDEBUG (standard `assert` hygiene).
#ifndef NDEBUG
#  define GEMM_Y_ASSERT(cond, msg)                                                  \
     do {                                                                          \
         if (!(cond)) {                                                            \
             std::fprintf(stderr,                                                  \
                           "Assertion failed at %s:%d: %s\n  condition: %s\n",     \
                           __FILE__, __LINE__, (msg), #cond);                      \
             std::abort();                                                         \
         }                                                                         \
     } while (0)
#else
#  define GEMM_Y_ASSERT(cond, msg)                                                 \
     do {                                                                          \
         (void)sizeof(cond);                                                        \
     } while (0)
#endif
