// CublasHandle.h — RAII wrapper around cublasHandle_t.
//
// One handle is owned by each Profiler<T> (not a process singleton). Setup
// is done once in the ctor; stream binding is per-call (in cublas_gemm),
// so Phase 2 async pipelining can bind different streams without recreating
// the handle (see ARD.md §5.5).
//
// Not thread-safe: cuBLAS handles are not thread-safe, and the bench is
// single-threaded. If multi-threaded bench is ever needed, allocate one
// handle per thread.

#pragma once

#include "CudaCheck.h"
#include "cuda_compat.h"

namespace gemm_y {

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        // One-time setup. No TF32 for the bf16 path; alpha/betas on host.
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
        CUBLAS_CHECK(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            (void)cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr) (void)cublasDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace gemm_y
