// CudaTimer.h — RAII cudaEvent_t pair for device-side timing.
//
// Records start/stop events on a stream; elapsed_ms() syncs the stop event
// and returns the GPU-measured elapsed time between record points. This is
// the only correct way to time device work — host steady_clock includes
// launch overhead and OS jitter (see ARD.md §7).
//
// Not thread-safe; one timer per stream. Move-only (no copy).

#pragma once

#include "CudaCheck.h"
#include "cuda_compat.h"

namespace gemm_y {

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        (void)cudaEventDestroy(start_);
        (void)cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    CudaTimer(CudaTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
    }

    CudaTimer& operator=(CudaTimer&& other) noexcept {
        if (this != &other) {
            (void)cudaEventDestroy(start_);
            (void)cudaEventDestroy(stop_);
            start_ = other.start_;
            stop_ = other.stop_;
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }
        return *this;
    }

    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    // Blocks until the stop event has completed on the GPU, then returns
    // elapsed time in milliseconds between start_ and stop_.
    [[nodiscard]] float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

} // namespace gemm_y
