// Copy.h — the only functions in the codebase that touch cudaMemcpy*.
//
// Dispatch:
//   - contiguous (is_contiguous()) -> cudaMemcpy (sync)
//   - strided                     -> cudaMemcpy2D (sync)
//
// Sync, not async: cudaMemcpyAsync on pageable host memory (which
// std::vector<T> gives) is silently synchronous — the runtime stages
// through an internal pinned buffer (extra host-side copy), then issues
// the DMA. The Phase 1 microbench confirmed sync ≈ async because both
// are sync; the async path adds staging overhead for no benefit.
// True async is deferred to Phase 2 prep, where Space::HostPinned +
// cudaHostAlloc enables real overlap on an explicit stream (ARD §2).
//
// Layout must match between src and dst (no transpose). rows/cols must
// match. The Space tags on src/dst fix the direction (H2D vs D2H).

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "CudaCheck.h"
#include "MatrixView.h"
#include "Space.h"
#include "cuda_compat.h"

namespace gemm_y {

namespace detail {

// Result of planning a copy: pitch/width/height + dispatch decision.
struct CopyPlan {
    std::size_t width_bytes;  // bytes per contiguous row (ColMajor: rows*sizeof(T))
    std::size_t height;       // number of rows (ColMajor: cols)
    std::size_t spitch;       // source pitch (bytes)
    std::size_t dpitch;       // dest pitch (bytes)
    bool contiguous;          // both src and dst contiguous -> cudaMemcpy
};

// Validate shapes/layouts and compute the CopyPlan. Aborts on shape
// mismatch (API contract from caller). Layout mismatch is a debug-only
// assert (Phase 1 invariant — bench runner guarantees ColMajor).
template <typename T, typename U>
CopyPlan plan_copy(MatrixView<T, Space::Host> dst,
                   MatrixView<U, Space::Device> src) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        std::fprintf(stderr,
                     "copy: shape mismatch (dst %dx%d, src %dx%d)\n",
                     dst.rows, dst.cols, src.rows, src.cols);
        std::abort();
    }
    GEMM_Y_ASSERT(dst.layout == src.layout,
                  "copy: layout mismatch (no transpose support)");

    CopyPlan p;
    if (src.layout == Layout::ColMajor) {
        p.width_bytes = static_cast<std::size_t>(src.rows) * sizeof(T);
        p.height      = static_cast<std::size_t>(src.cols);
    } else {
        p.width_bytes = static_cast<std::size_t>(src.cols) * sizeof(T);
        p.height      = static_cast<std::size_t>(src.rows);
    }
    p.spitch      = static_cast<std::size_t>(src.ld) * sizeof(T);
    p.dpitch      = static_cast<std::size_t>(dst.ld) * sizeof(T);
    p.contiguous  = src.is_contiguous() && dst.is_contiguous();
    return p;
}

// Overload with the spaces swapped (Host src, Device dst) — same logic,
// different parameter order so plan_copy can be called in either direction.
template <typename T, typename U>
CopyPlan plan_copy(MatrixView<T, Space::Device> dst,
                   MatrixView<U, Space::Host> src) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        std::fprintf(stderr,
                     "copy: shape mismatch (dst %dx%d, src %dx%d)\n",
                     dst.rows, dst.cols, src.rows, src.cols);
        std::abort();
    }
    GEMM_Y_ASSERT(dst.layout == src.layout,
                  "copy: layout mismatch (no transpose support)");

    CopyPlan p;
    if (src.layout == Layout::ColMajor) {
        p.width_bytes = static_cast<std::size_t>(src.rows) * sizeof(T);
        p.height      = static_cast<std::size_t>(src.cols);
    } else {
        p.width_bytes = static_cast<std::size_t>(src.cols) * sizeof(T);
        p.height      = static_cast<std::size_t>(src.rows);
    }
    p.spitch      = static_cast<std::size_t>(src.ld) * sizeof(T);
    p.dpitch      = static_cast<std::size_t>(dst.ld) * sizeof(T);
    p.contiguous  = src.is_contiguous() && dst.is_contiguous();
    return p;
}

inline void copy_contiguous(void* dst, const void* src,
                            std::size_t bytes, cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
}

inline void copy_strided(void* dst, std::size_t dpitch,
                         const void* src, std::size_t spitch,
                         std::size_t width_bytes, std::size_t height,
                         cudaMemcpyKind kind) {
    CUDA_CHECK(cudaMemcpy2D(dst, dpitch, src, spitch,
                            width_bytes, height, kind));
}

} // namespace detail

// Host -> Device.
// src:  MatrixView<U, Space::Host>      (U = T or const T)
// dst:  MatrixView<T, Space::Device>
template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy_h2d(MatrixView<T, Space::Device> dst,
              MatrixView<U, Space::Host> src) {
    const detail::CopyPlan p = detail::plan_copy(dst, src);
    if (p.contiguous) {
        detail::copy_contiguous(dst.ptr, src.ptr,
                                p.width_bytes * p.height,
                                cudaMemcpyHostToDevice);
    } else {
        detail::copy_strided(dst.ptr, p.dpitch, src.ptr, p.spitch,
                             p.width_bytes, p.height, cudaMemcpyHostToDevice);
    }
}

// Device -> Host.
// src:  MatrixView<U, Space::Device>    (U = T or const T)
// dst:  MatrixView<T, Space::Host>
template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy_d2h(MatrixView<T, Space::Host> dst,
              MatrixView<U, Space::Device> src) {
    const detail::CopyPlan p = detail::plan_copy(dst, src);
    if (p.contiguous) {
        detail::copy_contiguous(dst.ptr, src.ptr,
                                p.width_bytes * p.height,
                                cudaMemcpyDeviceToHost);
    } else {
        detail::copy_strided(dst.ptr, p.dpitch, src.ptr, p.spitch,
                             p.width_bytes, p.height, cudaMemcpyDeviceToHost);
    }
}

} // namespace gemm_y
