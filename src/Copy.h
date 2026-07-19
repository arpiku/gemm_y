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
// match. The Space tags on src/dst fix the direction (H2D vs D2H) via
// detail::copy_kind_v<Dst, Src> — a constexpr table that maps the
// (Dst, Src) Space pair to the cudaMemcpyKind. Wrong-direction
// instantiations (Host->Host, Device->Device) are rejected by a
// static_assert inside detail::copy (Phase 1.6.5/1.6.6).

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

// constexpr table mapping (Dst, Src) Space pair -> cudaMemcpyKind.
// Primary template is the poison value (defensive secondary net); only the
// two H2D/D2H specializations are defined. A static_assert inside
// detail::copy rejects invalid (Dst, Src) pairs at compile time, so the
// poison value is never reached at runtime (Phase 1.6.5).
template <Space Dst, Space Src>
inline constexpr cudaMemcpyKind copy_kind_v = static_cast<cudaMemcpyKind>(-1);
template <>
inline constexpr cudaMemcpyKind copy_kind_v<Space::Device, Space::Host> = cudaMemcpyHostToDevice;
template <>
inline constexpr cudaMemcpyKind copy_kind_v<Space::Host,   Space::Device> = cudaMemcpyDeviceToHost;

// Validate shapes/layouts and compute the CopyPlan. Aborts on shape
// mismatch (API contract from caller). Layout mismatch is a debug-only
// assert (Phase 1 invariant — bench runner guarantees ColMajor).
//
// Space-agnostic: the Space tags are caller-enforcement only; the body
// reads rows/cols/ld/layout/is_contiguous() (Phase 1.6.4).
template <Space Dst, Space Src, typename T, typename U>
CopyPlan plan_copy(MatrixView<T, Dst> dst, MatrixView<U, Src> src) {
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

// Single copy body shared by copy_h2d / copy_d2h. The cudaMemcpyKind is
// deduced at compile time from the (Dst, Src) Space pair via copy_kind_v
// (Phase 1.6.6). Both the cudaMemcpy (contiguous) and cudaMemcpy2D
// (strided) paths are preserved.
template <Space Dst, Space Src, typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy(MatrixView<T, Dst> dst, MatrixView<U, Src> src) {
    static_assert((Dst == Space::Device && Src == Space::Host) ||
                  (Dst == Space::Host   && Src == Space::Device),
                  "copy: only Host<->Device directions are supported");
    constexpr cudaMemcpyKind kind = copy_kind_v<Dst, Src>;
    const CopyPlan p = plan_copy(dst, src);
    if (p.contiguous) {
        CUDA_CHECK(cudaMemcpy(dst.ptr, src.ptr,
                              p.width_bytes * p.height, kind));
    } else {
        CUDA_CHECK(cudaMemcpy2D(dst.ptr, p.dpitch, src.ptr, p.spitch,
                                p.width_bytes, p.height, kind));
    }
}

} // namespace detail

// Host -> Device.
// src:  MatrixView<U, Space::Host>      (U = T or const T)
// dst:  MatrixView<T, Space::Device>
// 2-line wrapper delegating to detail::copy — preserves call-site API
// (Phase 1.6.7). No `inline` (templates are implicitly inline).
template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy_h2d(MatrixView<T, Space::Device> dst,
              MatrixView<U, Space::Host> src) {
    detail::copy(dst, src);
}

// Device -> Host.
// src:  MatrixView<U, Space::Device>    (U = T or const T)
// dst:  MatrixView<T, Space::Host>
template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy_d2h(MatrixView<T, Space::Host> dst,
              MatrixView<U, Space::Device> src) {
    detail::copy(dst, src);
}

} // namespace gemm_y
