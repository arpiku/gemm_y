// MatrixView.h — non-owning {ptr, rows, cols, ld, layout} view over a matrix.
//
// POD, pass-by-value (fits registers, no aliasing). Works for both T and
// const T (mirrors std::span semantics): MatrixView<const T, S> is the
// "read-only" view returned by const Matrix.
//
// Submatrix slicing via block(r,c,m,n) is zero-copy: returns a view with
// unchanged ld and offset ptr. This is what lets the bench runner pre-
// allocate a single 4096x4096 buffer and feed N x N submatrices to every
// kernel without copying (see ARD.md §5).
//
// ld semantics (ColMajor, the Phase 1 default):
//   - ld >= rows is the leading dimension (stride between contiguous columns).
//   - element (i,j) lives at ptr + i + j*ld.
//   - is_contiguous() iff ld == rows.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "CudaCheck.h"
#include "Layout.h"
#include "Space.h"

namespace gemm_y {

template <typename T, Space S>
struct MatrixView {
    using element_type = T;
    static constexpr Space space = S;

    T* ptr = nullptr;
    int rows = 0;
    int cols = 0;
    int ld = 0;
    Layout layout = Layout::ColMajor;

    MatrixView() noexcept = default;

    MatrixView(T* p, int r, int c, int leading, Layout lay = Layout::ColMajor) noexcept
        : ptr(p), rows(r), cols(c), ld(leading), layout(lay) {}

    // Converting constructor: MatrixView<T, S> -> MatrixView<const T, S>.
    // Mirrors std::span's const-conversion. Enabled only when the source
    // element type is non-const and the destination is const (so the
    // conversion adds const but never strips it).
    template <typename U,
              typename = std::enable_if_t<
                  std::is_same_v<T, const U> &&
                  !std::is_same_v<T, U>>>
    MatrixView(const MatrixView<U, S>& other) noexcept
        : ptr(other.ptr), rows(other.rows), cols(other.cols),
          ld(other.ld), layout(other.layout) {}

    // Zero-copy sub-view at offset (r,c) of size m x n. ld is unchanged.
    // Debug-only bounds asserts catch silent OOB if misused (R18).
    [[nodiscard]] MatrixView<T, S> block(int r, int c, int m, int n) const noexcept {
        GEMM_Y_ASSERT(r >= 0 && c >= 0 && m >= 0 && n >= 0,
                      "block(): negative args");
        GEMM_Y_ASSERT(r + m <= rows && c + n <= cols,
                      "block(): submatrix exceeds parent bounds");
        T* offset = nullptr;
        if (layout == Layout::ColMajor) {
            offset = ptr + static_cast<std::ptrdiff_t>(r)
                          + static_cast<std::ptrdiff_t>(c) * static_cast<std::ptrdiff_t>(ld);
        } else {
            // RowMajor: element (i,j) at ptr + i*ld + j.
            offset = ptr + static_cast<std::ptrdiff_t>(r) * static_cast<std::ptrdiff_t>(ld)
                          + static_cast<std::ptrdiff_t>(c);
        }
        return MatrixView<T, S>{offset, m, n, ld, layout};
    }

    // Contiguous iff the leading dimension equals the contiguous extent.
    // ColMajor: ld == rows. RowMajor: ld == cols.
    [[nodiscard]] bool is_contiguous() const noexcept {
        return (layout == Layout::ColMajor) ? (ld == rows) : (ld == cols);
    }

    // Element access — host-side test/debug only. Not for hot paths.
    // ColMajor: ptr + i + j*ld. RowMajor: ptr + i*ld + j.
    [[nodiscard]] T& operator()(int i, int j) noexcept {
        if (layout == Layout::ColMajor) {
            return ptr[static_cast<std::ptrdiff_t>(i)
                       + static_cast<std::ptrdiff_t>(j) * static_cast<std::ptrdiff_t>(ld)];
        }
        return ptr[static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(ld)
                   + static_cast<std::ptrdiff_t>(j)];
    }
    [[nodiscard]] const T& operator()(int i, int j) const noexcept {
        if (layout == Layout::ColMajor) {
            return ptr[static_cast<std::ptrdiff_t>(i)
                       + static_cast<std::ptrdiff_t>(j) * static_cast<std::ptrdiff_t>(ld)];
        }
        return ptr[static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(ld)
                   + static_cast<std::ptrdiff_t>(j)];
    }
};

} // namespace gemm_y
