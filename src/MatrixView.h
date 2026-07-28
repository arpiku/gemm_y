// MatrixView.h — non-owning {ptr, rows, cols, ld} view over a matrix.
//
// Dual-use contract:
//   (1) Host-side view: operator() for element access; converting ctor for
//       const-correctness (MatrixView<T,S> -> MatrixView<const T,S>).
//       Host-only utilities — NOT __device__-callable.
//   (2) Kernel-side POD descriptor: kernels receive MatrixView by value
//       and read ptr / rows / cols / ld directly. The host methods are
//       never called from device code.
//
// ColMajor is the only layout (enforced structurally by Matrix::alloc
// setting ld = rows); there is no Layout field. If RowMajor is ever
// needed, re-add a Layout tag + the branches (~5 lines per call site).
//
// POD, pass-by-value (fits registers, no aliasing). Works for both T and
// const T (mirrors std::span semantics): MatrixView<const T, S> is the
// "read-only" view returned by const Matrix.

#pragma once

#include <cstddef>
#include <type_traits>

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

    MatrixView() noexcept = default;

    MatrixView(T* p, int r, int c, int leading) noexcept
        : ptr(p), rows(r), cols(c), ld(leading) {}

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
          ld(other.ld) {}

    // Element access — host-side test/debug only. Not for hot paths.
    // ColMajor: element (i,j) at ptr + i + j*ld.
    [[nodiscard]] T& operator()(int i, int j) noexcept {
        return ptr[static_cast<std::ptrdiff_t>(i)
                   + static_cast<std::ptrdiff_t>(j) * static_cast<std::ptrdiff_t>(ld)];
    }
    [[nodiscard]] const T& operator()(int i, int j) const noexcept {
        return ptr[static_cast<std::ptrdiff_t>(i)
                   + static_cast<std::ptrdiff_t>(j) * static_cast<std::ptrdiff_t>(ld)];
    }
};

} // namespace gemm_y
