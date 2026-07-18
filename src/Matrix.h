// Matrix.h — owning matrix: Buffer<T,S> + shape (rows, cols, ld, layout).
//
// Owns a Buffer<T,S> and exposes a MatrixView over it. ld == rows for the
// ColMajor default (so a freshly-allocated Matrix is always contiguous).
// Move-only (inherits from Buffer). Factory: Matrix<T,S>::alloc(rows, cols).
//
// const Matrix returns MatrixView<const T, S>; non-const returns MatrixView<T, S>.

#pragma once

#include "Buffer.h"
#include "Layout.h"
#include "MatrixView.h"
#include "Space.h"

namespace gemm_y {

template <typename T, Space S>
class Matrix {
public:
    Matrix() noexcept = default;

    Matrix(Buffer<T, S>&& buf, int rows, int cols, int ld, Layout layout) noexcept
        : buf_(std::move(buf)), rows_(rows), cols_(cols), ld_(ld), layout_(layout) {}

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    [[nodiscard]] static Matrix alloc(int rows, int cols, Layout layout = Layout::ColMajor) {
        // ColMajor: contiguous extent = rows; ld = rows.
        // RowMajor: contiguous extent = cols; ld = cols.
        const int ld = (layout == Layout::ColMajor) ? rows : cols;
        Buffer<T, S> buf(static_cast<std::size_t>(ld) * static_cast<std::size_t>(cols));
        return Matrix(std::move(buf), rows, cols, ld, layout);
    }

    [[nodiscard]] int rows() const noexcept { return rows_; }
    [[nodiscard]] int cols() const noexcept { return cols_; }
    [[nodiscard]] int ld() const noexcept { return ld_; }
    [[nodiscard]] Layout layout() const noexcept { return layout_; }

    // Non-const: mutable view.
    [[nodiscard]] MatrixView<T, S> view() noexcept {
        return MatrixView<T, S>{buf_.data(), rows_, cols_, ld_, layout_};
    }

    // Const: read-only view.
    [[nodiscard]] MatrixView<const T, S> view() const noexcept {
        return MatrixView<const T, S>{buf_.data(), rows_, cols_, ld_, layout_};
    }

    [[nodiscard]] T* data() noexcept { return buf_.data(); }
    [[nodiscard]] const T* data() const noexcept { return buf_.data(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return buf_.bytes(); }

private:
    Buffer<T, S> buf_;
    int rows_ = 0;
    int cols_ = 0;
    int ld_ = 0;
    Layout layout_ = Layout::ColMajor;
};

} // namespace gemm_y
