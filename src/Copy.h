// Copy.h — the only functions in the codebase that touch cudaMemcpy*.
//
// Dispatch:
//   - contiguous (is_contiguous()) -> cudaMemcpy (sync)
//   - strided                     -> cudaMemcpy2D
// Async path is not wired in Phase 1 (no stream pipelining yet).
//
// Layout must match between src and dst (no transpose). rows/cols must
// match. The Space tags on src/dst fix the direction (H2D vs D2H).
//
// Per ARD.md §2: sync vs async are identical perf without overlap;
// cudaMemcpy2D is mandatory for strided submatrix copies.

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

// Host -> Device.
// src:  MatrixView<const T, Space::Host> (read-only)
// dst:  MatrixView<T,       Space::Device>
//
// We deduce the source element type independently (U) and SFINAE-constrain
// it to be T or const T, so callers can pass either MatrixView<T,Host> or
// MatrixView<const T,Host> (the latter is what const Matrix returns).
template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy_h2d(MatrixView<T, Space::Device> dst,
              MatrixView<U, Space::Host> src,
              cudaStream_t stream = nullptr) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        std::fprintf(stderr,
                     "copy_h2d: shape mismatch (dst %dx%d, src %dx%d)\n",
                     dst.rows, dst.cols, src.rows, src.cols);
        std::abort();
    }
    if (dst.layout != src.layout) {
        std::fprintf(stderr, "copy_h2d: layout mismatch (no transpose support)\n");
        std::abort();
    }

    const auto bytes_per_row = static_cast<std::size_t>(src.cols) * sizeof(T);
    // For ColMajor, the "row" of cudaMemcpy2D is a column of the matrix
    // (contiguous direction). We pass width = rows * sizeof(T) (one column's
    // worth of bytes) and height = cols (number of columns). The "spitch"
    // (source pitch) and "dpitch" (dest pitch) are ld * sizeof(T).
    // For RowMajor, the contiguous direction is along cols: width = cols*sizeof(T),
    // height = rows, pitch = ld * sizeof(T).
    std::size_t width_bytes = 0;
    std::size_t height = 0;
    if (src.layout == Layout::ColMajor) {
        width_bytes = static_cast<std::size_t>(src.rows) * sizeof(T);
        height = static_cast<std::size_t>(src.cols);
    } else {
        width_bytes = static_cast<std::size_t>(src.cols) * sizeof(T);
        height = static_cast<std::size_t>(src.rows);
    }
    const std::size_t spitch = static_cast<std::size_t>(src.ld) * sizeof(T);
    const std::size_t dpitch = static_cast<std::size_t>(dst.ld) * sizeof(T);

    if (src.is_contiguous() && dst.is_contiguous()) {
        CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr,
                                   width_bytes * height,
                                   cudaMemcpyHostToDevice, stream));
        if (stream == nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
        }
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(dst.ptr, dpitch,
                                     src.ptr, spitch,
                                     width_bytes, height,
                                     cudaMemcpyHostToDevice, stream));
        if (stream == nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
        }
    }
}

// Device -> Host.
// src:  MatrixView<const T, Space::Device> (read-only)
// dst:  MatrixView<T,       Space::Host>
template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
void copy_d2h(MatrixView<T, Space::Host> dst,
              MatrixView<U, Space::Device> src,
              cudaStream_t stream = nullptr) {
    if (dst.rows != src.rows || dst.cols != src.cols) {
        std::fprintf(stderr,
                     "copy_d2h: shape mismatch (dst %dx%d, src %dx%d)\n",
                     dst.rows, dst.cols, src.rows, src.cols);
        std::abort();
    }
    if (dst.layout != src.layout) {
        std::fprintf(stderr, "copy_d2h: layout mismatch (no transpose support)\n");
        std::abort();
    }

    std::size_t width_bytes = 0;
    std::size_t height = 0;
    if (src.layout == Layout::ColMajor) {
        width_bytes = static_cast<std::size_t>(src.rows) * sizeof(T);
        height = static_cast<std::size_t>(src.cols);
    } else {
        width_bytes = static_cast<std::size_t>(src.cols) * sizeof(T);
        height = static_cast<std::size_t>(src.rows);
    }
    const std::size_t spitch = static_cast<std::size_t>(src.ld) * sizeof(T);
    const std::size_t dpitch = static_cast<std::size_t>(dst.ld) * sizeof(T);

    if (src.is_contiguous() && dst.is_contiguous()) {
        CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr,
                                   width_bytes * height,
                                   cudaMemcpyDeviceToHost, stream));
        if (stream == nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
        }
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(dst.ptr, dpitch,
                                     src.ptr, spitch,
                                     width_bytes, height,
                                     cudaMemcpyDeviceToHost, stream));
        if (stream == nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(nullptr));
        }
    }
}

} // namespace gemm_y
