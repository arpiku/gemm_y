// Accuracy.h — host-side comparison of got vs ref, promoted to fp64.
//
// Returns max_abs_err and max_rel_err. Tolerance threshold is a constexpr
// so call sites can compare against it without magic numbers.
//
// Tolerance: 1e-2 (conservative; covers cuBLAS non-determinism across
// reduction orders). Tighten once cuBLAS self-error is characterized
// (see ARD.md §6).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "MatrixView.h"

namespace gemm_y {

// Tolerance for bf16 vs cuBLAS reference.
constexpr double kRelErrTol = 1e-2;

template <typename T>
struct ErrReport {
    double max_abs;
    double max_rel;
};

// Promote T -> double for the comparison. bf16/fp16/fp32 all fit cleanly.
template <typename T>
inline double to_double(T x) {
    return static_cast<double>(x);
}

template <typename T, typename U,
          typename = std::enable_if_t<std::is_same_v<U, T> || std::is_same_v<U, const T>>>
[[nodiscard]] ErrReport<T> compare(MatrixView<U, Space::Host> got,
                                  MatrixView<U, Space::Host> ref) {
    ErrReport<T> rep{0.0, 0.0};
    if (got.rows != ref.rows || got.cols != ref.cols) {
        std::fprintf(stderr,
                     "compare: shape mismatch (got %dx%d, ref %dx%d)\n",
                     got.rows, got.cols, ref.rows, ref.cols);
        std::abort();
    }

    for (int j = 0; j < got.cols; ++j) {
        for (int i = 0; i < got.rows; ++i) {
            const double g = to_double(got(i, j));
            const double r = to_double(ref(i, j));
            const double abs_err = std::fabs(g - r);
            // Relative error denominator: max(|r|, tiny) to avoid div-by-0
            // for exact-zero reference entries.
            const double denom = std::fmax(std::fabs(r), 1e-9);
            const double rel_err = abs_err / denom;
            if (abs_err > rep.max_abs) rep.max_abs = abs_err;
            if (rel_err > rep.max_rel) rep.max_rel = rel_err;
        }
    }
    return rep;
}

} // namespace gemm_y
