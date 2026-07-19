// GemmArgs.h — POD bundle of device matrix views passed to a kernel.
//
// Pass-by-value (fits registers, no aliasing). Future extension point for
// alpha/beta is commented out — plain C = A * B only (AGENTS.md non-goal:
// epilogue fusion).

#pragma once

#include "MatrixView.h"
#include "Space.h"

namespace gemm_y {

template <typename T>
struct GemmArgs {
    // A and B are kernel inputs (read-only). C is the output (mutable).
    // The const-ification relies on MatrixView's implicit converting
    // constructor (MatrixView<T,S> -> MatrixView<const T,S>), so call
    // sites passing writable views for A/B compile unchanged.
    MatrixView<const T, Space::Device> A;  // input, read-only
    MatrixView<const T, Space::Device> B;  // input, read-only
    MatrixView<T,       Space::Device> C;  // output, mutable
    // float alpha = 1.0f;  // reserved for Phase 2+
    // float beta  = 0.0f;  // reserved for Phase 2+
};

} // namespace gemm_y
