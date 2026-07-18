// GemmArgs.h — POD bundle of device matrix views passed to a kernel.
//
// Pass-by-value (fits registers, no aliasing). Future extension point for
// alpha/beta is commented out — Phase 1 is plain C = A * B only (AGENTS.md
// non-goal: epilogue fusion).

#pragma once

#include "MatrixView.h"
#include "Space.h"

namespace gemm_y {

template <typename T>
struct GemmArgs {
    MatrixView<T, Space::Device> A;
    MatrixView<T, Space::Device> B;
    MatrixView<T, Space::Device> C;
    // float alpha = 1.0f;  // reserved for Phase 2+
    // float beta  = 0.0f;  // reserved for Phase 2+
};

} // namespace gemm_y
