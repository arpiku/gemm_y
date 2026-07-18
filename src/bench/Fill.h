// Fill.h — deterministic host fill pattern for A and B (R13).
//
// Distinct patterns for A and B so they don't alias in cache; values kept
// small to keep bf16/fp32 accumulation in a reasonable range for accuracy
// comparison. Previously duplicated in Profiler.cu and test.cu.

#pragma once

#include "MatrixView.h"
#include "Space.h"

namespace gemm_y {
namespace bench {

// Fill A with ((i+j) & 7) - 3 and B with ((i-j) & 7) - 3. Small ints in
// [-3, 4] — fits exactly in bf16, accumulates in fp32 without rounding,
// so two correct implementations produce bit-identical output. Phase 2
// will use a pattern that produces non-zero reduction-order disagreement.
template <typename T>
void fill_sequential(MatrixView<T, Space::Host> A, MatrixView<T, Space::Host> B) {
    for (int j = 0; j < A.cols; ++j) {
        for (int i = 0; i < A.rows; ++i) {
            A(i, j) = static_cast<T>(((i + j) & 7) - 3);
        }
    }
    for (int j = 0; j < B.cols; ++j) {
        for (int i = 0; i < B.rows; ++i) {
            B(i, j) = static_cast<T>(((i - j) & 7) - 3);
        }
    }
}

} // namespace bench
} // namespace gemm_y
