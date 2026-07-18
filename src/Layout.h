// Layout.h — compile-time matrix layout tag.
//
// ColMajor is the cuBLAS native layout and the Phase 1 default. RowMajor
// is reserved (no code path in Phase 1) but the tag must propagate through
// MatrixView so future template specializations can dispatch on it without
// runtime branches in hot paths (see ARD.md §1).

#pragma once

#include <cstdint>

namespace gemm_y {

enum class Layout : std::uint8_t {
    ColMajor,
    RowMajor,
};

} // namespace gemm_y
