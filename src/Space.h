// Space.h — compile-time memory-space tag.
//
// Drives copy_h2d / copy_d2h overload resolution and Buffer<T,S> storage
// selection. A runtime tag would lose the compile-time dispatch and risk
// calling cudaMemcpy on a host pointer from device code with no compile
// error (see ARD.md §1).

#pragma once

#include <cstdint>

namespace gemm_y {

enum class Space : std::uint8_t {
    Host,
    Device,
};

} // namespace gemm_y
