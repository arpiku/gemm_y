// Space.h — compile-time memory-space tag.
//
// Tags Buffer<T,S> storage selection (Host = std::vector, Device =
// cudaMalloc). A runtime tag would lose the compile-time guarantee and
// risk calling cudaMemcpy on a host pointer from device code with no
// compile error (see ARD.md §1).

#pragma once

#include <cstdint>

namespace gemm_y {

enum class Space : std::uint8_t {
    Host,
    Device,
};

} // namespace gemm_y
