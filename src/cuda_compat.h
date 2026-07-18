// cuda_compat.h — single include wrapper around CUDA runtime headers.
//
// All other project headers include CUDA through this file only. It pushes
// GCC diagnostic pragmas around the CUDA headers to suppress the
// -Wold-style-cast / -Wconversion / -Wpedantic noise that NVIDIA's own
// inline templates trip heavily. Our own code stays under the project's
// full warning policy.
//
// Re-exports __nv_bfloat16 / __half / float as dtype aliases under
// namespace gemm_y::dtypes so call sites don't depend on the CUDA-internal
// header names directly.

#pragma once

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#include <string_view>

namespace gemm_y {
namespace dtypes {

using bf16 = __nv_bfloat16;
using fp16 = __half;
using fp32 = float;

} // namespace dtypes
} // namespace gemm_y
