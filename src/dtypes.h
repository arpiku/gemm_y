// dtypes.h — dtype aliases + name<T>() for the supported storage dtypes.
//
// Co-locates the alias set (previously in cuda_compat.h) with the name<T>()
// mapping (previously a free function dtype_name<T>() in Profiler.cu).
// cuda_compat.h now only handles CUDA header inclusion + diagnostic
// suppression; this header pulls in the underlying CUDA types it needs.

#pragma once

#include <string_view>

#include "cuda_compat.h"

namespace gemm_y {
namespace dtypes {

using bf16 = __nv_bfloat16;
using fp16 = __half;
using fp32 = float;

// Short name for use in CSV `dtype` column, log output, and file paths.
template <typename T> constexpr std::string_view name();
template <> constexpr std::string_view name<__nv_bfloat16>() { return "bf16"; }
template <> constexpr std::string_view name<__half>()        { return "fp16"; }
template <> constexpr std::string_view name<float>()         { return "fp32"; }

} // namespace dtypes
} // namespace gemm_y
