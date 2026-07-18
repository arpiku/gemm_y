// cuda_compat.h — single include wrapper around CUDA runtime + cuBLAS headers.
//
// All other project headers include CUDA through this file only. It pushes
// GCC diagnostic pragmas around the CUDA headers to suppress the
// -Wold-style-cast / -Wconversion / -Wpedantic noise that NVIDIA's own
// inline templates trip heavily. Our own code stays under the project's
// full warning policy.
//
// Dtype aliases (bf16/fp16/fp32) and name<T>() live in dtypes.h, which
// includes this header for the underlying CUDA types.

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
#include <cublas_v2.h>

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif
