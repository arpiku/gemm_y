// Arch.h — single source of truth for the target arch name.
//
// The CMake-selected arch (sm_90 / sm_120) propagates as a CUDA_ARCH_SM_*
// preprocessor define. This header maps it to a string for use in CSV rows,
// log output, and file paths. Previously duplicated in main.cpp, test.cu,
// and Profiler.cu (R11).

#pragma once

#if defined(CUDA_ARCH_SM_90)
    #define GEMM_Y_ARCH_NAME "sm_90"
#elif defined(CUDA_ARCH_SM_120)
    #define GEMM_Y_ARCH_NAME "sm_120"
#else
    #error "Neither CUDA_ARCH_SM_90 nor CUDA_ARCH_SM_120 is defined."
#endif

namespace gemm_y {

// Compile-time arch name. constexpr so it can be used in constexpr contexts
// (e.g. as a std::string_view literal).
constexpr const char* kArchName = GEMM_Y_ARCH_NAME;

} // namespace gemm_y
