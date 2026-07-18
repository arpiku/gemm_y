cmake_minimum_required(VERSION 3.21)

project(gemm_y
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# Single-config generators (Make/Ninja) need a default build type.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# CUDA toolkit baseline
# ---------------------------------------------------------------------------
# CUDA 12.8 is the first release supporting both sm_90 (Hopper) and
# sm_120 (Blackwell). Fail loudly if the local toolkit is older.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR
        "CUDA toolkit >= 12.8 required (found ${CMAKE_CUDA_COMPILER_VERSION}).")
endif()

# ---------------------------------------------------------------------------
# Target architecture selection (one binary per arch, build-time)
# ---------------------------------------------------------------------------
# Usage:
#   cmake -B build                                    # -> Blackwell sm_120
#   cmake -B build -DGEMM_Y_CUDA_ARCH=sm_90           # -> Hopper    sm_90
set(GEMM_Y_CUDA_ARCH "sm_120" CACHE STRING "Target CUDA architecture")
set_property(CACHE GEMM_Y_CUDA_ARCH PROPERTY STRINGS sm_90 sm_120)

if(GEMM_Y_CUDA_ARCH STREQUAL "sm_90")
    set(_gemm_y_cuda_arch_num 90)
    set(_gemm_y_cuda_arch_define CUDA_ARCH_SM_90)
elseif(GEMM_Y_CUDA_ARCH STREQUAL "sm_120")
    set(_gemm_y_cuda_arch_num 120)
    set(_gemm_y_cuda_arch_define CUDA_ARCH_SM_120)
else()
    message(FATAL_ERROR
        "GEMM_Y_CUDA_ARCH must be 'sm_90' or 'sm_120' (got '${GEMM_Y_CUDA_ARCH}')")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${_gemm_y_cuda_arch_num})

message(STATUS "gemm_y: CUDA arch = ${GEMM_Y_CUDA_ARCH} "
               "(CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")

# ---------------------------------------------------------------------------
# Warning flags
# ---------------------------------------------------------------------------
option(ENABLE_WERROR "Treat compiler warnings as errors" OFF)

# Host-compiler warnings applied to pure CXX sources.
set(_gemm_y_host_warnings
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wformat=2
    -Wnull-dereference
)

# Subset of host warnings forwarded to nvcc's host compiler via -Xcompiler.
# Excludes -Wold-style-cast, -Wconversion, and -Wpedantic because CUDA's own
# runtime headers, nvcc's generated stub files, and nvcc's #line directives
# trip them heavily.
set(_gemm_y_nvcc_xcompiler_warnings
    -Wall
    -Wextra
    -Wshadow
    -Wnon-virtual-dtor
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wformat=2
    -Wnull-dereference
)

# nvcc-native warnings.
set(_gemm_y_nvcc_warnings
    -Wreorder
    -Winit-self
)

if(ENABLE_WERROR)
    list(APPEND _gemm_y_host_warnings -Werror)
    list(APPEND _gemm_y_nvcc_xcompiler_warnings -Werror)
    list(APPEND _gemm_y_nvcc_warnings -Werror)
endif()

# Join the nvcc-host warnings into a single -Xcompiler=<a;b;c> argument.
list(JOIN _gemm_y_nvcc_xcompiler_warnings ";" _gemm_y_nvcc_xcompiler_joined)
set(_gemm_y_nvcc_xcompiler_flag "Xcompiler=${_gemm_y_nvcc_xcompiler_joined}")

# ---------------------------------------------------------------------------
# Per-build-type CUDA optimization flags
# ---------------------------------------------------------------------------
# nvcc accepts -O0/-O1/-O2/-O3 directly. Mirror the host build type so device
# code is optimized consistently with host code.
set(CMAKE_CUDA_FLAGS_DEBUG          "-O0 -g"        CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE        "-O3 -DNDEBUG"  CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g"        CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_MINSIZEREL     "-Os -DNDEBUG"  CACHE STRING "" FORCE)

# ---------------------------------------------------------------------------
# Helper: apply standards + warnings + arch define to a target
# ---------------------------------------------------------------------------
function(gemm_y_apply_common_flags target)
    target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)

    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${_gemm_y_host_warnings}>
        $<$<COMPILE_LANGUAGE:CUDA>:${_gemm_y_nvcc_warnings}>
        $<$<COMPILE_LANGUAGE:CUDA>:-${_gemm_y_nvcc_xcompiler_flag}>
    )

    target_compile_definitions(${target} PRIVATE ${_gemm_y_cuda_arch_define})
endfunction()

# ---------------------------------------------------------------------------
# cuBLAS (required for the reference path)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Arch-specific source directories (prepared, not populated)
# ---------------------------------------------------------------------------
# When src/sm90/ or src/sm120/ gain .cu files, only the directory matching the
# selected arch is compiled in. Keeps "one binary per arch" clean.
set(_gemm_y_arch_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/${GEMM_Y_CUDA_ARCH}")
# Directory naming convention is sm90 / sm120 (no underscore), while the
# GEMM_Y_CUDA_ARCH cache variable uses the canonical sm_90 / sm_120 form.
# Map between them so the glob finds the right directory.
string(REPLACE "sm_" "sm" _gemm_y_arch_dir_name ${GEMM_Y_CUDA_ARCH})
set(_gemm_y_arch_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/${_gemm_y_arch_dir_name}")
if(EXISTS "${_gemm_y_arch_src_dir}")
    file(GLOB _gemm_y_arch_sources CONFIGURE_DEPENDS "${_gemm_y_arch_src_dir}/*.cu")
else()
    set(_gemm_y_arch_sources "")
endif()
message(STATUS "gemm_y: arch_src_dir=${_gemm_y_arch_src_dir}")
message(STATUS "gemm_y: arch_sources=${_gemm_y_arch_sources}")

# Arch-agnostic bench sources (Profiler only). The microbench files
# (memcpy_microbench.cu, launch_overhead_microbench.cu) define their own
# main() and are built as standalone executables below — exclude them from
# the gemm_y target to avoid duplicate main() symbols.
file(GLOB _gemm_y_bench_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/bench/*.cu")
list(FILTER _gemm_y_bench_sources EXCLUDE REGEX "microbench\.cu$")

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(gemm_y
    src/main.cpp
    ${_gemm_y_arch_sources}
    ${_gemm_y_bench_sources}
)
target_include_directories(gemm_y PRIVATE src)
target_link_libraries(gemm_y PRIVATE CUDA::cublas)
gemm_y_apply_common_flags(gemm_y)

# ---------------------------------------------------------------------------
# Build-verification + unit-test target
# ---------------------------------------------------------------------------
add_executable(test_cuda
    tests/test.cu
)
target_include_directories(test_cuda PRIVATE src)
target_link_libraries(test_cuda PRIVATE CUDA::cublas)
gemm_y_apply_common_flags(test_cuda)

# ---------------------------------------------------------------------------
# Microbench targets (optional; not part of the default build)
# ---------------------------------------------------------------------------
# These are standalone executables with their own main(). They are NOT added
# to the default `all` target so a plain `cmake --build build` only produces
# gemm_y + test_cuda. Build explicitly with `cmake --build build --target
# memcpy_microbench launch_overhead_microbench`.
add_executable(memcpy_microbench EXCLUDE_FROM_ALL
    src/bench/memcpy_microbench.cu
)
target_include_directories(memcpy_microbench PRIVATE src)
gemm_y_apply_common_flags(memcpy_microbench)

add_executable(launch_overhead_microbench EXCLUDE_FROM_ALL
    src/bench/launch_overhead_microbench.cu
)
target_include_directories(launch_overhead_microbench PRIVATE src)
gemm_y_apply_common_flags(launch_overhead_microbench)
