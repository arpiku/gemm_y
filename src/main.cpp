// main.cpp — gemm_y entry point.
//
// Phase 1: hardcode the bf16 sweep (no argparse dependency). Registers the
// naive bf16 kernel and runs the full size sweep, writing CSV to
// results/bench_<arch>_bf16.csv.

#include <cstdio>
#include <string>
#include <vector>

#include "bench/Profiler.h"
#include "cuda_compat.h"

#if defined(CUDA_ARCH_SM_90)
    #include "sm90/gemm_bf16_naive.cuh"
    #define GEMM_Y_ARCH_NAME "sm_90"
#elif defined(CUDA_ARCH_SM_120)
    #include "sm120/gemm_bf16_naive.cuh"
    #define GEMM_Y_ARCH_NAME "sm_120"
#else
    #error "Neither CUDA_ARCH_SM_90 nor CUDA_ARCH_SM_120 is defined."
#endif

// NaiveGemm<__nv_bfloat16> is declared in the arch-specific .cuh header
// (src/sm90/ or src/sm120/) selected at configure time, and defined in the
// matching .cu file. main.cpp only needs the declaration to register it.

int main() {
    using T = gemm_y::dtypes::bf16;

    const std::vector<int> kSweepSizes = {
        32, 64, 96, 128, 192, 256, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096
    };

    const std::string out_csv = std::string("results/bench_") +
                                GEMM_Y_ARCH_NAME + "_bf16.csv";

    std::printf("gemm_y: arch=%s dtype=bf16  out=%s\n",
                GEMM_Y_ARCH_NAME, out_csv.c_str());

    gemm_y::Profiler<T> prof;
    prof.register_kernel<gemm_y::NaiveGemm<T>>();
    prof.run_sweep(kSweepSizes, out_csv);

    std::printf("gemm_y: done. CSV written to %s\n", out_csv.c_str());
    return 0;
}
