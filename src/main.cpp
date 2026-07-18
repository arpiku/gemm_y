// main.cpp — gemm_y entry point.
//
// Phase 1.5: Profiler::run_sweep returns a SweepResult; main.cpp owns the
// CsvWriter and iterates result.rows to write the CSV. This decouples bench
// logic from I/O (R3).
//
// Phase 1: hardcode the bf16 sweep (no argparse dependency). Registers the
// naive bf16 kernel and runs the full size sweep, writing CSV to
// results/bench_<arch>_bf16.csv.

#include <cstdio>
#include <string>
#include <vector>

#include "Arch.h"
#include "bench/CsvWriter.h"
#include "bench/Profiler.h"
#include "dtypes.h"

#if defined(CUDA_ARCH_SM_90)
    #include "sm90/gemm_bf16_naive.cuh"
#elif defined(CUDA_ARCH_SM_120)
    #include "sm120/gemm_bf16_naive.cuh"
#endif

namespace {

// Write a SweepResult to a CSV file. Schema is defined here (one place — R3).
void write_csv(const gemm_y::SweepResult& result, const std::string& path) {
    gemm_y::CsvWriter csv;
    if (!csv.open(path)) {
        std::fprintf(stderr, "Failed to open %s for write\n", path.c_str());
        std::abort();
    }
    csv.write_header(
        "arch,dtype,N,kernel_name,kernel_desc,h2d_ns,kernel_min_ns,kernel_median_ns,"
        "d2h_ns,ref_kernel_min_ns,ref_kernel_median_ns,max_abs_err,max_rel_err");
    for (const auto& r : result.rows) {
        csv.append_row(r.arch, r.dtype, r.N, r.kernel_name, r.kernel_desc,
                       r.h2d_ns, r.kernel_min_ns, r.kernel_median_ns,
                       r.d2h_ns, r.ref_kernel_min_ns, r.ref_kernel_median_ns,
                       r.max_abs_err, r.max_rel_err);
    }
}

} // namespace

int main() {
    using T = gemm_y::dtypes::bf16;

    const std::vector<int> kSweepSizes = {
        32, 64, 96, 128, 192, 256, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096
    };

    const std::string out_csv = std::string("results/bench_") +
                                gemm_y::kArchName + "_bf16.csv";

    std::printf("gemm_y: arch=%s dtype=bf16  out=%s\n",
                gemm_y::kArchName, out_csv.c_str());

    gemm_y::Profiler<T> prof;
    prof.register_kernel<gemm_y::NaiveGemm<T>>();
    const gemm_y::SweepResult result = prof.run_sweep(kSweepSizes);

    write_csv(result, out_csv);

    std::printf("gemm_y: done. %zu rows written to %s\n",
                result.rows.size(), out_csv.c_str());
    return 0;
}
