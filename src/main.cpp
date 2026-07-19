// main.cpp — gemm_y entry point.
//
// Runs three sequential sweeps (bf16, fp16, tfloat). Each sweep: construct a
// Profiler<T>, register custom kernels (bf16 only for Phase 2A; fp16/tfloat
// register cuBLAS-only — custom kernels land in Phase 2C), call run_sweep,
// write the CSV + a key=value `.meta` sidecar to results/.
//
// One (arch, dtype) pair per CSV. tf32 rows live in the tfloat CSV (the only
// float path — see ARD §9). Hardcoded sweep (no argparse dependency).

#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "Arch.h"
#include "bench/Accuracy.h"
#include "bench/CsvWriter.h"
#include "bench/Profiler.h"
#include "dtypes.h"

#if defined(CUDA_ARCH_SM_90)
    #include "sm90/gemm_bf16_naive.cuh"
#elif defined(CUDA_ARCH_SM_120)
    #include "sm120/gemm_bf16_naive.cuh"
#endif

namespace {

// 14-size square sweep (powers of 2 + midpoints).
const std::vector<int> kSweepSizes = {
    32, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096
};

// Write a SweepResult to a CSV file. Schema is defined here (one place).
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

// ISO 8601 UTC timestamp, second precision (e.g. "2026-07-19T17:44:00Z").
std::string iso8601_now() {
    using std::chrono::system_clock;
    const auto now = system_clock::now();
    const auto t = system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

// Write a key=value `.meta` sidecar alongside the CSV. Parsed by
// scripts/ingest.py. `|` separates kernel name and description (descriptions
// may contain commas/spaces; `|` will not). Multiple `kernel=` lines, cuBLAS
// first. No git sha (captured by ingest.py at ingest time).
void write_meta(const std::string& meta_path,
                const std::string& arch,
                const std::string& dtype,
                int warmup_iters,
                int timed_iters,
                double tol,
                const std::vector<std::pair<std::string, std::string>>& custom_kernels) {
    std::FILE* f = std::fopen(meta_path.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "Failed to open %s for write\n", meta_path.c_str());
        std::abort();
    }
    std::fprintf(f, "arch=%s\n", arch.c_str());
    std::fprintf(f, "dtype=%s\n", dtype.c_str());
    std::fprintf(f, "warmup_iters=%d\n", warmup_iters);
    std::fprintf(f, "timed_iters=%d\n", timed_iters);
    std::fprintf(f, "tol=%.6g\n", tol);
    std::fprintf(f, "sweep_sizes=");
    for (std::size_t i = 0; i < kSweepSizes.size(); ++i) {
        std::fprintf(f, "%s%d", (i == 0 ? "" : ","), kSweepSizes[i]);
    }
    std::fprintf(f, "\n");
    // cuBLAS is always the first kernel row (the implicit reference).
    std::fprintf(f, "kernel=cublas|cublasGemmEx reference (fp32 accum)\n");
    for (const auto& [name, desc] : custom_kernels) {
        std::fprintf(f, "kernel=%s|%s\n", name.c_str(), desc.c_str());
    }
    std::fprintf(f, "timestamp=%s\n", iso8601_now().c_str());
    std::fclose(f);
}

} // namespace

int main() {
    const std::string arch = gemm_y::kArchName;

    // bf16: register NaiveGemm (the existing baseline kernel).
    {
        using T = gemm_y::dtypes::bf16;
        gemm_y::Profiler<T> prof;
        prof.register_kernel<gemm_y::NaiveGemm<T>>();
        const gemm_y::SweepResult result = prof.run_sweep(kSweepSizes);

        const std::string out_csv = "results/bench_" + arch + "_bf16.csv";
        const std::string out_meta = "results/bench_" + arch + "_bf16.meta";
        std::printf("gemm_y: arch=%s dtype=bf16  out=%s\n", arch.c_str(), out_csv.c_str());
        write_csv(result, out_csv);
        std::vector<std::pair<std::string, std::string>> kernels;
        kernels.emplace_back(std::string(gemm_y::NaiveGemm<T>::name()),
                             std::string(gemm_y::NaiveGemm<T>::description()));
        write_meta(out_meta, arch, "bf16", 20, 50, gemm_y::kRelErrTol<T>(), kernels);
        std::printf("gemm_y: done. %zu rows written to %s\n", result.rows.size(), out_csv.c_str());
    }

    // fp16: cuBLAS-only for Phase 2A (custom kernel deferred to Phase 2C).
    {
        using T = gemm_y::dtypes::fp16;
        gemm_y::Profiler<T> prof;
        const gemm_y::SweepResult result = prof.run_sweep(kSweepSizes);

        const std::string out_csv = "results/bench_" + arch + "_fp16.csv";
        const std::string out_meta = "results/bench_" + arch + "_fp16.meta";
        std::printf("gemm_y: arch=%s dtype=fp16  out=%s\n", arch.c_str(), out_csv.c_str());
        write_csv(result, out_csv);
        write_meta(out_meta, arch, "fp16", 20, 50, gemm_y::kRelErrTol<T>(), {});
        std::printf("gemm_y: done. %zu rows written to %s\n", result.rows.size(), out_csv.c_str());
    }

    // tfloat: cuBLAS-only for Phase 2A (custom kernel deferred to Phase 2C).
    {
        using T = gemm_y::dtypes::tfloat;
        gemm_y::Profiler<T> prof;
        const gemm_y::SweepResult result = prof.run_sweep(kSweepSizes);

        const std::string out_csv = "results/bench_" + arch + "_tf32.csv";
        const std::string out_meta = "results/bench_" + arch + "_tf32.meta";
        std::printf("gemm_y: arch=%s dtype=tf32  out=%s\n", arch.c_str(), out_csv.c_str());
        write_csv(result, out_csv);
        write_meta(out_meta, arch, "tf32", 20, 50, gemm_y::kRelErrTol<T>(), {});
        std::printf("gemm_y: done. %zu rows written to %s\n", result.rows.size(), out_csv.c_str());
    }

    return 0;
}
