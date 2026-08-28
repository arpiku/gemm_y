// Stats.h — min/median/std/p95/CI summary of cudaEvent timing samples.
//
// Both Profiler.cu and the microbenches need to reduce a vector of
// per-iteration elapsed-ms samples to {min, median, std, p95, ci_low, ci_high}.
// Extracted here to eliminate the duplicated `summarize` logic.
//
// Statistical methodology:
//   - std: sample standard deviation (n-1 denominator).
//   - p95: 95th percentile (sort, index ceil(0.95*n)-1; n=50 -> index 47).
//   - 95% CI for the median: median ± 1.253 * std / sqrt(n). The 1.253
//     factor is the asymptotic ratio SE(median)/SE(mean).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gemm_y {
namespace bench {

struct TimedStats {
    double min_ns;
    double median_ns;
    double std_ns;       // sample standard deviation (n-1 denominator)
    double p95_ns;       // 95th percentile
    double ci_low_ns;    // 95% CI lower bound for the median
    double ci_high_ns;   // 95% CI upper bound for the median
};

// Reduce a vector of elapsed-ms samples (one per timed iteration) to
// {min, median, std, p95, ci_low, ci_high}. Sorts `samples` in place.
inline TimedStats summarize_ns(std::vector<float>& samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();

    TimedStats s;
    s.min_ns = static_cast<double>(samples.front()) * 1e6;
    s.median_ns = (n % 2 == 0)
        ? static_cast<double>(samples[n / 2 - 1] + samples[n / 2]) * 0.5 * 1e6
        : static_cast<double>(samples[n / 2]) * 1e6;

    // Sample standard deviation (n-1 denominator). Units: ns.
    double acc = 0.0;
    for (const float ms : samples) acc += static_cast<double>(ms) * 1e6;
    const double mean = acc / static_cast<double>(n);
    double sum_sq = 0.0;
    for (const float ms : samples) {
        const double v = static_cast<double>(ms) * 1e6;
        const double d = v - mean;
        sum_sq += d * d;
    }
    s.std_ns = (n > 1) ? std::sqrt(sum_sq / static_cast<double>(n - 1)) : 0.0;

    // p95: 95th percentile. Index = ceil(0.95 * n) - 1 (0-based).
    // With n=50: ceil(47.5) - 1 = 48 - 1 = 47 (the 48th sample).
    const std::size_t p95_idx =
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
    const std::size_t clamped = std::min(p95_idx, n - 1);
    s.p95_ns = static_cast<double>(samples[clamped]) * 1e6;

    // 95% CI for the median: median ± 1.253 * std / sqrt(n).
    // 1.253 = asymptotic SE(median) / SE(mean).
    constexpr double kMedianSeFactor = 1.253;
    const double se = (n > 1)
        ? kMedianSeFactor * s.std_ns / std::sqrt(static_cast<double>(n))
        : 0.0;
    s.ci_low_ns = s.median_ns - se;
    s.ci_high_ns = s.median_ns + se;

    return s;
}

} // namespace bench
} // namespace gemm_y
