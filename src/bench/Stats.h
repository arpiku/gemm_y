// Stats.h — min/median summary of cudaEvent timing samples.
//
// Both Profiler.cu and the microbenches need to reduce a vector of
// per-iteration elapsed-ms samples to {min_ns, median_ns}. Extracted here
// to eliminate the duplicated `summarize` logic.

#pragma once

#include <algorithm>
#include <vector>

namespace gemm_y {
namespace bench {

struct TimedStats {
    double min_ns;
    double median_ns;
};

// Reduce a vector of elapsed-ms samples (one per timed iteration) to
// {min_ns, median_ns}. Sorts `samples` in place.
inline TimedStats summarize_ns(std::vector<float>& samples) {
    std::sort(samples.begin(), samples.end());
    TimedStats s;
    s.min_ns = static_cast<double>(samples.front()) * 1e6;
    const std::size_t n = samples.size();
    s.median_ns = (n % 2 == 0)
        ? static_cast<double>(samples[n / 2 - 1] + samples[n / 2]) * 0.5 * 1e6
        : static_cast<double>(samples[n / 2]) * 1e6;
    return s;
}

} // namespace bench
} // namespace gemm_y
