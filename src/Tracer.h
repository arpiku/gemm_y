// Tracer.h — host-side timer using steady_clock.
//
// Host-only (steady_clock). Do NOT use for kernel timing; use CudaTimer
// (cudaEvent_t) for device-side timing including H2D/D2H. Tracer measures
// wall time including launch overhead (~5 us), which dominates for small
// kernels and is useless for sub-us device work. Use Tracer only for host
// orchestration (e.g. total sweep wall time, CSV write time).

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace tracer {

using clock = std::chrono::steady_clock; // monotonic — do not change
using time_point = clock::time_point;

struct Event {
    std::string_view label;     // caller MUST outlive the Timer
    time_point timestamp;
};

template<std::size_t Capacity = 128>
class Timer {
    static_assert(Capacity >= 2, "Timer needs room for origin + at least one mark");
    // Not thread-safe. Use one Timer per thread.

public:
    Timer() noexcept {
        event_markers_[0] = Event{"__Origin__", clock::now()};
        count_ = 1;
    }

    [[nodiscard]]
    bool mark(std::string_view label = "^") noexcept {
        if (count_ == Capacity) {
            ++dropped_events_;
            return false;
        }

        event_markers_[count_++] = Event{label, clock::now()};
        return true;
    }

    void reset() noexcept {
        count_ = 1;
        dropped_events_ = 0;
        event_markers_[0] = Event{"__Origin__", clock::now()};
    }

    [[nodiscard]]
    const Event* data() const noexcept {
        return event_markers_.data();
    }

    [[nodiscard]]
    std::size_t size() const noexcept {
        return count_;
    }

    [[nodiscard]]
    const Event& operator[](std::size_t i) const noexcept {
        return event_markers_[i];
    }

    [[nodiscard]]
    std::size_t dropped_events() const noexcept {
        return dropped_events_;
    }

    inline void print_traces(std::ostream& out) const;
    inline void print_traces_delta(std::ostream& out) const;

private:
    std::array<Event, Capacity> event_markers_{};
    std::size_t count_ = 1;
    std::size_t dropped_events_ = 0;
};

template<std::size_t Capacity>
inline void Timer<Capacity>::print_traces(std::ostream& out) const {
    // Absolute time since the Timer's origin, per event.
    const time_point origin = event_markers_[0].timestamp;
    for (std::size_t i = 0; i < count_; ++i) {
        const auto since_origin =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                event_markers_[i].timestamp - origin);
        out << "[" << i << "] " << event_markers_[i].label
            << " : " << since_origin.count() << " ns\n";
    }
    if (dropped_events_ != 0) {
        out << "[!] " << dropped_events_ << " event(s) dropped (capacity = "
            << Capacity << ")\n";
    }
}

template<std::size_t Capacity>
inline void Timer<Capacity>::print_traces_delta(std::ostream& out) const {
    // Delta from the previous event, per event.
    for (std::size_t i = 1; i < count_; ++i) {
        const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
            event_markers_[i].timestamp - event_markers_[i - 1].timestamp);
        out << "[" << i << "] " << event_markers_[i - 1].label
            << " -> " << event_markers_[i].label
            << " : " << delta.count() << " ns\n";
    }
    if (count_ <= 1) {
        out << "[i] no deltas to report (only origin event present)\n";
    }
    if (dropped_events_ != 0) {
        out << "[!] " << dropped_events_ << " event(s) dropped (capacity = "
            << Capacity << ")\n";
    }
}

} // namespace tracer
