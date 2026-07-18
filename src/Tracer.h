#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <ostream>
#include <span>
#include <string_view>

namespace tracer {

using clock = std::chrono::steady_clock;
using time_point = clock::time_point;

struct Event {
    std::string_view label;
    time_point timestamp;
};

template<std::size_t Capacity = 128>
class Timer {
    static_assert(Capacity > 0);

public:
    Timer() noexcept {
        event_markers_[0] = {"__Origin__", clock::now()};
        count_ = 1;
    }

    [[nodiscard]]
    bool mark(std::string_view label = "^") noexcept {
        if (count_ == Capacity) {
            ++dropped_events_;
            return false;
        }

        event_markers_[count_++] = {label, clock::now()};
        return true;
    }

    void reset() noexcept {
        count_ = 1;
        dropped_events_ = 0;
        event_markers_[0] = {"__Origin__", clock::now()};
    }

    [[nodiscard]]
    std::span<const Event> events() const noexcept {
        return {event_markers_.data(), count_};
    }

    [[nodiscard]]
    std::size_t dropped_events() const noexcept {
        return dropped_events_;
    }

    void print_traces(std::ostream& out) const;
    void print_traces_delta(std::ostream& out) const;

private:
    std::array<Event, Capacity> event_markers_{};
    std::size_t count_ = 0;
    std::size_t dropped_events_ = 0;
};

} // namespace tracer
