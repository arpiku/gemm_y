#pragma once

#include<iostream>
#include<chrono>
#include <ratio>

namespace Tracer {
    using hres_clk = std::chrono::high_resolution_clock;
    using time_pt = hres_clk::time_point;
    using nano_sec = std::chrono::nanoseconds;
    using micro_sec = std::chrono::microseconds;
    using milli_sec = std::chrono::milliseconds;

    enum Units {MILLI, MICRO, NANO};

    struct Event {
        std::string_view label;
        time_pt tp;
    };

    template<std::size_t Capacity = 128>
    class Timer {
        public:
            Timer() : count_(0) {
                event_markers_[count_] = Event{"__Origin__", hres_clk::now()};
            }
            void mark(std::string_view = "^"); // Random "^" symbol for marking without names
            void print_traces(); // Print out collected markers list, with time stamp as is
            void print_traces_delta(); // Print out collected markers list, with delta w.r.t previous marker
            void print_delta_events(std::size_t event_0_idx, std::size_t event_1_idx); // To print delta between given event indices

        private:
            std::array<Event,Capacity> event_markers_;
            std::size_t count_;
    };

    class Zone_Timer {
        public:
            Zone_Timer(Units unit) : start_(hres_clk::now()), u_(unit) {}
            ~Zone_Timer() {}
        private:
            time_pt start_;
            Units u_;
    };
};
