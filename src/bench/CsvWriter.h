// CsvWriter.h — minimal CSV writer (no external deps).
//
// Opens the file in the ctor, writes a header row, appends rows via
// append_row(...). Flushes on destruction. Field escaping is minimal:
// we control all callers (bench harness), so we don't embed commas in
// kernel descriptions.

#pragma once

#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>

namespace gemm_y {

class CsvWriter {
public:
    CsvWriter() = default;

    explicit CsvWriter(const std::string& path) : out_(path, std::ios::out | std::ios::trunc) {}

    ~CsvWriter() {
        if (out_.is_open()) out_.flush();
    }

    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;
    CsvWriter(CsvWriter&&) noexcept = default;
    CsvWriter& operator=(CsvWriter&&) noexcept = default;

    bool open(const std::string& path) {
        out_.open(path, std::ios::out | std::ios::trunc);
        return out_.is_open();
    }

    bool is_open() const noexcept { return out_.is_open(); }

    void write_header(std::string_view header) {
        out_ << header << '\n';
    }

    // Variadic append_row: writes each arg separated by commas, then a newline.
    template <typename... Args>
    void append_row(const Args&... args) {
        append_impl(args...);
        out_ << '\n';
    }

private:
    template <typename T>
    static void write_one(std::ofstream& o, const T& v) {
        if constexpr (std::is_same_v<T, bool>) {
            o << (v ? "true" : "false");
        } else if constexpr (std::is_same_v<T, std::string> ||
                             std::is_same_v<T, std::string_view>) {
            o << v;
        } else if constexpr (std::is_floating_point_v<T>) {
            // Use enough precision to round-trip doubles (e.g. 1e-9 rel err).
            o.setf(std::ios::scientific);
            o.precision(15);
            o << v;
            o.unsetf(std::ios::scientific);
        } else {
            o << v;
        }
    }

    void append_impl() const {}

    template <typename Head, typename... Tail>
    void append_impl(const Head& head, const Tail&... tail) {
        write_one(out_, head);
        if constexpr (sizeof...(tail) > 0) {
            out_ << ',';
            append_impl(tail...);
        }
    }

    std::ofstream out_;
};

} // namespace gemm_y
