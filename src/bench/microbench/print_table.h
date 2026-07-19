// print_table.h — tiny helper for aligned fixed-width column output.
//
// Microbenches are one-off and human-readable; raw CSV-to-stdout is hard to
// scan. This helper accumulates rows of strings and prints aligned columns
// with a header row + separator. Units go in the header (e.g. "min (ns)").
//
// Usage:
//   gemm_y::bench::Table t({"direction", "variant", "N", "min (ns)", "median (ns)"});
//   t.add_row({"h2d", "contiguous_sync", "32", "4480", "4800"});
//   t.print();

#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace gemm_y {
namespace bench {

class Table {
public:
    explicit Table(std::vector<std::string> header) : header_(std::move(header)) {}

    void add_row(std::vector<std::string> row) {
        rows_.push_back(std::move(row));
    }

    void print() const {
        const std::size_t ncols = header_.size();
        std::vector<std::size_t> widths(ncols, 0);
        for (std::size_t c = 0; c < ncols; ++c) {
            widths[c] = header_[c].size();
        }
        for (const auto& r : rows_) {
            for (std::size_t c = 0; c < ncols && c < r.size(); ++c) {
                if (r[c].size() > widths[c]) widths[c] = r[c].size();
            }
        }

        // Header row (left-aligned).
        for (std::size_t c = 0; c < ncols; ++c) {
            if (c > 0) std::fputs("  ", stdout);
            std::fputs(header_[c].c_str(), stdout);
            for (std::size_t i = header_[c].size(); i < widths[c]; ++i) std::fputc(' ', stdout);
        }
        std::fputc('\n', stdout);

        // Separator.
        for (std::size_t c = 0; c < ncols; ++c) {
            if (c > 0) std::fputs("  ", stdout);
            for (std::size_t i = 0; i < widths[c]; ++i) std::fputc('-', stdout);
        }
        std::fputc('\n', stdout);

        // Data rows. Right-align numeric-looking cells (first char is digit,
        // '+', '-', or '.'); left-align the rest.
        for (const auto& r : rows_) {
            for (std::size_t c = 0; c < ncols && c < r.size(); ++c) {
                if (c > 0) std::fputs("  ", stdout);
                const std::string& cell = r[c];
                const bool numeric = !cell.empty() &&
                    (cell[0] == '+' || cell[0] == '-' || cell[0] == '.' ||
                     (cell[0] >= '0' && cell[0] <= '9'));
                if (numeric) {
                    for (std::size_t i = cell.size(); i < widths[c]; ++i) std::fputc(' ', stdout);
                    std::fputs(cell.c_str(), stdout);
                } else {
                    std::fputs(cell.c_str(), stdout);
                    for (std::size_t i = cell.size(); i < widths[c]; ++i) std::fputc(' ', stdout);
                }
            }
            std::fputc('\n', stdout);
        }
    }

private:
    std::vector<std::string> header_;
    std::vector<std::vector<std::string>> rows_;
};

} // namespace bench
} // namespace gemm_y
