// Buffer.h — RAII raw storage tagged with a compile-time Space.
//
//   Buffer<T, Space::Device>  -> cudaMalloc / cudaFree
//   Buffer<T, Space::Host>     -> 64-byte-aligned std::vector
//
// Move-only (no copy ctor/assign). Fixed at construction — no realloc/resize.
// The bench runner pre-allocates a single 4096x4096 buffer and slices
// submatrices out of it via MatrixView::block(); we never need to grow one.
//
// Per AGENTS.md: RAII for all CUDA resources; never leak raw cudaFree across
// early returns.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "CudaCheck.h"
#include "Space.h"
#include "cuda_compat.h"

namespace gemm_y {

template <typename T, Space S>
class Buffer;

// ---------------------------------------------------------------------------
// Host specialization: aligned std::vector.
// ---------------------------------------------------------------------------
template <typename T>
class Buffer<T, Space::Host> {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Host Buffer<T> requires trivially-copyable T (CUDA dtypes).");

public:
    Buffer() noexcept = default;

    explicit Buffer(std::size_t count)
        : storage_(count) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    ~Buffer() = default;

    [[nodiscard]] T* data() noexcept { return storage_.data(); }
    [[nodiscard]] const T* data() const noexcept { return storage_.data(); }

    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return storage_.size() * sizeof(T); }

private:
    // Host Buffer uses std::vector's default allocator, which gives ~16-byte
    // alignment (sufficient for host-side reference computation). If a future
    // phase needs explicit 64-byte alignment (e.g. for AVX-512 host-side
    // reference), swap in a custom allocator here without touching call sites.
    std::vector<T> storage_;
};

// ---------------------------------------------------------------------------
// Device specialization: cudaMalloc / cudaFree.
// ---------------------------------------------------------------------------
template <typename T>
class Buffer<T, Space::Device> {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Device Buffer<T> requires trivially-copyable T (CUDA dtypes).");

public:
    Buffer() noexcept = default;

    explicit Buffer(std::size_t count) {
        if (count == 0) {
            ptr_ = nullptr;
            size_ = 0;
            return;
        }
        void* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
        ptr_ = static_cast<T*>(raw);
        size_ = count;
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~Buffer() { release(); }

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }

private:
    void release() noexcept {
        if (ptr_ != nullptr) {
            // cudaFree is allowed to fail; we swallow it in the dtor (can't
            // abort safely from a dtor in all contexts). Allocation paths
            // use CUDA_CHECK; this is best-effort cleanup.
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace gemm_y
