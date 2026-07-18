// KernelTraits.h — concept-like SFINAE check for the kernel functor contract.
//
// A kernel K<T> must provide:
//   static constexpr std::string_view name()
//   static constexpr std::string_view description()
//   void operator()(GemmArgs<T>, cudaStream_t) const
//
// `description()` carries design info ("naive triple-loop", "wmma 16x16x16",
// "tiled 128x128 8-warps") that flows to CSV -> plot.py line labels.

#pragma once

#include <string_view>
#include <type_traits>

#include "GemmArgs.h"
#include "cuda_compat.h"

namespace gemm_y {

template <typename K, typename T, typename = void>
struct KernelTraits : std::false_type {};

template <typename K, typename T>
struct KernelTraits<K, T,
    std::void_t<
        decltype(K::name()),
        decltype(K::description()),
        decltype(std::declval<const K&>()(
            std::declval<GemmArgs<T>>(), std::declval<cudaStream_t>()))
    >> : std::true_type {
    static_assert(std::is_same_v<decltype(K::name()), std::string_view>,
                  "K::name() must return std::string_view");
    static_assert(std::is_same_v<decltype(K::description()), std::string_view>,
                  "K::description() must return std::string_view");
};

template <typename K, typename T>
inline constexpr bool KernelTraits_v = KernelTraits<K, T>::value;

} // namespace gemm_y
