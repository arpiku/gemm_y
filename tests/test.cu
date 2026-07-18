// Build-verification target for gemm_y.
//
// Verifies that:
//   - the CUDA toolchain produces a working binary for the selected arch,
//   - the CUDA_ARCH_SM_* preprocessor define propagates from CMake -> nvcc,
//   - a trivial kernel round-trips data through device memory.
//
// Not a correctness/perf test. Intended to be replaced by real tests later.
/*  */
// CUDA runtime headers (cuda_runtime.h) use old-style casts in inline
// templates and trip -Wold-style-cast. Push diagnostics around the include
// so our own code stays under the warning policy.
#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

#include <cuda_runtime.h>

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(CUDA_ARCH_SM_90)
    #define GEMM_Y_TARGET_ARCH_NAME "sm_90"
#elif defined(CUDA_ARCH_SM_120)
    #define GEMM_Y_TARGET_ARCH_NAME "sm_120"
#else
    #error "Neither CUDA_ARCH_SM_90 nor CUDA_ARCH_SM_120 is defined."
#endif

__global__ void fill_kernel(int* data, int n, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

int main() {
    std::printf("test_cuda: compiled for %s\n", GEMM_Y_TARGET_ARCH_NAME);

    constexpr int kN = 256;
    std::vector<int> host(kN, 0);

    int* device = nullptr;
    if (cudaMalloc(&device, kN * sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return EXIT_FAILURE;
    }

    fill_kernel<<<(kN + 63) / 64, 64>>>(device, kN, 42);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "kernel launch / sync failed\n");
        cudaFree(device);
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(host.data(), device, kN * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D->H failed\n");
        cudaFree(device);
        return EXIT_FAILURE;
    }
    cudaFree(device);

    for (int i = 0; i < kN; ++i) {
        if (host[i] != 42) {
            std::fprintf(stderr, "mismatch at index %d: got %d, expected 42\n",
                         i, host[i]);
            return EXIT_FAILURE;
        }
    }

    std::printf("test_cuda: OK (%d elements verified)\n", kN);
    return EXIT_SUCCESS;
}
