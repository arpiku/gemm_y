// tests/kernel_smoke.cu — fast smoke test for newly added sm120 bf16 kernels.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "Buffer.h"
#include "CudaCheck.h"
#include "MatrixView.h"
#include "bench/Accuracy.h"
#include "bench/Profiler.h"
#include "cublas/cublas_gemm.h"
#include "dtypes.h"

#if defined(CUDA_ARCH_SM_90)
#error "kernel_smoke currently targets sm_120 bf16 experiments only"
#elif defined(CUDA_ARCH_SM_120)
#include "sm120/gemm_bf16.cuh"
#else
#error "kernel_smoke requires a supported CUDA architecture"
#endif

template <typename Kernel> bool check_padded_kernel(const char *label) {
  using T = gemm_y::dtypes::bf16;
  constexpr int M = 37;
  constexpr int N = 53;
  constexpr int K = 29;
  constexpr int ldA = M + 5;
  constexpr int ldB = K + 7;
  constexpr int ldC = M + 9;

  gemm_y::Buffer<T, gemm_y::Space::Host> hA(static_cast<std::size_t>(ldA) * K);
  gemm_y::Buffer<T, gemm_y::Space::Host> hB(static_cast<std::size_t>(ldB) * N);
  gemm_y::Buffer<T, gemm_y::Space::Host> hRef(static_cast<std::size_t>(ldC) *
                                              N);
  gemm_y::Buffer<T, gemm_y::Space::Host> hGot(static_cast<std::size_t>(ldC) *
                                              N);
  gemm_y::MatrixView<T, gemm_y::Space::Host> hAv(hA.data(), M, K, ldA);
  gemm_y::MatrixView<T, gemm_y::Space::Host> hBv(hB.data(), K, N, ldB);
  gemm_y::MatrixView<T, gemm_y::Space::Host> hRefv(hRef.data(), M, N, ldC);
  gemm_y::MatrixView<T, gemm_y::Space::Host> hGotv(hGot.data(), M, N, ldC);
  for (int j = 0; j < K; ++j)
    for (int i = 0; i < M; ++i)
      hAv(i, j) = static_cast<T>(((i + 2 * j) % 7) - 3);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < K; ++i)
      hBv(i, j) = static_cast<T>(((3 * i - j) % 7) - 3);

  gemm_y::Buffer<T, gemm_y::Space::Device> dA(hA.size());
  gemm_y::Buffer<T, gemm_y::Space::Device> dB(hB.size());
  gemm_y::Buffer<T, gemm_y::Space::Device> dRef(hRef.size());
  gemm_y::Buffer<T, gemm_y::Space::Device> dGot(hGot.size());
  CUDA_CHECK(
      cudaMemcpy(dA.data(), hA.data(), hA.bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(dB.data(), hB.data(), hB.bytes(), cudaMemcpyHostToDevice));

  gemm_y::MatrixView<T, gemm_y::Space::Device> dAv(dA.data(), M, K, ldA);
  gemm_y::MatrixView<T, gemm_y::Space::Device> dBv(dB.data(), K, N, ldB);
  gemm_y::MatrixView<T, gemm_y::Space::Device> dRefv(dRef.data(), M, N, ldC);
  gemm_y::MatrixView<T, gemm_y::Space::Device> dGotv(dGot.data(), M, N, ldC);
  gemm_y::CublasHandle handle;
  gemm_y::cublas_gemm(handle, dAv, dBv, dRefv);
  Kernel{}(gemm_y::GemmArgs<T>{dAv, dBv, dGotv}, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef.data(), hRef.bytes(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hGot.data(), dGot.data(), hGot.bytes(),
                        cudaMemcpyDeviceToHost));

  const gemm_y::MatrixView<const T, gemm_y::Space::Host> got(hGot.data(), M, N,
                                                             ldC);
  const gemm_y::MatrixView<const T, gemm_y::Space::Host> ref(hRef.data(), M, N,
                                                             ldC);
  const auto errors = gemm_y::compare<T>(got, ref);
  std::printf("kernel smoke padded %s: max_rel=%.3e\n", label, errors.max_rel);
  return errors.max_rel <= gemm_y::kRelErrTol<T>();
}

int main() {
  using T = gemm_y::dtypes::bf16;
  const std::vector<int> sizes = {32, 96, 128};
  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k0_ldg>();
  profiler.register_kernel<gemm_y::k0_coalesced>();
  profiler.register_kernel<gemm_y::k1_smem>();
  profiler.register_kernel<gemm_y::k1_t<64, 64, 16>>();

  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 5;
  const std::size_t expected_rows = sizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected_rows) {
    std::fprintf(stderr, "kernel smoke failed: expected %zu rows, got %zu\n",
                 expected_rows, result.rows.size());
    return EXIT_FAILURE;
  }

  for (const int size : sizes) {
    bool has_cublas = false;
    bool has_ldg = false;
    bool has_coalesced = false;
    bool has_smem = false;
    bool has_tiled = false;
    for (const auto &row : result.rows) {
      if (row.N != size)
        continue;
      if (row.kernel_name == "cublas")
        has_cublas = true;
      if (row.kernel_name == "k0_ldg")
        has_ldg = true;
      if (row.kernel_name == "k0_coalesced")
        has_coalesced = true;
      if (row.kernel_name == "k1_smem")
        has_smem = true;
      if (row.kernel_name == "k1_t")
        has_tiled = true;
    }
    if (!has_cublas || !has_ldg || !has_coalesced || !has_smem || !has_tiled) {
      std::fprintf(stderr,
                   "kernel smoke failed at N=%d: cublas=%d k0_ldg=%d "
                   "k0_coalesced=%d k1_smem=%d k1_t=%d\n",
                   size, has_cublas, has_ldg, has_coalesced, has_smem,
                   has_tiled);
      return EXIT_FAILURE;
    }
  }

  if (!check_padded_kernel<gemm_y::k1_smem>("k1_smem") ||
      !check_padded_kernel<gemm_y::k1_t<64, 64, 16>>("k1_t")) {
    std::fprintf(stderr, "kernel smoke failed: padded kernel mismatch\n");
    return EXIT_FAILURE;
  }

  std::printf("kernel smoke passed: %zu rows across %zu sizes\n",
              result.rows.size(), sizes.size());
  return EXIT_SUCCESS;
}
