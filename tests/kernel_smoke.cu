// tests/kernel_smoke.cu — focused BF16 correctness and dispatch smoke test.

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
#include "sm90/gemm_bf16.cuh"
#elif defined(CUDA_ARCH_SM_100)
#include "sm100/gemm_bf16.cuh"
#include "sm100/k1_dispatch.cuh"
#elif defined(CUDA_ARCH_SM_120)
#include "sm120/gemm_bf16.cuh"
#include "sm120/k1_dispatch.cuh"
#else
#error "kernel_smoke requires a supported CUDA architecture"
#endif

template <typename Kernel, int MValue = 37, int NValue = 64, int KValue = 29,
          int LdAValue = MValue + 5, int LdBValue = KValue + 7,
          int LdCValue = MValue + 9>
bool check_padded_kernel(const char *label) {
  using T = gemm_y::dtypes::bf16;
  constexpr int M = MValue;
  constexpr int N = NValue;
  constexpr int K = KValue;
  constexpr int ldA = LdAValue;
  constexpr int ldB = LdBValue;
  constexpr int ldC = LdCValue;

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
#if defined(CUDA_ARCH_SM_90)
  const std::vector<int> sizes = {32, 96, 128};
  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k0>();
  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 2;
  if (result.rows.size() != sizes.size() * kExpectedRowsPerN)
    return EXIT_FAILURE;
  for (const int size : sizes) {
    bool has_cublas = false;
    bool has_k0 = false;
    for (const auto &row : result.rows) {
      if (row.N != size)
        continue;
      has_cublas |= row.kernel_name == "cublas";
      has_k0 |= row.kernel_name == "k0";
    }
    if (!has_cublas || !has_k0)
      return EXIT_FAILURE;
  }
  if (!check_padded_kernel<gemm_y::k0>("k0"))
    return EXIT_FAILURE;
  std::printf("sm90 kernel smoke passed: %zu rows across %zu sizes\n",
              result.rows.size(), sizes.size());
  return EXIT_SUCCESS;
#elif defined(CUDA_ARCH_SM_100)
  const std::vector<int> sizes = {256, 512};
  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k1_dispatch>();
  profiler.register_kernel<gemm_y::k_tcgen05x>();
  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 3;
  for (const int size : sizes) {
    int count = 0;
    bool has_cublas = false;
    bool has_dispatch = false;
    bool has_tcgen = false;
    for (const auto &row : result.rows) {
      if (row.N != size)
        continue;
      ++count;
      has_cublas |= row.kernel_name == "cublas";
      has_dispatch |= row.kernel_name == "k1_dispatch";
      has_tcgen |= row.kernel_name == "k_tcgen05x";
    }
    if (count != static_cast<int>(kExpectedRowsPerN) || !has_cublas ||
        !has_dispatch || !has_tcgen) {
      std::fprintf(stderr, "sm100 kernel smoke failed at N=%d\n", size);
      return EXIT_FAILURE;
    }
  }
  std::printf("sm100 kernel smoke passed: %zu rows across %zu sizes\n",
              result.rows.size(), sizes.size());
  return EXIT_SUCCESS;
#else
  const std::vector<int> sizes = {32, 96, 128};
  gemm_y::Profiler<T> profiler;
  profiler.register_kernel<gemm_y::k0_ldg>();
  profiler.register_kernel<gemm_y::k0_coalesced>();
  profiler.register_kernel<gemm_y::k1_smem>();
  profiler.register_kernel<gemm_y::k1_dispatch>();
  const gemm_y::SweepResult result = profiler.run_sweep(sizes);
  constexpr std::size_t kExpectedRowsPerN = 5;
  const std::size_t expected_rows = sizes.size() * kExpectedRowsPerN;
  if (result.rows.size() != expected_rows)
    return EXIT_FAILURE;
  for (const int size : sizes) {
    bool has_cublas = false;
    bool has_ldg = false;
    bool has_coalesced = false;
    bool has_smem = false;
    bool has_dispatch = false;
    for (const auto &row : result.rows) {
      if (row.N != size)
        continue;
      has_cublas |= row.kernel_name == "cublas";
      has_ldg |= row.kernel_name == "k0_ldg";
      has_coalesced |= row.kernel_name == "k0_coalesced";
      has_smem |= row.kernel_name == "k1_smem";
      has_dispatch |= row.kernel_name == "k1_dispatch";
    }
    if (!has_cublas || !has_ldg || !has_coalesced || !has_smem || !has_dispatch)
      return EXIT_FAILURE;
  }
  if (!check_padded_kernel<gemm_y::k1_smem>("k1_smem") ||
      !check_padded_kernel<gemm_y::k1_dispatch>("k1_dispatch") ||
      !check_padded_kernel<gemm_y::k2_tma, 64, 64, 64, 72, 72, 73>("k2_tma"))
    return EXIT_FAILURE;
  if (gemm_y::k1_dispatch::selected_kernel_name(999) != "k1_smem")
    return EXIT_FAILURE;
  std::printf("kernel smoke passed: %zu rows across %zu sizes\n",
              result.rows.size(), sizes.size());
  return EXIT_SUCCESS;
#endif
}
