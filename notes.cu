
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-object-layout
// mbarrier.init{.layout}{.shared{::cta}}.b64 [addr], count;
// .layout = { .layout::v0, .layout::v1 } (defaults to v0, if not specified)
/*  An mbarrier always tracks the following:
 *  - Current primary and conditinal phases of the barrier (primary/conditional)
 *  - Count of Pending Arrivals (N)
 *  - Count of expeceted Arrivals for next phase (N')
 *  - Count of Pending Async memory operations (or transactions, "tx-count")
 *  - Count of Pending Async memory operations (or transactions, "tx-count")
 *  .layout (v1) will additionally track:
 *  - Payload report corresponding  primary phase
 */
__device__ inline void mbarrier_init(uint64_t *barrier) {
  const unsigned address = __cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
               :
               : "r"(address)
               : "memory");
}

{
  cuda::ptx::mbarrier_init(&bar, block.size());
}

// Release local barrier state via generic proxy.
{
  mbarrier.init[bar];
  fence.mbarrier_init.release.cluster;
  barrier.cluster.arrive.relaxed;
}
// NOTE:
/* The semantics have been updated (following from library code
 * /usr/include/cub/cccl.../kernel_transform.cuh) */

{
  if (cuda::device::__block_elect_one()) {
    ptx::mbarrier_init(&bar, 1);
    // an update to the CUDA memory model blesses skipping the following fence
    // ptx::fence_proxy_async(ptx::space_shared); --> Hmmmmm

    char *smem = smem_base;
    ::cuda::std::uint32_t total_copied = 0;
  }

// TMA Operation PTX
// global -> shared::cta
{ cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache_policy}

.dst =                  { .shared::cta }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 } // The 1 is when mbar resides in the same memory as the dst, 2 when the mbar may be in peer CTA (can still be in same CTA smem)
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }


// global -> shared::cluster
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
                                   {, ctaMask} {, cache_policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }


// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
                                   [tensorMap, tensorCoords], [srcMem] {, cache_policy}

.dst =                  { .global }
.src =                  { .shared::cta }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode =            { .tile, .tile::scatter4, .im2col_no_offs }
.level::cache_hint =    { .L2::cache_hint }
}

// Release and acquire semantics
/*
The release pattern makes prior operations from the current thread visible to some operations from other threads.
The acquire pattern makes some operations from other threads visible to later operations from the current thread.
*/
// Ref:  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-acquire-patterns


// There are two completion mechanisms: mbarrier and async group
// Ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms
// Async Group semantics ~ init, commit, wait, access
// mbarrier semantics ~ init bar, init async op, test barrier, when test succeeds do work



// cp.async.bulk with async group semantics
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-bulk-tensor-copy-completion
{
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group{.read} N; // Will wait till until only N or fewer of the most recent operations remain
}

// mbarrier
// An mbarrier object is an opaque object in memory which can be initialized and invalidated using :
{
    mbarrier.init // 2.5
    mbarrier.inval // 3.2
}

// Operations supported on mbarrier objects are :
{
// Instruction -> Availability in libc++ (CCCL/CUDA ver)
1. mbarrier.expect_tx -> Yes // 2.8 / 12.9
2. mbarrier.complete_tx -> NO
3. mbarrier.arrive -> Yes // 2.3 / 12.4

4. mbarrier.arrive_drop --> No

5. mbarrier.test_wait --> Yes // 2.3 /12.4
6. mbarrier.try_wait --> Yes // 2.3 / 12.4

7. mbarrier.pending_count --> No
8. mbarrier.check_layout --> ??

9. cp.async.mbarrier.arrive -> // 2.8 / 12.9
}

// Variant of bulk async copy
{ if (is_elected()) {
  // Launch the async copy and communicate how many bytes are expected to come in (the transaction count).

  // Version 1: cuda::memcpy_async
  cuda::memcpy_async( // CUDA 11.0
      smem_data, data + offset,
      cuda::aligned_size_t<16>(sizeof(smem_data)),
      bar);

//  Version 2: cuda::device::memcpy_async_tx
  cuda::device::memcpy_async_tx( // CUDA 11.1
    smem_data, data + offset,
    cuda::aligned_size_t<16>(sizeof(smem_data)),
    bar);
  cuda::device::barrier_expect_tx(
      cuda::device::barrier_native_handle(bar),
      sizeof(smem_data));

// Version 3: cuda::ptx::cp_async_bulk
  ptx::cp_async_bulk( // CUDA 12.5
      ptx::space_shared, ptx::space_global,
      smem_data, data + offset,
      sizeof(smem_data),
      cuda::device::barrier_native_handle(bar));
      cuda::device::barrier_expect_tx(
      cuda::device::barrier_native_handle(bar),
      sizeof(smem_data));
}

// Further `cp.async` variants
// https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions.html
// Instruction -> Available in libcu++
cp.async -> No
cp.async.commit_group -> No
cp.async.wait_group -> No

cp.async.bulk -> Yes // CCCL 2.4.0 / CUDA 12.5
cp.reduce.async.bulk -> Yes // CCCL 2.4.0 / CUDA 12.5

cp.async.bulk.prefetch -> No
cp.async.bulk.prefetch.tensor ->  No

cp.reduce.async.bulk ->  Yes //CCCL 2.4.0 / CUDA 12.5
cp.reduce.async.bulk.tensor -> Yes // CCCL 2.4.0 / CUDA 12.5

cp.async.bulk.commit_group --> Yes //CCCL 1.4.0 / CUDA 12.5
cp.async.bulk.wait_group --> Yes //CCCL 2.4.0 / CUDA 12.5

tensormap.replace -> Yes // CCCL 2.4.0 / CUDA 12.5


// Pipeline object
// Introduces in CUDA 11.0
{
template <cuda::thread_scope Scope>
class cuda::pipeline {
public:
  pipeline() = delete;
  __host__ __device__ ~pipeline();
  pipeline& operator=(pipeline const&) = delete;
  __host__ __device__ void producer_acquire();
  __host__ __device__ void producer_commit();
  __host__ __device__ void consumer_wait();
  template <typename Rep, typename Period>
  __host__ __device__ bool consumer_wait_for(cuda::std::chrono::duration<Rep, Period> const& duration);
  template <typename Clock, typename Duration>
  __host__ __device__
  bool consumer_wait_until(cuda::std::chrono::time_point<Clock, Duration> const& time_point);
  __host__ __device__ void consumer_release();
  __host__ __device__ bool quit();
};
