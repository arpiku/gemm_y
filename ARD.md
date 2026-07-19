# ARD.md — Architecture Decision Record

> Records the architectural thinking behind `gemm_y`. Each section captures
> a decision, the alternatives considered, and the rationale. Update when
> decisions change; do not delete historical context — strike through and
> supersede.

## Table of contents

1.  Memory model: ownership / shape / space, separated
2.  Memcpy variant selection
3.  Kernel abstraction: functors, not function pointers
4.  Profiler: type-erased registry, cuBLAS implicit
5.  Bench runner: pre-alloc + submatrix slicing
5.5. cuBLAS handle: stateful context, per-call stream binding
6.  Accuracy tolerance
7.  Timing: `CudaTimer` (device) vs `Tracer` (host)
8.  Arch-specific code: separate `.cu` files, no `#ifdef`
9.  cuBLAS API choice: `cublasGemmEx` first
10. Phase 1 validation
11. Iteration policy: warmup / timed counts
12. Phase 2 plan
13. Phase 1.5 refactor inventory
14. Phase 1.5 validation

---

## 1. Memory model: ownership / shape / space, separated

### Decision
Three orthogonal concerns are modeled by three types:

| Type | Role | Ownership |
|------|------|-----------|
| `Buffer<T, Space>` | raw storage | owning (RAII) |
| `MatrixView<T, Space>` | `{ptr, rows, cols, ld, Layout}` | non-owning |
| `Matrix<T, Space>` | `Buffer` + shape | owning |

`Space` (`Host` / `Device`) and `Layout` (`ColMajor` / `RowMajor`) are
**compile-time** template/enum tags, not runtime fields.

**`MatrixView` dual-use contract:** the type serves two roles:
1. **Host-side view** — `block(r,c,m,n)` (zero-copy sub-view),
   `operator()(i,j)` (element access), `is_contiguous()` (copy dispatch),
   converting ctor `MatrixView<T,S> -> MatrixView<const T,S>` (const-correctness).
2. **Kernel-side POD descriptor** — only `ptr`/`rows`/`cols`/`ld` are read
   directly (`A.ptr[i + k*A.ld]`). The host methods are **not**
   `__device__`-callable; kernels read fields directly. This is intentional:
   the kernel knows its layout at compile time (hardcoded ColMajor in Phase 1)
   and bypasses the runtime `layout` branch that `operator()` would incur.

### Alternatives considered
- **Runtime memory-space tag**: rejected. Loses compile-time dispatch of
  `copy_*`; risks calling `cudaMemcpy` on a host pointer from a device kernel
  with no compile error. The tag costs nothing (1 byte in the view) and
  buys type-level safety.
- **Single `Tensor` type owning data + shape** (à la `torch::Tensor`):
  rejected. Conflates ownership with shape; submatrix would either copy or
  require a separate non-owning type anyway. The `Matrix`/`MatrixView` split
  mirrors `std::vector`/`std::span` and is the CUTLASS/CuTe convention.
- **Layout as runtime field only**: rejected. Compile-time tag enables future
  template specializations per layout without runtime branches in hot paths.
  ColMajor is the default; RowMajor is reserved but not implemented in Phase 1.

### Consequences
- Kernels take `MatrixView<T, Space::Device>` (or unpacked `ptr+rows+cols+ld`)
  — never raw `T* + N`. Enforces `ld`-awareness from day 1.
- Submatrix slicing (`block(r,c,m,n)`) is zero-copy: returns a view with
  unchanged `ld` and offset `ptr`. This is what lets the bench runner
  pre-allocate a single 4096×4096 buffer and feed submatrices to every kernel.
- `copy_h2d` / `copy_d2h` are the **only** functions that touch `cudaMemcpy*`.
  No hidden data movement anywhere else in the codebase.
- `GemmArgs<T>` const-correctness: `A`/`B` are `MatrixView<const T, Device>`
  (read-only inputs), `C` is `MatrixView<T, Device>` (mutable output). Relies
  on `MatrixView`'s implicit converting ctor — zero call-site churn. `cublas_gemm`
  is the exception: it takes writable views for A/B because C++ template
  argument deduction does not consider implicit conversions (see §3).

---

## 2. Memcpy variant selection

### Decision
- **Contiguous copies** (`ld == rows` for ColMajor): `cudaMemcpy` (sync).
- **Strided submatrix copies** (`ld > rows`): `cudaMemcpy2D`.
- **Async path** (`cudaMemcpyAsync` on explicit stream): not wired in Phase 1.
- **Direction dispatch**: `detail::copy_kind_v<Dst, Src>` is a `constexpr`
  variable template mapping the `(Dst, Src)` Space pair to the
  `cudaMemcpyKind` enum. Only `Host<->Device` specializations are defined;
  a `static_assert` inside `detail::copy` rejects wrong-direction
  instantiations (e.g. `copy<Host, Host>`) at compile time. The poison
  primary template (`cudaMemcpyKind(-1)`) is a defensive secondary net.
  This makes copy direction a compile-time property of the Space tags,
  not a runtime argument.

### Rationale
- Sync vs async (default stream + sync) are **identical perf** when there is no
  overlap with compute. Async only wins when H2D/kernel/D2H are pipelined on
  a non-default stream — which is a Phase 2 concern (tiled kernels with
  double-buffering). Wiring it now is YAGNI.
- `cudaMemcpy2D` is **mandatory** for strided submatrix copies. `cudaMemcpy`
  would read contiguous bytes spanning rows, producing garbage. This is the
  correctness floor for the bench runner's 4096-ld buffer feeding N×N kernels.
- The bench runner isolates timings (no overlap), so sync is both simpler and
  sufficient.

### Validation
`src/bench/memcpy_microbench.cu` (Chunk 2.2) measures all variants across the
size sweep. Results recorded here once run:

```
# Run: ./build/memcpy_microbench > results/microbench_memcpy.csv
# Hardware: RTX 5070 (sm_120), CUDA 13.2, driver 2026-03.
# Warmup=20, timed=50, reporting min_ns / median_ns.
#
# Key observations (full CSV in results/microbench_memcpy.csv):
# 1. sync vs async_default_stream: indistinguishable at every N (within
#    event-timing noise ~1 us). Confirms the hypothesis — no overlap without
#    explicit stream pipelining. Async wiring is YAGNI for Phase 1.
# 2. contiguous_sync vs strided_2d: at small N (32-256) the strided path is
#    marginally slower (~5-15%) due to per-row pitch setup; at large N
#    (>=1024) they converge as the copy becomes bandwidth-bound. The strided
#    path is mandatory for correctness on the 4096-ld submatrix bench, so
#    the small-N overhead is the price of correct submatrix copies.
# 3. H2D vs D2H: D2H is ~1.5x slower than H2D at large N (PCIe asymmetry),
#    consistent with RTX 5070's host interface. Not actionable for Phase 1.
#
# Launch overhead (Chunk 2.4): min=1184 ns, median=2112 ns, max=8288 ns
# over 1000 empty-kernel launches. Sets the floor for kernel time
# interpretation: any kernel reporting < ~1.2 us is measurement noise.
```

### Phase 1.5 correction — revert to sync
The Phase 1 implementation used `cudaMemcpyAsync` + `cudaStreamSynchronize`
on the default stream, believing this was forward-compatible with Phase 2
explicit streams. **This was wrong**: `cudaMemcpyAsync` on pageable host
memory (which `std::vector<T>` gives) is **silently synchronous** — the
runtime stages through an internal pinned buffer (extra host-side copy),
then issues the DMA. The microbench confirmed `sync ≈ async` because both
are sync; the async path adds staging overhead for no benefit.

**Phase 1.5 fix (R1):** revert `Copy.h` to sync `cudaMemcpy` /
`cudaMemcpy2D`. Drop the `cudaStream_t` parameter and the
`cudaStreamSynchronize` calls. True async is deferred to Phase 2 prep,
where `Space::HostPinned` + `cudaHostAlloc` enables real overlap on an
explicit stream. The `copy_h2d` / `copy_d2h` API regains the stream
parameter only when pinned-ness is enforced at the call site.

---

## 3. Kernel abstraction: functors, not function pointers

### Decision
Every GEMM kernel is a **functor type** satisfying `KernelTraits`:

```cpp
template <typename T>
struct NaiveGemm {
    static constexpr std::string_view name()        { return "naive"; }
    static constexpr std::string_view description() { return "triple-loop, 1 thread/element, fp32 accum"; }
    void operator()(GemmArgs<T> args, cudaStream_t s = nullptr) const;
};
```

`GemmArgs<T>` is a POD (`{MatrixView<const T,Device> A, B,
MatrixView<T,Device> C}`), passed by value. `A`/`B` are read-only inputs
(`MatrixView<const T, ...>`); `C` is the mutable output
(`MatrixView<T, ...>`). The const contract is enforced at the `GemmArgs`
level via `MatrixView`'s implicit converting ctor — kernels and `Profiler`
construct `GemmArgs` from writable views with no call-site churn.

`cublas_gemm` is the **exception**: it takes writable `MatrixView<T, Device>`
for A/B because C++ template argument deduction does not consider implicit
conversions — `MatrixView<T,S> -> MatrixView<const T,S>` would fail deduction
at every call site. The const contract for the reference path is documented
in `cublas_gemm.h` rather than enforced by the type system.

### Alternatives considered
- **Raw function pointers** (`void(*)(T*,T*,T*,int,int,int)`): rejected.
  - Loses metadata (`name`, `description`) needed for CSV/plotter labels.
  - No room for compile-time config (tile size, warp count) without runtime args.
  - Cannot carry state if a kernel later needs precomputed params.
- **Virtual base class** (`IGemm`): rejected. vtable dispatch (~2 ns) is
  negligible vs kernel launch (~5 µs), but adds a header, a vtable indirection,
  and forces heap allocation for stateful kernels. No upside over functors.
- **Type-erased `std::function` everywhere**: rejected as the *primary* API.
  Used internally by `Profiler` for storage (see §4), but the user-facing
  registration API is templated on the functor type — preserving inlining and
  SBO for stateless kernels.

### Consequences
- `Profiler::register_kernel<K>()` type-erases into `std::function` once, at
  registration. Stateless `K` (the common case) fits SBO — zero heap.
  Stateful `K` exceeding SBO pays one heap alloc at registration, amortized
  over the entire sweep. Negligible vs launch cost.
- `description()` is the design-info channel ("wmma 16×16×16", "tiled 128×128
  8-warps"). Flows: kernel → `Profiler` → CSV → `plot.py` line labels.
- Compile-time config (e.g. `TiledGemm<128,8>`) is natural via template params.
- Future extension to `alpha`/`beta` is non-breaking: add fields to `GemmArgs`
  with defaults; existing kernels ignore them.

---

## 4. Profiler: type-erased registry, cuBLAS implicit

### Decision
```cpp
template <typename T>
class Profiler {
    struct Entry {
        std::string name, description;
        std::function<void(GemmArgs<T>, cudaStream_t)> run;
    };
    std::vector<Entry> kernels_;
    CublasHandle cublas_;  // owned
    // ... pre-allocated 4096×4096 A, B, C_max buffers
    void run_sweep(std::vector<int> sizes, std::string csv_path);
};
```

cuBLAS is **not** registered as a kernel — it is the implicit reference, run
first per `N`, cached on device, and used as the accuracy ground truth.

### Rationale
- cuBLAS is the success metric (AGENTS.md); treating it as just-another-kernel
  would either (a) recompute it per kernel-under-test (waste) or (b) require
  special-casing anyway. Making it implicit is cleaner.
- Per-`N` cuBLAS caching avoids re-running the reference for each registered
  kernel. cuBLAS bf16 accumulates in fp32 by default — natural ground truth.
- One `Profiler<T>` instance per dtype. Phase 1 wires `bf16` only; `fp16`/`fp32`
  are Phase 2+ and get their own `Profiler` instance in `main.cpp`.

### Overhead analysis
- `std::function` indirect call: ~1 ns. Kernel launch: ~5 µs. Kernel runtime:
  µs–ms. Indirection is ~0.02% of launch overhead, ~0.0001% of total. Not a
  perf concern; do not optimize prematurely.
- SBO (small-buffer optimization) in libstdc++/libc++ is 16–24 bytes — enough
  for any stateless functor. Stateful functors exceeding SBO pay one heap
  alloc at `register_kernel` time, amortized across the full sweep.

### Phase 1.5 decoupling — `run_sweep` returns `SweepResult`
Phase 1 coupled benchmarking and serialization: `run_sweep(sizes, csv_path)`
opened a `CsvWriter` internally and wrote rows as it ran. This conflated
GPU work with I/O side effects, made `run_sweep` untestable in isolation,
and buried the CSV schema inside bench logic.

**Phase 1.5 fix (R3):** `run_sweep` returns `SweepResult { std::vector<SweepRow>
rows; }`. `main.cpp` owns the `CsvWriter` and iterates `result.rows`.
`Profiler.cu` drops `#include "CsvWriter.h"`. The CSV schema is now
defined in one place (`main.cpp`), and `run_sweep` is testable: a unit
test can register a kernel, sweep `{32, 64}`, and assert row counts +
accuracy without touching disk.

**Phase 1.5 fix (R4):** cuBLAS is measured **once per N**, stored in
`SweepResult`, and its `ref_*` columns are reused for every kernel at
that N. The Phase 1 implementation re-timed cuBLAS inside the per-kernel
loop (K × S × kTimed extra launches) — a harness perf bug, not a kernel
bug. After R3, the cuBLAS row is simply the first row per N in
`SweepResult`; subsequent kernel rows reference its `kernel_min_ns` /
`kernel_median_ns`.

---

## 5. Bench runner: pre-alloc + submatrix slicing

### Decision
- Pre-allocate 4096×4096 `Matrix<T,Device>` for A, B, C_max, **C_ref_max** (×4).
- Pre-allocate 4096×4096 `Matrix<T,Host>` for H2D source + D2H sink.
- Fill host A, B with deterministic pattern; `copy_h2d` once.
- Per `N`: build sub-views via `block(0,0,N,N)` (ld=4096), feed to kernels.

### Rationale
- **Memory**: 4 × 4096² × 2 bytes (bf16) = 128 MB device + 96 MB host.
  Trivial on RTX 5070 (12 GB) and H100 (80 GB).
- **Four separate allocations, not aliased**: A, B, C_max, C_ref_max are
  distinct `cudaMalloc` calls. CUDA guarantees live allocations have
  disjoint address ranges — this is a runtime contract, not best-effort.
  No aliasing risk between `C` (custom kernel output) and `C_ref` (cuBLAS
  reference). See §5.1 for the OOB-write risk and its mitigation.
- **Alloc cost**: paid once at startup, not per-`N`. Removes allocation noise
  from timing.
- **`ld`-awareness is non-negotiable**: the submatrix has `ld=4096, N=32`.
  A kernel assuming `ld==N` would silently read/write out-of-bounds memory.
  The `MatrixView` API enforces `ld` propagation; kernels that ignore it will
  fail accuracy checks immediately (garbage output), not silently.
- **Same view feeds cuBLAS and custom kernels**: cuBLAS natively accepts
  `lda/b/c`, so no special path. Apples-to-apples comparison.

### 5.1 C_ref storage: device global memory (VRAM), not host RAM
- `C_ref_max` lives in **device global memory** for the full sweep lifetime.
- Per `N`: cuBLAS writes `C_ref_max.block(0,0,N,N)` once; this block is
  **read-only** for the rest of that `N`'s iteration. Custom kernels write
  to `C_max`, never `C_ref_max`.
- **OOB-write risk**: a buggy custom kernel that ignores `ld` could write
  past `N` rows and corrupt `C_ref_max` (or `A_max`/`B_max`). CUDA will not
  error — silent corruption. Mitigation: in **Debug builds only**, snapshot
  `C_ref_max`'s N×N block before each custom kernel and verify unchanged
  after. Cheap (one D2H + memcmp), debug-only, catches the bug class that
  would otherwise make accuracy failures unattributable.
- **Why device, not host**: avoids a D2H copy of `C_ref` per kernel-under-test.
  Only one D2H of `C_ref` per `N` (for accuracy comparison on host). The
  device copy is the source of truth; host copy is for the comparator only.

### Iteration policy
- Warmup: **20 launches**, untimed. Covers GPU boost-clock stabilization
  (5–10 launches for small kernels) and thermal steady state for large
  kernels (~15–20). Below 10 risks measuring a cold-clock kernel.
- Timed: **50 launches**. Report **min** (most robust vs OS noise) and
  **median** (characterizes typical case). **Never mean** — long-tail OS
  preemption skews it.
- **Why 20/50, not higher**: `min` of 50 gives P(min ≤ true_min × 1.01) ≈
  0.995; `median` SE ≈ 1.253σ/√50 ≈ 0.18σ. Past this, diminishing returns
  at proportionally longer sweep time. At 100/1000, large-N alone adds
  ~280 s with no statistical gain. See §11 for the full tradeoff table.
- H2D / kernel / D2H timed separately via `CudaTimer`, reported as separate
  CSV columns. Headline = `kernel_ns`. H2D/D2H reported for context and to
  validate the §2 memcpy decision.

### CSV schema
```
arch,dtype,N,kernel_name,kernel_desc,h2d_ns,kernel_min_ns,kernel_median_ns,d2h_ns,ref_kernel_min_ns,ref_kernel_median_ns,max_abs_err,max_rel_err
```
- `kernel_desc` carries design info (see §3) → plotter labels lines with it.
- `ref_kernel_min_ns` / `ref_kernel_median_ns` are cuBLAS time at the same
  `N` — lets `plot.py` draw the goal line per-kernel without a separate
  cuBLAS row.
- **Phase 1.5 fix (R15):** `h2d_ns` is a global measurement (H2D of A+B
  once at sweep start), not per-kernel. The Phase 1 implementation wrote
  `0.0` in every row, which read as "H2D took 0 ns" — misleading. R15
  repeats the global value in every row so the column is consistent and
  pandas-friendly. A future schema revision may emit it once in a
  metadata header line instead.

---

## 5.5. cuBLAS handle: stateful context, per-call stream binding

### Decision
- `CublasHandle` is RAII-owned by `Profiler<T>` (one handle per Profiler,
  not a process singleton). Created in ctor, destroyed in dtor.
- **One-time setup in ctor**:
  - `cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)` — no TF32 for bf16 path.
  - `cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST)` — alpha/beta on host.
- **Per-call** (in `cublas_gemm` wrapper):
  - `cublasSetStream(h, stream)` — bound per invocation, not on the handle.
  - `cublasGemmEx(...)` with `alpha=1.0f`, `beta=0.0f` passed by address.

### Rationale
- The handle is a **stateful context** (workspace, math mode, pointer mode),
  not a connection or a stream. Setup once; reuse across all calls.
- **Per-call stream binding** (not handle-bound): lets Phase 2 async
  pipelining bind different streams to the same handle without recreating
  it. Phase 1 uses `stream = nullptr` (legacy default stream) — simplest,
  correct, no overlap.
- **Not a singleton**: AGENTS.md doesn't mandate it; per-Profiler ownership
  is cleaner for future multi-dtype (each `Profiler<T>` owns its handle)
  and avoids global state.
- **Not thread-safe**: cuBLAS handles are not thread-safe. Bench is
  single-threaded; one handle per Profiler is sufficient. Documented in
  header. If multi-threaded bench is ever needed, one handle per thread.
- **`computeType = CUDA_R_32F`** for bf16/fp16 inputs: fp32 accumulation,
  cuBLAS default, matches our kernels' accumulation — natural ground
  truth. For fp32 inputs (Phase 2+), same `CUDA_R_32F` (pedantic, CUDA cores)
  or `CUDA_R_32F` with `CUBLAS_TF32_CUBLAS_MATH` (tensor cores, tf32) —
  selected per-kernel via a separate handle setup, not Phase 1 concern.

---

## 6. Accuracy tolerance

### Decision
- Per-dtype compile-time constant: `template <typename T> constexpr
  double kRelErrTol<T>()` in `src/bench/Accuracy.h`.
- **Failed kernels are skipped at the Profiler level**: if
  `err.max_rel > kRelErrTol<T>`, the row is not written to the CSV and a
  stderr FAIL message is printed (N, kernel name, rel_err, tol). Timing
  of mathematically invalid kernels is meaningless — storing it would
  pollute the dashboard. The cuBLAS reference row is always written
  (ground truth, err == 0).

| Dtype | Storage | Compute | Hardware | Tolerance |
|-------|---------|---------|----------|-----------|
| `bf16`   | `__nv_bfloat16` | fp32 accum, tensor cores | TC | 1e-2 |
| `fp16`   | `__half`        | fp32 accum, tensor cores | TC | 1e-3 |
| `tfloat` | `float`         | tf32, tensor cores       | TC | 1e-3 |

### Rationale
- **bf16** (1e-2): bf16's 8-bit mantissa is the loosest of the three.
  Conservative — covers cuBLAS non-determinism across runs/streams
  without false negatives, while still catching real bugs (a broken
  kernel typically produces 1e-1 or worse).
- **fp16** (1e-3): tighter mantissa (10 bits) than bf16; cuBLAS fp16
  typically agrees with fp32 reference to ~1e-4.
- **tfloat** (1e-3): tf32 truncates fp32 mantissa to 10 bits (same as
  fp16) but keeps fp32 range; tensor-core reduction order adds noise.
  1e-3 is conservative, matches fp16.
- **No fp32 pedantic tolerance**: the pedantic CUDA-core path is dropped
  entirely (see §9). Only the tf32 path exists for 32-bit float storage.
- **Skip-on-fail policy**: tolerance is a property of the C-matrix check,
  not a per-row CSV column. The CSV has no `tol` or `pass` column — every
  row in the CSV passed by construction. The tolerance value is recorded
  in the `.meta` sidecar (per-run) for documentation.

---

## 7. Timing: `CudaTimer` (device) vs `Tracer` (host)

### Decision
- `Tracer.h` (`steady_clock`): host-side orchestration only. Measures wall
  time including launch overhead. **Never** used for kernel timing.
- `CudaTimer.h` (`cudaEvent_t` pair): device-side timing for H2D, kernel,
  D2H. RAII, move-only, ~20 lines.

### Rationale
- Host `steady_clock` includes kernel launch latency (~5 µs) and any OS
  scheduling jitter. For a 32×32 bf16 kernel that runs in ~1 µs, this is
  5× noise — useless.
- `cudaEvent` records are inserted into the stream and timestamped by the
  GPU. They measure exactly the device work between record points, excluding
  launch overhead. This is the only correct way to time device work.
- `Tracer` is preserved for host orchestration (e.g. total sweep wall time,
  CSV write time) where launch overhead is irrelevant.

---

## 8. Arch-specific code: separate `.cu` files, no `#ifdef`

### Decision
- `src/sm90/*.cu` and `src/sm120/*.cu` are separate files.
- CMake compiles only the directory matching `GEMM_Y_CUDA_ARCH`.
- No `#ifdef CUDA_ARCH_SM_*` branches in kernel code.

### Rationale
- AGENTS.md mandates one binary per arch, build-time selected.
- `#ifdef` branches in a single file would (a) bloat the file, (b) make it
  impossible to compile-check the other arch locally, (c) encourage
  copy-paste divergence.
- Separate files let each arch evolve independently. The `NaiveGemm` kernel
  is identical across arches in Phase 1 (so it's duplicated) — this is
  acceptable; divergence begins with tensor-core kernels in Phase 2.

---

## 9. cuBLAS API choice: `cublasGemmEx` first

### Decision
- Phase 1 reference: `cublasGemmEx` (the older, simpler API).
- Phase 3+ goal post: `cublasLtMatmul` (the newer, lower-level API).

### Rationale
- AGENTS.md explicitly sets this progression. `cublasGemmEx` is sufficient
  for a baseline reference; `cublasLtMatmul` exposes more knobs (tile
  selection, autotuning, epilogue fusion) that only matter once we're
  competing at the cutting edge.
- Switching later is a localized change to `src/cublas/cublas_gemm.h` —
  the `Profiler` and kernels are agnostic to which cuBLAS API is used.

### Phase 2A — tf32 path via `CublasTypeMap<T>::math_mode` + `CublasMathModeGuard`

**Decision (revised 2026-07-19):** the pedantic fp32 / CUDA-core path is
**dropped entirely**. Only the tf32 path exists for 32-bit float storage.
`tfloat = float` is a dtype alias (in `src/dtypes.h`), always commented:
`// tfloat = tf32 path (TC), not pedantic fp32 (CUDA cores).`

**Mechanism:**
- `CublasTypeMap<T>` gains a `static constexpr cublasMath_t math_mode`
  field per specialization:
  - `bf16`   → `CUBLAS_DEFAULT_MATH`
  - `fp16`   → `CUBLAS_DEFAULT_MATH`
  - `tfloat` → `CUBLAS_TF32_TENSOR_OP_MATH`
  (Note: the enum constant in CUDA's `cublas_api.h` is
  `CUBLAS_TF32_TENSOR_OP_MATH` (value 3); there is no
  `CUBLAS_TF32_CUBLAS_MATH`.)
- `cublas_gemm` wraps the `cublasGemmEx` call in a `CublasMathModeGuard`:
  ```cpp
  CublasMathModeGuard guard(handle.get(), TM::math_mode);
  CUBLAS_CHECK(cublasGemmEx(...));
  ```
  For bf16/fp16 this is a no-op (sets DEFAULT_MATH, restores DEFAULT_MATH).
  For tfloat it sets TF32_CUBLAS_MATH and restores DEFAULT_MATH.
- **No distinct `cublas_gemm_tf32` entry point** — since there's no
  pedantic path to distinguish from, `cublas_gemm<float>` *is* the tf32
  path. The math mode is selected by `CublasTypeMap<T>::math_mode`.

**`CublasMathModeGuard`** (free class in `src/cublas/CublasHandle.h`):
- RAII: ctor captures prev mode via `cublasGetMathMode`, sets new mode;
  dtor restores prev mode. Non-copyable, non-movable.
- Uses `cublasGetMathMode` (not a hardcoded DEFAULT_MATH restore) for
  robustness — if the handle's default mode ever changes, the guard
  still restores correctly. Overhead is negligible vs the GEMM call.
- **Non-thread-safe toggle** (already documented for `CublasHandle` in
  §5.5): the math mode is handle state, so concurrent calls on the same
  handle with different modes would race. Bench is single-threaded; no
  issue. If multi-threaded bench ever lands, one handle per thread.

**Profiler integration:** `Profiler<float>` (referred to as
`Profiler<tfloat>` at call sites) registers only the tf32 cuBLAS
reference. There is no pedantic-vs-tf32 split — one reference row per N
in the tf32 CSV. Custom tfloat kernels (Phase 2C) compare against this
reference.

---

## 10. Phase 1 validation

### Exit criteria
- `gemm_y` produces a well-formed CSV with cuBLAS + naive kernel rows across
  the full size sweep, all `max_rel_err ≤ 1e-2`.
- `test_cuda` covers Buffer / MatrixView / CudaTimer / CublasHandle unit tests.
- Memcpy microbench results recorded in §2.
- Cross-arch sanity: CSV produces on both sm_120 (RTX 5070) and sm_90 (H100).

### Results
```
# Run: ./build/gemm_y  (RTX 5070, sm_120, CUDA 13.2, Release)
# CSV: results/bench_sm_120_bf16.csv  (29 rows = 1 header + 14 sizes x 2 kernels)
#
# Accuracy: max_rel_err = 0.0e+00 at every N for the naive kernel vs cuBLAS.
#   The deterministic fill pattern (small ints in [-3, 4]) fits exactly in
#   bf16 and accumulates in fp32 without rounding, so both implementations
#   produce bit-identical output. This validates the harness end-to-end
#   (H2D, kernel launch, D2H, accuracy compare, CSV write) but does NOT
#   exercise the tolerance — Phase 2 will use a pattern that produces
#   non-zero reduction-order disagreement.
#
# Performance (kernel_median_ns, naive vs cuBLAS):
#   N=   32:  naive 2.3 us   | cuBLAS 5.0 us   (naive wins — launch-dominated)
#   N=  128:  naive 6.0 us   | cuBLAS 5.1 us   (parity)
#   N=  512:  naive 137 us   | cuBLAS 13 us    (cuBLAS 10x faster)
#   N= 1024:  naive 1040 us  | cuBLAS 41 us    (cuBLAS 25x faster)
#   N= 4096:  naive 77293 us | cuBLAS 1983 us  (cuBLAS 39x faster)
# As expected: the naive triple-loop is competitive at tiny N (launch-
# dominated regime where cuBLAS's overhead dominates) and falls off a cliff
# past N=256. This is the baseline Phase 2 tiled/wmma kernels must beat.
#
# Sweep wall time: ~6 s (matches ARD §11 budget estimate).
# Debug-build OOB snapshot checks: passed (no C_ref corruption detected).
#
# Cross-arch (sm_90 / H100): not run in this session — no H100 available
# locally. The sm_90 kernel file (src/sm90/gemm_bf16_naive.cu) is identical
# to sm_120 in Phase 1; build with -DGEMM_Y_CUDA_ARCH=sm_90 and run on H100
# to complete the cross-arch sanity exit criterion.
```

---

## 11. Iteration policy: warmup / timed counts

### Decision
- **Warmup: 20 launches** (untimed).
- **Timed: 50 launches** — report **min** and **median**, never mean.
- Uniform across all `N` for simplicity.

### Tradeoff analysis

| Warmup / Timed | Pro | Con |
|----------------|-----|-----|
| 5 / 10  | Fast sweep | min of 10 noisy; median CI wide |
| 20 / 50 | Stable min; tight median CI | ~5× longer sweep |
| 50 / 100| Marginally better stats | Diminishing returns; large-N slow |
| 100 / 1000 | Best stats | Large-N sweep takes minutes; no real gain |

### Why 20 warmup
- GPU boost clock stabilizes in 5–10 launches for small kernels.
- Large kernels may trigger thermal throttle; need ~15–20 to reach steady state.
- 20 covers both regimes with margin. Below 10 risks measuring a cold-clock kernel.

### Why 50 timed
- `min` of 50: P(min ≤ true_min × 1.01) ≈ 1 − 0.99^50 ≈ 0.995. Robust.
- `median` of 50: standard error ≈ 1.253σ/√50 ≈ 0.18σ. Tight.
- Below 30: min unstable (one outlier dominates). Above 100: marginal gain.

### Sweep time budget (20/50, bf16, 14 sizes, 2 kernels)
- Small N (32–256): kernel ~1–50 µs. 50 iters ≈ 50 µs–2.5 ms. Negligible.
- Mid N (512–1024): kernel ~50–500 µs. 50 iters ≈ 2.5–25 ms. Fine.
- Large N (2048–4096): kernel ~1–10 ms. 50 iters ≈ 50–500 ms per kernel per N.
- Total: ~14 sizes × 2 kernels × ~150 ms avg ≈ 4 s. Plus cuBLAS reference
  (14 × ~150 ms ≈ 2 s). **~6 s per dtype per arch.** Acceptable.

### Why not higher
- At 100/1000: large-N alone = 14 × 2 × 10 s = 280 s. No statistical gain
  over 20/50. The bottleneck for measurement quality is GPU boost-clock
  stability and `cudaEvent` resolution (~1 µs), not sample count.

### Future refinement (not Phase 1)
- If tiny-kernel noise becomes visible in Phase 2 (min bouncing between runs),
  consider **time-bounded mode** for small N: run for ≥10 ms total, count
  iters. Better signal-to-noise for sub-µs kernels than fixed count. Not
  needed in Phase 1 — naive kernel is slow enough that fixed-50 is fine.

---

## 12. Phase 2 plan

### 12.1 Three dtypes (priority order)
Phase 2 organizes work around three storage dtypes. The pedantic fp32 /
CUDA-core path is dropped entirely (see §9) — only the tf32 path exists
for 32-bit float storage.

1. **bf16 (primary)**: tensor cores, fp32 accumulation. Phase 1 baseline.
2. **fp16 (primary sibling)**: tensor cores, fp32 accumulation. Should
   show **no meaningful perf difference** vs bf16 on Hopper/Blackwell
   tensor cores (same MMA throughput, same memory layout width).
   Strategy: optimize one (bf16, as the Phase 1 baseline), then replicate
   the final kernel's ideas to fp16 with minimal tuning.
3. **tfloat (secondary)**: `tfloat = float` alias. tf32 compute (TC),
   not pedantic fp32 (CUDA cores). Relevant for fp32-input workloads
   that can tolerate tf32's mantissa truncation. Same tensor-core MMA
   as bf16/fp16, different dtype config.

### 12.2 Phase 2A — cuBLAS references for bf16 / fp16 / tf32
- `dtypes.h`: add `tfloat = float` alias; `name<float>()` returns
  `"tf32"`. Drop the fp32 pedantic name.
- `CublasTypeMap<T>` gains `math_mode` field; `cublas_gemm` wraps the
  call in `CublasMathModeGuard` (see §9). No distinct
  `cublas_gemm_tf32` entry point.
- `kRelErrTol<T>` per-dtype constant; failed kernels skipped at the
  Profiler level (see §6).
- `main.cpp` runs three sequential sweeps (bf16, fp16, tfloat). Only
  `NaiveGemm<bf16>` registered for 2A; fp16/tfloat sweeps are
  cuBLAS-only. `NaiveGemm<fp16>` / `NaiveGemm<tfloat>` deferred to 2C.
- `main.cpp` writes a `.meta` sidecar (key=value) alongside each CSV for
  the Python ingest layer.
- Unit tests for fp16/tfloat cuBLAS paths; tfloat test verifies
  math-mode restore.

### 12.3 Phase 2B — Visualization (Python + Plotly + Dash + SQLite)
- Stack: Plotly (interactive hover), Dash (reactive checkboxes/toggles
  + built-in Flask server), SQLite (`sqlite3` stdlib, queryable,
  git-trackable). No matplotlib.
- `db/gemm_y.db` tracked in git, declared binary in `.gitattributes`.
  CSVs in `results/` stay gitignored (regenerable).
- `scripts/ingest.py` reads CSV + `.meta` sidecar, appends to SQLite.
  Captures `git_sha` at ingest time.
- `scripts/server.py` — Dash app at `localhost:8050`. Single page,
  sidebar + three tabs (Timing / Accuracy / Run History). Sidebar
  filters: arch, dtype, custom-vs-cuBLAS, runs (multi-select), scale
  (log-log / linear). Hover shows arch, dtype, custom/ref, kernel name,
  kernel desc, N, median_ns, ref_median_ns, speedup vs cuBLAS.
- `scripts/dump_db.py` — optional JSONL export for human inspection.
- `pyenv/` venv (Python 3.14) for all scripts.

### 12.4 Phase 2C — First tiled bf16 kernel (after 2A + 2B)
- `src/sm120/gemm_bf16_tiled_128.cu` (+ `.cuh`) — 128×128 tile, 8 warps/CTA,
  `wmma`/`mma.sync` for bf16 on sm_120.
- `src/sm90/gemm_bf16_tiled_128.cu` (+ `.cuh`) — same algorithm, sm_90
  `wmma` API.
- `NaiveGemm<fp16>` and `NaiveGemm<tfloat>` land here (sm90 + sm120) so
  the fp16/tfloat sweeps have a custom kernel to compare against cuBLAS.
  Register in `main.cpp` alongside `NaiveGemm<bf16>`.
- Run sweep, ingest to DB, view in dashboard, compare vs cuBLAS line.
  Iterate per AGENTS.md experiment discipline (one variable per commit).
- Once bf16 tiled kernel is competitive, replicate ideas to fp16 (Path 1
  sibling). Expect near-identical tensor-core perf.

---

## 13. Phase 1.5 refactor inventory

One-line rationale per refactor item. Serves as audit trail for why each
cleanup was made. Items map to `TODO.md` Phase 1.5 R1–R20.

| ID | File / area | Change | Why |
|----|-------------|--------|-----|
| R1  | `Copy.h` | Revert to sync `cudaMemcpy`/`cudaMemcpy2D`; drop stream param | Async-on-pageable is silently sync + staging overhead (§2 correction) |
| R2  | `Copy.h` | Extract `detail::plan_copy` + `copy_contiguous`/`copy_strided` | ~50 lines duplicated between `copy_h2d`/`copy_d2h` |
| R3  | `Profiler` | `run_sweep` returns `SweepResult`; CSV writing moves to `main.cpp` | Decouple bench logic from I/O; make `run_sweep` testable |
| R4  | `Profiler` | Measure cuBLAS once per N, reuse for all kernels | Phase 1 re-timed cuBLAS per kernel (K×S×kTimed extra launches) |
| R5  | `cuda_compat.h` | Include `<cublas_v2.h>` under pragma; remove direct includes | AGENTS.md: all CUDA includes via the single wrapper |
| R6  | `Buffer.h` | Fix misleading 64-byte alignment comment | `std::vector` default allocator gives ~16 bytes, not 64 |
| R7  | `test.cu` | Remove `test_smoke`; trim `test_buffer_device` | `test_smoke` is RAII-violating and redundant; round-trip covered by Copy tests |
| R8  | `test.cu` | Strengthen `test_cuda_timer`; add 5 new tests | Cover const-conversion, strided cuBLAS, naive kernel, small profiler sweep |
| R9  | `src/bench/microbench/` | Move microbenches to subdir; cleaner CMake glob | Replace fragile `list(FILTER ... REGEX)` with directory-based selection |
| R10 | `print_table.h` | Aligned fixed-width table output for microbenches | Raw CSV-to-stdout is hard to scan; one-off microbenches don't need CSV |
| R11 | `src/Arch.h` | Single `kArchName` definition | Duplicated in `main.cpp`, `test.cu`, `Profiler.cu` |
| R12 | `src/bench/Stats.h` | Extract `TimedStats` + `summarize_ns` | Duplicated in `Profiler.cu` and `memcpy_microbench.cu` |
| R13 | `src/bench/Fill.h` | Extract `fill_sequential(A, B)` | Duplicated in `Profiler.cu` and `test.cu` |
| R14 | `src/dtypes.h` | `dtypes::name<T>()`; co-locate dtype aliases | `dtype_name<T>()` in `Profiler.cu` is duplicated knowledge; inconsistent with `string_view` convention |
| R15 | `Profiler.cu` | `h2d_ns` column repeats global value, not `0.0` | Current `0.0` reads as "H2D took 0 ns" — misleading |
| R16 | `Profiler.cu` | `Timer<>` default capacity (drop `<4096>`) | Only 3 marks used; 4096 is confusing |
| R17 | `CudaTimer.h` | `elapsed_ms()` marked `const` | No state mutation; const-correctness per AGENTS.md |
| R18 | `MatrixView.h` | `block()` debug-only bounds asserts | Silent OOB if misused; cheap debug-only check |
| R19 | `cublas_gemm.h` | Extract `GEMM_Y_ASSERT`; layout check → debug assert | 13 lines of `fprintf`+`abort` is noise; Phase 1 invariant, not runtime API contract |
| R20 | `CudaCheck.h` | Rename macro local vars `_gemm_y_err` → `gemm_y_err_` | Suffix-underscore avoids reserved-identifier edge cases |

---

## 14. Phase 1.5 validation

### Exit criteria
- All R1–R20 refactor items landed; `test_cuda` passes with 874 checks.
- `Copy.h` reverted to sync; `Profiler::run_sweep` returns `SweepResult`;
  cuBLAS measured once per N.
- Microbenches relocated to `src/bench/microbench/` with human-readable
  table output.
- Dedup extractions (`Arch.h`, `Stats.h`, `Fill.h`, `dtypes.h`) in place.

### Results
```
# Phase 1.5 refactor complete. No new features; correctness + hygiene only.
# Test suite: 874 checks, 0 failures (was ~30 checks pre-Phase 1.5).
# Copy.h: ~50 lines of duplication eliminated via detail::plan_copy.
# Profiler: cuBLAS re-timing removed; sweep ~K× faster on K-kernel runs.
# Microbench output: human-readable tables replace raw CSV-to-stdout.
#
# Phase history: git log --grep="Phase: 1.5"
```
