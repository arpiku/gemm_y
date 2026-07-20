# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

---

## Phase 2C.0 — Restructure arch kernel files (no logic change)

Goal: rename the dtype-agnostic naive kernel away from its `bf16` misnomer
and introduce a single shared `.cuh` per arch that declares all
bf16-specific custom kernels. No kernel logic changes; no perf change.
Sets the file organization for Phase 2C.1+.

**Rationale:** `NaiveGemm<T>` is dtype-agnostic (casts to fp32, accumulates,
casts back) and is instantiated for bf16 / fp16 / tfloat from one template
definition — the `gemm_bf16_naive` filename was always wrong. Tiled TC
kernels are dtype-specific by nature (wmma fragment types differ per
dtype), so they belong in a separate `gemm_bf16.cuh` that declares only
bf16-specific structs. See ARD §16.

- [ ] **2C.0.1** Rename `src/sm120/gemm_bf16_naive.{cuh,cu}` →
  `src/sm120/gemm_naive.{cuh,cu}`. Mirror in `src/sm90/`.
- [ ] **2C.0.2** Create `src/sm120/gemm_bf16.cuh` — shared declaration
  header for all bf16-specific custom kernels on sm_120. Empty of kernel
  structs for now (populated in 2C.1). Mirror `src/sm90/gemm_bf16.cuh`.
- [ ] **2C.0.3** Update `#include` in `src/main.cpp` and `tests/test.cu`:
  `sm{90,120}/gemm_bf16_naive.cuh` → `sm{90,120}/gemm_naive.cuh`.
  Add `#include "sm{90,120}/gemm_bf16.cuh"` to `main.cpp` (forward decl
  for the 2C.1 struct; harmless until referenced).
- [ ] **2C.0.4** Build + ctest: `cmake --build build -j && ctest --test-dir
  build`. Expect 879/879 (no test changes — `NaiveGemm<T>` symbol is
  unchanged, only the file name moved).

---

## Phase 2C.1 — Dummy `k0` kernel (owner: user)

Goal: land a placeholder custom kernel that is a verbatim copy of the
naive kernel body, registered alongside `NaiveGemm<bf16>` in `main.cpp`.
**No optimization.** Purpose: develop the UI / workflow / measurement
loop against real data before investing in tiled TC kernels.

**Naming convention (see ARD §16):** custom kernels are named `k0`, `k1`,
`k2`, … during development — the counter tracks iteration progress
(`k0` ≈ naive-level perf, `k9` faster than `k1`). Switch to descriptive
names (`Tiled128`, `Tiled128V2`, …) only when a kernel is finalized /
competitive. `NaiveGemm<T>` stays as the permanent sanity baseline.

**File organization (see ARD §16):** one shared `gemm_bf16.cuh` declares
all bf16-specific kernel structs; each kernel's `operator()` definition
lives in its own `gemm_bf16_k<n>.cu` file (compile isolation — a one-line
edit to `k5` should not recompile `k0`–`k4`). CMake's
`file(GLOB _gemm_y_arch_sources CONFIGURE_DEPENDS "${arch_dir}/*.cu")`
auto-picks new `.cu` files; zero CMake changes per kernel.

- [ ] **2C.1.1** Declare `k0` in `src/sm120/gemm_bf16.cuh`:
  ```cpp
  // k0 — dummy custom kernel (verbatim naive body). Workflow development
  // only; no optimization. Perf ≈ NaiveGemm. Rename to descriptive name
  // when a real strategy lands (see ARD §16).
  struct k0 {
      static constexpr std::string_view name()        { return "k0"; }
      static constexpr std::string_view description() {
          return "k0: dummy = naive kernel body (workflow development)";
      }
      void operator()(GemmArgs<__nv_bfloat16> args,
                      cudaStream_t stream) const;
  };
  ```
  bf16-only (no template parameter) — tiled/dummy kernels are
  dtype-specific by nature. Mirror declaration in `src/sm90/gemm_bf16.cuh`.
- [ ] **2C.1.2** Define `k0::operator()` in `src/sm120/gemm_bf16_k0.cu` —
  verbatim copy of `NaiveGemm<__nv_bfloat16>::operator()` body (same
  `naive_gemm_kernel` device function, same launch config). Mirror in
  `src/sm90/gemm_bf16_k0.cu`.
- [ ] **2C.1.3** Register in `src/main.cpp` bf16 block, alongside
  `NaiveGemm<bf16>`:
  ```cpp
  prof.register_kernel<gemm_y::NaiveGemm<T>>();
  prof.register_kernel<gemm_y::k0>();
  ```
  fp16 / tfloat blocks unchanged (k0 is bf16-only for now). Update
  `write_meta` `kernels` list to include `k0`.
- [ ] **2C.1.4** Build + ctest: expect 879/879 (no test changes — k0 is
  registered in `main.cpp` only, not exercised by `test.cu`).
- [ ] **2C.1.5** Run `./build/gemm_y` → bf16 CSV now has 42 rows (14 sizes
  × 3 kernels: cuBLAS + NaiveGemm + k0). All PASS (k0 is a verbatim naive
  copy, so accuracy is identical to NaiveGemm). Ingest to DB with
  `--label "k0-dummy"`.

---

## Phase 2B.2 — Dashboard UX extensions + manual UI review

Goal: extend the dashboard with the `% perf vs cuBLAS` comparison metric
and a run-management CLI, then do the manual UI review against the k0
data. Triggered after 2C.1.5 (k0 in the DB).

**`% perf vs cuBLAS` definition (see ARD §15):**
```
perf_pct = (cublas_median_ns - custom_median_ns) / cublas_median_ns * 100
```
- `+X%` → custom is X% faster (good). `-X%` → custom is X% slower (bad).
- `0%` → parity.
- Label convention: `% vs cuBLAS (+ = faster)` — the sign is non-obvious
  and must be explicit everywhere it appears (hover, axis title, run
  history column header).
- Always display alongside absolute time — the percentage compresses
  large-N regressions and amplifies small-N noise (cuBLAS at N=32 is
  launch-overhead dominated; a 5us→50us regression is `-900%`).

### Tasks

- [ ] **2B.2.1** `scripts/db.py`: add a query helper
  `measurements_with_perf_pct(conn, run_id)` that joins each custom
  measurement to its cuBLAS sibling (same `run_id`, same `N`,
  `kernel_name == 'cublas'`) and computes `perf_pct`. Returns rows with
  the existing columns + `perf_pct`. cuBLAS rows themselves get
  `perf_pct = 0` (or NULL — TBD at implementation).
- [ ] **2B.2.2** `scripts/server.py` timing hover: add `perf_pct` to the
  `customdata` array and to the `hovertemplate`, alongside the existing
  speedup ratio. Format: `perf=%{customdata[5]:+.1f}% vs cuBLAS`.
- [ ] **2B.2.3** `scripts/server.py` new comparison view: a fourth tab
  (or a toggle in the Timing tab — TBD) plotting `perf_pct` vs `N`,
  horizontal line at 0 (parity). Lines above 0 = beating cuBLAS. Default
  linear y-axis (the percentage is the point; log scale obscures it).
  Apply the same sidebar filters as the timing tab.
- [ ] **2B.2.4** `scripts/server.py` Run History tab: add a column
  `median % vs cuBLAS @ N=4096` (largest common sweep size) as a
  single-number summary per run. Use the `perf_pct` of the run's
  best custom kernel at N=4096 (or the only custom kernel, for now).
- [ ] **2B.2.5** `scripts/delete_run.py` (new): CLI to manage runs in the
  DB. Subcommands:
  - `python scripts/delete_run.py list` — print all runs (id, ingested_at,
    git_sha, label, arch, dtype, measurement count). Same columns as the
    Run History tab, plus the count.
  - `python scripts/delete_run.py delete <id> [<id> ...]` — delete the
    named run(s) and their measurements (cascading delete via
    `FOREIGN KEY ... REFERENCES runs(id)`). Confirm prompt before delete
    unless `--force`.
  - Refuse to delete if the run is the only one for its `(arch, dtype)`
    pair unless `--force` (guard against wiping the last baseline).
  - Print the deleted row count + remaining run count on success.
- [ ] **2B.2.6** `AGENTS.md` Python tooling section: add
  `python scripts/delete_run.py list|delete <id>...` to the workflow
  block.

### Manual UI review (after 2B.2.1–2B.2.5 land)

- [ ] **2B.2.7** Run the full pipeline and review in the browser:
  ```sh
  ./build/gemm_y
  source pyenv/bin/activate
  python scripts/ingest.py results/bench_sm_120_bf16.csv --label "k0-dummy" --force
  python scripts/ingest.py results/bench_sm_120_fp16.csv --label "..." --force
  python scripts/ingest.py results/bench_sm_120_tf32.csv --label "..." --force
  python scripts/delete_run.py list
  python scripts/server.py
  # → open http://localhost:8050
  ```
  Verify:
  - Timing tab: bf16 shows 3 custom lines (NaiveGemm + k0) + 1 cuBLAS;
    fp16 / tf32 show 1 custom + 1 cuBLAS each.
  - Hover shows `perf=...% vs cuBLAS` alongside the existing speedup ratio
    and absolute ns.
  - New comparison view: `perf_pct` vs `N`, parity line at 0, k0 line
    hugs 0 (it's a naive copy — should match NaiveGemm's perf_pct).
  - Run History tab: new `median % vs cuBLAS @ N=4096` column populated.
  - `delete_run.py list` output matches the Run History tab.
  - log/linear toggle, accuracy tab tol lines, all prior 2B.1 behavior
    still works.
- [ ] **2B.2.8** Collect UI / workflow change suggestions from the user
  after the review. These feed into a future 2B.3 (TBD scope) — better
  comparison UX, additional metrics, etc. Not scoped yet.

---

## Phase 2C.2 — Methodology / measurement strategy (owner: user)

Goal: before writing optimized (Tiled, double-buffer, …) kernels,
establish a repeatable workflow for testing and measuring them. Drives
the structure of 2C.3+.

**Owner:** user. The agent supports (dashboard extensions, CLI helpers,
documentation) but does not define the methodology.

- [ ] **2C.2.1** Define the iteration workflow: how to label runs
  (`--label "k1-tiling-128"`?), how to compare `k_n` vs `k_(n-1)` in the
  dashboard (run multi-select already exists — sufficient? or need a
  diff view?), how to record the hypothesis for each commit (commit body
  per AGENTS.md — sufficient? or need a sidecar file?).
- [ ] **2C.2.2** Decide on statistical rigor: current 20 warmup / 50
  timed median. Is median enough? Need min (best-case) or p99 (tail)?
  Need confidence intervals? If yes, extend `Profiler::run_sweep` to
  emit more stats and the CSV/DB schema to store them.
- [ ] **2C.2.3** Profiling tool setup — nsys / ncu. Currently out of
  scope (see Out of scope section). Decide whether to bring them in
  here, or defer to a later phase. If bringing in: wrapper scripts,
  report parsing, dashboard integration (TBD scope).
- [ ] **2C.2.4** Document the workflow in `ARD.md` (new section) and
  `AGENTS.md` (workflow blurb) once it's stable.

---

## Phase 2C.3+ — Real custom kernels (DEFERRED, owner: user)

Goal: tiled tensor-core kernels that beat cuBLAS at large N. Deferred
until 2C.2 (methodology) is in place — the user wants the measurement
loop solid before investing in optimization.

**Owner:** user. The agent's first tiled attempt (128×128 CTA, 8 warps,
`nvcuda::wmma` 16×16×16) failed accuracy and was reverted; the failure
analysis is in commit `73ef7fb`. The user will re-attempt from scratch
with a debugging approach driven by the 2C.2 methodology.

- [ ] **2C.3.1** First tiled TC kernel for bf16 on sm_120 (file name
  TBD — `gemm_bf16_k1.cu` during development, renamed to
  `gemm_bf16_tiled128.cu` when finalized, per ARD §16).
- [ ] **2C.3.2** Mirror on sm_90 (same algorithm, sm_90 wmma API).
- [ ] **2C.3.3** Iterate: one variable per commit (tile size, warp count,
  K-dim unroll, memory layout). Ingest each iteration, compare in
  dashboard, record hypothesis + result in commit body.
- [ ] **2C.3.4** Replicate to fp16 (Path 1 sibling) once bf16 is
  competitive. Expect near-identical perf on tensor cores.
- [ ] **2C.3.5** Replicate to tfloat (tf32 path) — same TC MMA, different
  dtype config.

---

## Phase 2 prep (deferred)

- `Space::HostPinned` + `Buffer<T, HostPinned>` via `cudaHostAlloc`.
- Bench runner host buffers → pinned.
- Async `cudaMemcpyAsync` on explicit stream (when pipelining lands).
- Debug-build assert or `static_assert` on pinned-ness at `copy_*` call
  sites when passing non-null stream (catches silent staging).

---

## Out of scope (unchanged)

- `cublasLtMatmul` (Phase 3+).
- Batched, transposed, epilogue-fused, non-square variants (AGENTS.md non-goals).
- nsys / ncu profiling integration (may enter in 2C.2 — TBD by user).
- Multi-GPU / multi-node.
- fp32 pedantic (CUDA cores) — dropped entirely; only tf32 path for
  32-bit float storage (see ARD §9).
- TFLOPS metric — deferred. Needs peak-TFLOPS lookup per `(arch, dtype)`.
  Will be added in a future sub-phase after the `% perf vs cuBLAS` metric
  is in place.
