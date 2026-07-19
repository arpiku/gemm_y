# TODO.md

> Forward-looking task list. Completed work lives in git history
> (`git log --grep="Phase: X.X"`) and ARD phase-summary sections.
> Do not carry completed items here. Durable project state in `AGENTS.md`;
> decision rationale in `ARD.md`.

## Phase 2B.1 — Manual UI review

Goal: user-side verification of the dashboard after the Phase 2B.1
bug fix + polish (callback signature, accuracy zero-clamp, dropdown
labels, hover formatting, horizontal legend, semi-transparent cuBLAS,
connection hygiene). All implementation items (2B.1.1–2B.1.9) are
done; only the manual browser review remains.

- [ ] **2B.1.10** Manual UI review. Run the full pipeline:
  ```sh
  ./build/gemm_y
  source pyenv/bin/activate
  python scripts/ingest.py results/bench_sm_120_bf16.csv --label "..." --force
  python scripts/ingest.py results/bench_sm_120_fp16.csv --label "..." --force
  python scripts/ingest.py results/bench_sm_120_tf32.csv --label "..." --force
  python scripts/server.py
  # → open http://localhost:8050
  ```
  Verify: timing tab shows 6 lines (3 cuBLAS + 3 naive), hover shows
  all 7 fields, log/linear toggle works, accuracy tab shows tol lines,
  run history tab lists all ingested runs.

---

## Phase 2C — Tiled tensor-core kernels (owner: user)

Goal: first tensor-core kernel for bf16 that beats cuBLAS at large N,
then replicate to fp16 / tf32. The naive kernel (Phase 2C partial,
already landed) gives every dtype a custom kernel to compare against
cuBLAS, but it's ~10–40× slower than cuBLAS at N≥512 — the tiled TC
kernel is the real work.

**Owner:** user. The agent's first attempt (128×128 CTA, 8 warps,
`nvcuda::wmma` 16×16×16) failed accuracy and was reverted; the failure
analysis is in commit `73ef7fb`. Re-attempt from scratch — the user
will drive the debugging approach.

- [ ] **2C.1** `src/sm120/gemm_bf16_tiled_128.cu` (+ `.cuh`) — 128×128
  tile, 8 warps/CTA, `wmma`/`mma.sync` for bf16 on sm_120.
- [ ] **2C.2** `src/sm90/gemm_bf16_tiled_128.cu` (+ `.cuh`) — same
  algorithm, sm_90 `wmma` API.
- [ ] **2C.3** Register in `main.cpp` alongside `NaiveGemm<bf16>`. Run
  sweep, ingest to DB, view in dashboard, compare vs cuBLAS line.
  Iterate (one variable per commit per AGENTS.md experiment discipline).
- [ ] **2C.4** Once bf16 tiled kernel is competitive, replicate ideas to
  fp16 (Path 1 sibling). Expect near-identical perf on tensor cores.
- [ ] **2C.5** Replicate to tfloat (tf32 path) — same TC MMA, different
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
- nsys / ncu profiling integration.
- Multi-GPU / multi-node.
- fp32 pedantic (CUDA cores) — dropped entirely; only tf32 path for
  32-bit float storage (see ARD §9).
