# TODO.md — Containerized remote development

## Remaining container follow-up

The local Docker parity image and initial GPU-backed verification are complete.
Docker is used only as a sanity check for matching the remote RunPod software
environment; normal local builds and tests may continue to run directly on the
host.

- [ ] Record GPU, compute capability, driver, CUDA toolkit, compiler, CMake,
      and Docker versions in the project documentation or validation notes.
- [ ] Document the required NVIDIA Container Toolkit/runtime setup and
      `--gpus all` usage.

## Manual remote workflow

- [ ] Create a minimal source/archive payload containing the required Docker
      and build context:

  ```text
  CMakeLists.txt
  Dockerfile
  .dockerignore
  src/
  tests/
  ```

- [ ] Exclude `.git/`, `build/`, `results/`, `db/`, `artifacts/`, `pyenv/`,
      `refs/`, generated CMake files, and unrelated local environments.
- [ ] Copy the archive to the remote machine with `scp`.
- [ ] SSH into the remote machine manually.
- [ ] Build or run the container remotely.
- [ ] Run `kernel_smoke` first and stop on correctness failure.
- [ ] Run benchmarks only after smoke validation passes.
- [ ] Copy logs and benchmark results back with `scp`.
- [ ] Keep database ingestion manual and local.
- [ ] Document the remote workspace layout and manual cleanup commands.
- [ ] Leave pod lifecycle management to the user; do not add RunPod API
      provisioning, termination, or credential handling.

## Optional shell helper

- [ ] Validate the complete workflow manually before adding automation.
- [ ] Identify repetitive, stable commands after the manual workflow works.
- [ ] Add a simple shell helper for tar/scp/ssh orchestration only if it
      provides clear value.
- [ ] Keep artifact retrieval explicit and inspectable.
- [ ] Preserve nonzero exit statuses from configure, build, smoke, and retrieval
      operations.
- [ ] Do not recreate the removed Python harness with retry, manifest, database,
      or lifecycle-management infrastructure.

## Resume kernel work

- [ ] Reconfirm the primary target architecture after the container workflow is
      validated.
- [ ] Continue BF16 Hopper optimization on `sm_90` according to the kernel
      contract and experiment workflow.
- [ ] Run `kernel_smoke` before benchmark comparisons.
- [ ] Use the clock-locked benchmark wrapper for reproducible performance runs.
- [ ] Revisit `tcgen05` only through a new architecture decision on confirmed
      compatible hardware.
- [ ] Reintroduce reference code only when the `tcgen05` effort is explicitly
      restarted and its indexing, API, build, and correctness contracts are
      defined again.
