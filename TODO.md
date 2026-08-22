# TODO.md — Containerized remote development

## Decision

- The Python remote smoke harness has been deprecated and removed.
- The temporary `refs/` directory and its `tcgen05` reference code have been
  removed for now; revisit that work only through a new documented decision.
- First reproduce the remote RunPod environment locally with Docker using the
  target image:

  ```text
  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
  ```

- Remote validation will initially use a deliberately manual SSH/SCP workflow.
  No RunPod API lifecycle automation or Python remote orchestration is planned.
- The container should be a reusable development environment with the source
  mounted at runtime, so kernel edits do not require rebuilding the image.
- Run `kernel_smoke` before any remote benchmark. Keep benchmark interpretation
  and database ingestion manual and local.

## 1. Remove obsolete workflow

- [x] Delete `scripts/remote_smoke.py`.
- [x] Delete the temporary `refs/` directory.
- [x] Remove obsolete remote-harness instructions from `AGENTS.md`.
- [x] Mark the remote-harness decision in `ARD.md` as superseded.
- [x] Search the repository for stale `remote_smoke`, RunPod-harness, and
      removed-reference code mentions.
- [x] Confirm dashboard, ingestion, database, benchmark, source, and test code
      remain unaffected by the removals.

## 2. Verify the target container environment

- [ ] Confirm the exact base image:
      `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- [ ] Verify the image provides `nvcc`, CMake, GCC, and G++.
- [ ] Verify Docker GPU passthrough:

  ```sh
  docker run --rm --gpus all <image> nvidia-smi
  ```

- [ ] Verify the host NVIDIA driver is compatible with the container CUDA
      toolkit.
- [ ] Record GPU, compute capability, driver, CUDA toolkit, compiler, CMake,
      and Docker versions.
- [ ] Confirm the container can configure the project for the selected
      architecture.
- [ ] Confirm the container can build and run `kernel_smoke`.
- [ ] Confirm CUDA 12.8.1 supports the selected target architecture before
      relying on the container for remote `sm_120` work.

## 3. Add the reproducible Docker development environment

- [ ] Add a minimal `Dockerfile` based on the verified remote image.
- [ ] Add a `.dockerignore` that excludes generated and local-only state:
      `.git/`, `build/`, `results/`, `db/`, `artifacts/`, `pyenv/`, and
      generated CMake files.
- [ ] Keep source and build output outside the image where practical.
- [ ] Define the image build command.
- [ ] Define a runtime command that mounts the repository and runs from the
      mounted source tree.
- [ ] Document the required NVIDIA container runtime and `--gpus all` usage.
- [ ] Verify local container behavior matches the target remote environment.
- [ ] Do not place credentials, SSH keys, API tokens, or sensitive environment
      variables in the image or Docker configuration.

Recommended local validation shape:

```sh
docker build -t gemm-y:cu1281 .
docker run --rm --gpus all -v "$PWD:/workspace/gemm_y" -w /workspace/gemm_y \
    gemm-y:cu1281 \
    sh -c 'cmake -B build -DGEMM_Y_CUDA_ARCH=sm_120 && \
           cmake --build build --target kernel_smoke -j && \
           ./build/kernel_smoke'
```

## 4. Define the manual remote workflow

- [ ] Validate the complete workflow manually before adding automation.
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

## 5. Add a small shell helper only after manual validation

- [ ] Identify repetitive, stable commands after the manual workflow works.
- [ ] Add a simple shell helper for tar/scp/ssh orchestration only if it
      provides clear value.
- [ ] Keep artifact retrieval explicit and inspectable.
- [ ] Preserve nonzero exit statuses from configure, build, smoke, and retrieval
      operations.
- [ ] Do not recreate the removed Python harness with retry, manifest, database,
      or lifecycle-management infrastructure.

## 6. Resume kernel work

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
