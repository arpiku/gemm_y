# TODO.md — Remote GPU test harness

## Decision

- `tcgen05.mma` is not supported on `sm_120` RTX 50-series GPUs, including
  the local RTX 5070 and the remote RTX 5090 test system.
- Do not implement or optimize the `k2_` tcgen05 kernel for `sm_120`.
- First build a remote execution harness and validate it using the existing
  `sm_120` kernels on a manually launched RTX 5090 pod.
- The first harness version connects to an already-running pod over SSH. It
  does not create, stop, or terminate pods and does not require a RunPod API
  credential.
- The pod remains running after artifact retrieval. The user stops it
  manually.
- After the harness is validated, use the same mechanism for occasional
  tcgen05 validation on confirmed `sm_100` hardware and for future Hopper
  (`sm_90`) development.
- The tcgen05 effort is limited to one base kernel and correctness/basic
  measurement. No k2 autotuning or Blackwell optimization program is planned.

## Remote environment

- [ ] Use the manually launched image:

  ```text
  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
  ```

- [ ] Confirm remotely that the image provides the required development tools:

  ```sh
  nvidia-smi
  nvidia-smi --query-gpu=name,compute_cap --format=csv
  nvcc --version
  cmake --version
  ```

- [ ] Confirm the actual GPU reports `sm_120`/compute capability 12.0 for the
      RTX 5090 test. This run validates the harness and existing kernels only;
      it must not be used to validate `tcgen05.mma`.
- [ ] Confirm `/workspace` is writable by the SSH user before uploading code.
- [ ] Record GPU, driver, CUDA compiler, CMake, requested architecture, and
      command in the run manifest.

## 1. Implement the SSH smoke harness

- [ ] Add one local entry point under `scripts/` for an already-running pod.
      Do not add RunPod API lifecycle code in this first version.
- [ ] Accept these inputs:

  ```text
  --host
  --port
  --user
  --identity-file
  --arch              default: sm_120
  --artifact-dir
  --timeout
  ```

- [ ] Do not accept or store private-key contents. Use the supplied local SSH
      identity-file path.
- [ ] Use bounded SSH connection retries and a startup/connection timeout.
- [ ] Use a temporary known-hosts file. Do not permanently modify the user's
      SSH configuration.
- [ ] Print the generated run ID and remote workspace immediately so a failed
      local process can still be cleaned up manually.

## 2. Transfer an isolated source snapshot

- [ ] Default to the current working tree so uncommitted kernel changes can be
      tested.
- [ ] Create a temporary archive locally and exclude:

  ```text
  .git/
  build/
  results/
  db/
  compile_commands.json
  ```

- [ ] Upload and extract the archive into a unique remote workspace:

  ```text
  /workspace/gemm_y_remote/<run-id>/
  ```

- [ ] Use this remote layout:

  ```text
  /workspace/gemm_y_remote/<run-id>/
  ├── source/
  ├── build/
  ├── logs/
  └── artifacts/
  ```

- [ ] Run all build and test commands from `source/`; do not depend on a
      pre-existing checkout or stale remote build directory.
- [ ] Remove only the harness-created run directory when explicitly requested;
      do not delete unrelated files under `/workspace`.

## 3. Run the first smoke test

- [ ] Capture toolchain information before building:

  ```sh
  nvidia-smi > logs/nvidia-smi.txt
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv \
      > logs/gpu.txt
  nvcc --version > logs/nvcc.txt
  cmake --version > logs/cmake.txt
  ```

- [ ] Configure the existing project for the RTX 5090:

  ```sh
  cmake -B build -DGEMM_Y_CUDA_ARCH=sm_120
  ```

- [ ] Build only the focused smoke target:

  ```sh
  cmake --build build --target kernel_smoke -j
  ```

- [ ] Run the smoke executable with a bounded timeout:

  ```sh
  ./build/kernel_smoke
  ```

- [ ] Capture configure, build, and execution output independently. Preserve
      logs when any stage fails.
- [ ] Return the remote command's exit status to the local harness.
- [ ] Do not run the full benchmark during harness bring-up.

## 4. Retrieve and describe artifacts

- [ ] Copy the following back to the requested local artifact directory:

  ```text
  nvidia-smi.txt
  gpu.txt
  nvcc.txt
  cmake.txt
  configure.log
  build.log
  smoke.log
  manifest.json
  ```

- [ ] Support CSV and `.meta` retrieval now, even though the first smoke test
      is not required to produce them. The next harness milestone will use
      this path for benchmark data.
- [ ] Generate a manifest containing at least:

  ```json
  {
    "run_id": "...",
    "remote_workspace": "/workspace/gemm_y_remote/...",
    "requested_arch": "sm_120",
    "command": "kernel_smoke",
    "gpu_name": "...",
    "compute_capability": "...",
    "cuda_version": "...",
    "driver_version": "...",
    "cmake_version": "...",
    "source_mode": "working-tree",
    "exit_code": 0,
    "started_at": "...",
    "finished_at": "...",
    "artifacts": []
  }
  ```

- [ ] Do not include SSH private keys, API credentials, or sensitive
      environment variables in logs or manifests.
- [ ] Leave the pod running after successful or failed artifact retrieval;
      pod shutdown is a manual user action.

## 5. Validate harness behavior

- [ ] Perform a local dry run covering argument parsing and archive exclusion.
- [ ] Run the remote toolchain-only command and verify SSH, `/workspace`, and
      artifact retrieval.
- [ ] Run the complete remote `kernel_smoke` workflow on the RTX 5090.
- [ ] Verify that build failures and smoke-test failures still download logs
      and produce a nonzero local exit status.
- [ ] Verify that interrupted local execution does not falsely report success.
- [ ] Verify that no local `db/gemm_y.db` or remote database is modified.
- [ ] Manually stop the pod after confirming the artifacts are available.

## 6. Next harness milestone: benchmark retrieval

- [ ] Add an explicit, bounded benchmark/profile command after smoke validation
      succeeds.
- [ ] Retrieve generated CSV and `.meta` files without automatic ingestion.
- [ ] Preserve remote stdout/stderr and the manifest beside the CSV artifacts.
- [ ] Use a small benchmark scope first to control pod runtime and cost.
- [ ] Do not run `sudo`, modify SQLite, or claim reproducible performance until
      the user reviews the retrieved result and confirms the clock-locking
      procedure for the remote GPU.

## Future phase: tcgen05 validation

- [ ] Use the harness on confirmed `sm_100` hardware only; an RTX 5090 cannot
      validate `tcgen05.mma`.
- [ ] Add `sm_100` build support only when a compatible remote target is
      available and verified.
- [ ] Implement one fixed `k2` base kernel using 2D TMA and `tcgen05.mma`:
      initially `128x128x64`, one CTA tile, one stage, ColMajor A/B/C, BF16
      inputs, and FP32 accumulation.
- [ ] Reuse the existing references only for instruction sequencing and replace
      their row-major indexing, Torch dependencies, and unchecked Driver API
      calls.
- [ ] Validate correctness and basic CUDA-event timing remotely before any
      further k2 work.

## Future primary phase: Hopper

- [ ] Shift optimization focus to `sm_90` after the limited tcgen05 experiment.
- [ ] Use the remote harness for targeted Hopper smoke tests and benchmarks
      only when local hardware is insufficient.
- [ ] Keep Hopper candidates independently identifiable and out of production
      dispatch until correctness and reproducible timing are validated.
