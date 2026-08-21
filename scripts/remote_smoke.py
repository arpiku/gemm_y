#!/usr/bin/env python3
"""Run the existing kernel smoke test on an already-running SSH pod.

This is intentionally a connection-and-test harness, not a pod lifecycle tool.
It snapshots the current working tree, excludes local build/results/database/
artifact state, runs only ``kernel_smoke`` remotely, and retrieves logs and metadata.
No database ingestion or sudo operation is performed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

DEFAULT_REMOTE_ROOT = "/workspace/gemm_y_remote"
EXCLUDED_TOP_LEVEL = {
    ".git",
    "build",
    "results",
    "db",
    "artifacts",
    "compile_commands.json",
}
RETRIEVED_FILES = (
    "logs/nvidia-smi.txt",
    "logs/gpu.txt",
    "logs/nvcc.txt",
    "logs/cmake.txt",
    "logs/configure.log",
    "logs/build.log",
    "logs/smoke.log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gemm_y kernel_smoke on an already-running SSH pod."
    )
    parser.add_argument("--host", required=True, help="SSH host or address")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--user", required=True, help="SSH login user")
    parser.add_argument(
        "--identity-file", required=True, type=Path, help="local SSH private-key path"
    )
    parser.add_argument(
        "--arch",
        default="sm_120",
        choices=("sm_90", "sm_120"),
        help="GEMM_Y_CUDA_ARCH passed to CMake (default: sm_120)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/remote_smoke"),
        help="local directory for retrieved artifacts",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="overall timeout in seconds for each remote stage (default: 900)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate archive creation and command setup without SSH",
    )
    return parser.parse_args()


def run_id() -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def is_excluded_top_level(name: str) -> bool:
    return name in EXCLUDED_TOP_LEVEL or name.startswith(("build-", "build_"))


def make_archive(source_root: Path) -> Path:
    """Create a temporary gzipped snapshot of source_root's working tree."""
    archive_file = tempfile.NamedTemporaryFile(
        prefix="gemm_y_remote_", suffix=".tar.gz", delete=False
    )
    archive_file.close()
    archive_path = Path(archive_file.name)

    def include(path: Path) -> bool:
        relative = path.relative_to(source_root)
        return not relative.parts or not is_excluded_top_level(relative.parts[0])

    with tarfile.open(archive_path, "w:gz") as archive:
        for child in sorted(source_root.iterdir()):
            if include(child):
                archive.add(child, arcname=child.name, recursive=True, filter=None)
    return archive_path


def ssh_base(args: argparse.Namespace, known_hosts: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.identity_file),
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
        f"{args.user}@{args.host}",
    ]


def scp_base(args: argparse.Namespace, known_hosts: Path) -> list[str]:
    return [
        "scp",
        "-i",
        str(args.identity_file),
        "-P",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=10",
    ]


def run_with_retries(
    command: list[str],
    timeout: int,
    label: str,
    *,
    input_text: str | None = None,
    retry: bool = True,
) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + timeout
    last: subprocess.CompletedProcess[str] | None = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        remaining = max(1, int(deadline - time.monotonic()))
        print(f"[remote-smoke] {label} (attempt {attempt})", flush=True)
        try:
            result = subprocess.run(
                command,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"[remote-smoke] {label} timed out", file=sys.stderr, flush=True)
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return subprocess.CompletedProcess(command, 124, stdout, stderr)
        last = result
        if result.returncode == 0 or not retry:
            return result
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(3, max(1, int(deadline - time.monotonic()))))
    assert last is not None
    return last


def remote_script(remote_workspace: str, arch: str, timeout: int) -> str:
    # Every stage writes its own log and continues to the next stage so logs
    # remain available when configure, build, or execution fails.
    quoted_workspace = shlex.quote(remote_workspace)
    quoted_arch = shlex.quote(arch)
    return f"""#!/bin/sh
set -u
WORKSPACE={quoted_workspace}
ARCH={quoted_arch}
cd "$WORKSPACE/source" || exit 20
mkdir -p "$WORKSPACE/logs" "$WORKSPACE/build" "$WORKSPACE/artifacts"

nvidia-smi > "$WORKSPACE/logs/nvidia-smi.txt" 2>&1 || true
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv > "$WORKSPACE/logs/gpu.txt" 2>&1 || true
nvcc --version > "$WORKSPACE/logs/nvcc.txt" 2>&1 || true
cmake --version > "$WORKSPACE/logs/cmake.txt" 2>&1 || true

case "$ARCH" in
  sm_120) expected_cap="12.0" ;;
  sm_90) expected_cap="9.0" ;;
  *) expected_cap="" ;;
esac
reported_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NR == 1 {{gsub(/[[:space:]]/, ""); print}}')
if [ -z "$expected_cap" ] || [ "$reported_cap" != "$expected_cap" ]; then
  printf '%s\\n' "stage=gpu_check" "reported_compute_capability=$reported_cap" "expected_compute_capability=$expected_cap" > "$WORKSPACE/logs/status.txt"
  exit 127
fi

cmake -S "$WORKSPACE/source" -B "$WORKSPACE/build" -DGEMM_Y_CUDA_ARCH="$ARCH" > "$WORKSPACE/logs/configure.log" 2>&1
configure_rc=$?
if [ "$configure_rc" -eq 0 ]; then
  cmake --build "$WORKSPACE/build" --target kernel_smoke -j > "$WORKSPACE/logs/build.log" 2>&1
  build_rc=$?
else
  build_rc=125
fi
if [ "$configure_rc" -eq 0 ] && [ "$build_rc" -eq 0 ]; then
  timeout {timeout} "$WORKSPACE/build/kernel_smoke" > "$WORKSPACE/logs/smoke.log" 2>&1
  smoke_rc=$?
else
  smoke_rc=126
fi
if [ "$configure_rc" -ne 0 ]; then
  failed_stage="configure"
elif [ "$build_rc" -ne 0 ]; then
  failed_stage="build"
else
  failed_stage="smoke"
fi
printf '%s\\n' "stage=$failed_stage" "configure=$configure_rc" "build=$build_rc" "smoke=$smoke_rc" > "$WORKSPACE/logs/status.txt"
if [ "$configure_rc" -ne 0 ]; then exit "$configure_rc"; fi
if [ "$build_rc" -ne 0 ]; then exit "$build_rc"; fi
exit "$smoke_rc"
"""


def read_first_line(path: Path) -> str:
    try:
        return (
            path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
        )
    except (OSError, IndexError):
        return "unknown"


def gpu_manifest_fields(path: Path) -> tuple[str, str, str]:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            rows = csv.reader(stream)
            row = next(rows)
            if row and row[0].strip().lower() == "name":
                row = next(rows)
        if len(row) >= 3:
            values = tuple(value.strip() for value in row[:3])
            return (values[0], values[1], values[2])
    except (OSError, StopIteration, csv.Error):
        pass
    return ("unknown", "unknown", "unknown")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    remote_workspace: str,
    started: str,
    finished: str,
    exit_code: int,
    artifacts: list[str],
    *,
    stage: str,
    kernel_exit_code: int | None = None,
    retrieval_exit_code: int | None = None,
    error: str | None = None,
) -> None:
    gpu_name, compute_capability, driver_version = gpu_manifest_fields(
        path.parent / "logs/gpu.txt"
    )
    nvcc_text = read_first_line(path.parent / "logs/nvcc.txt")
    cmake_text = read_first_line(path.parent / "logs/cmake.txt")
    manifest = {
        "run_id": path.parent.name,
        "remote_workspace": remote_workspace,
        "requested_arch": args.arch,
        "command": "kernel_smoke",
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "cuda_version": nvcc_text,
        "driver_version": driver_version,
        "cmake_version": cmake_text,
        "source_mode": "working-tree",
        "exit_code": exit_code,
        "stage": stage,
        "kernel_exit_code": kernel_exit_code,
        "retrieval_exit_code": retrieval_exit_code,
        "error": error,
        "started_at": started,
        "finished_at": finished,
        "artifacts": artifacts,
    }
    (path.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    if not args.dry_run and not args.identity_file.is_file():
        raise SystemExit(f"identity file does not exist: {args.identity_file}")

    source_root = Path(__file__).resolve().parents[1]
    archive_path = make_archive(source_root)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
        excluded_present = [
            name for name in names if is_excluded_top_level(name.split("/", 1)[0])
        ]
        if excluded_present:
            raise RuntimeError(f"archive exclusion failed: {excluded_present[:3]}")
        print(f"[remote-smoke] archive ready: {archive_path}")
        if args.dry_run:
            print(f"[remote-smoke] dry run passed ({len(names)} archive entries)")
            return 0

        if not args.artifact_dir.is_absolute():
            args.artifact_dir = Path.cwd() / args.artifact_dir
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        current_run_id = run_id()
        remote_workspace = f"{DEFAULT_REMOTE_ROOT}/{current_run_id}"
        print(f"[remote-smoke] run_id={current_run_id}", flush=True)
        print(f"[remote-smoke] remote_workspace={remote_workspace}", flush=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        local_run_dir = args.artifact_dir / current_run_id
        local_run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = local_run_dir / "manifest.json"
        stage = "connect"
        kernel_exit: int | None = None
        retrieval_exit: int | None = None
        write_manifest(
            manifest_path,
            args,
            remote_workspace,
            started,
            started,
            125,
            [],
            stage=stage,
            error="remote workflow not started",
        )

        with tempfile.TemporaryDirectory(
            prefix="gemm_y_known_hosts_"
        ) as known_hosts_dir:
            known_hosts = Path(known_hosts_dir) / "known_hosts"
            ssh = ssh_base(args, known_hosts)
            scp = scp_base(args, known_hosts)
            setup = " && ".join(
                (
                    f"mkdir -p {shlex.quote(remote_workspace)}/source",
                    f"test -w {shlex.quote(DEFAULT_REMOTE_ROOT)}",
                )
            )
            stage = "workspace"
            result = run_with_retries(
                ssh + [setup], args.timeout, "connect/check workspace"
            )
            if result.returncode != 0:
                finished = dt.datetime.now(dt.timezone.utc).isoformat()
                write_manifest(
                    manifest_path,
                    args,
                    remote_workspace,
                    started,
                    finished,
                    result.returncode,
                    [],
                    stage=stage,
                    error=result.stderr.strip() or "workspace check failed",
                )
                return result.returncode

            stage = "upload"
            remote_archive = f"{remote_workspace}/source_snapshot.tar.gz"
            result = run_with_retries(
                scp + [str(archive_path), f"{args.user}@{args.host}:{remote_archive}"],
                args.timeout,
                "upload archive",
            )
            if result.returncode != 0:
                finished = dt.datetime.now(dt.timezone.utc).isoformat()
                write_manifest(
                    manifest_path,
                    args,
                    remote_workspace,
                    started,
                    finished,
                    result.returncode,
                    [],
                    stage=stage,
                    error=result.stderr.strip() or "archive upload failed",
                )
                return result.returncode
            stage = "extract"
            extract = (
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_workspace)}/source "
                f"&& rm -f {shlex.quote(remote_archive)}"
            )
            result = run_with_retries(ssh + [extract], args.timeout, "extract archive")
            if result.returncode != 0:
                finished = dt.datetime.now(dt.timezone.utc).isoformat()
                write_manifest(
                    manifest_path,
                    args,
                    remote_workspace,
                    started,
                    finished,
                    result.returncode,
                    [],
                    stage=stage,
                    error=result.stderr.strip() or "archive extraction failed",
                )
                return result.returncode

            stage = "configure"
            script = remote_script(remote_workspace, args.arch, args.timeout)
            result = run_with_retries(
                ssh + ["sh -s"],
                args.timeout,
                "configure/build/smoke",
                input_text=script,
                retry=False,
            )
            remote_exit = result.returncode
            kernel_exit = remote_exit
            stage = "retrieve"

            retrieve = local_run_dir
            retrieved = list(RETRIEVED_FILES) + ["logs/status.txt"]
            listing = run_with_retries(
                ssh
                + [
                    "find",
                    remote_workspace,
                    "-type",
                    "f",
                    "\\(",
                    "-name",
                    "*.csv",
                    "-o",
                    "-name",
                    "*.meta",
                    "\\)",
                ],
                args.timeout,
                "list CSV/metadata artifacts",
                retry=False,
            )
            if listing.returncode != 0:
                retrieval_exit = listing.returncode
            else:
                for remote_path in listing.stdout.splitlines():
                    prefix = remote_workspace.rstrip("/") + "/"
                    if remote_path.startswith(prefix):
                        relative_path = remote_path[len(prefix) :]
                        if relative_path not in retrieved:
                            retrieved.append(relative_path)
            for relative in retrieved:
                destination = retrieve / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = f"{args.user}@{args.host}:{remote_workspace}/{relative}"
                try:
                    copied = subprocess.run(
                        scp + [source, str(destination)],
                        text=True,
                        capture_output=True,
                        timeout=args.timeout,
                        check=False,
                    )
                    if copied.returncode != 0:
                        retrieval_exit = copied.returncode
                        print(
                            f"[remote-smoke] warning: could not retrieve {relative}: {copied.stderr.rstrip()}",
                            file=sys.stderr,
                        )
                except subprocess.TimeoutExpired:
                    print(
                        f"[remote-smoke] warning: timed out retrieving {relative}",
                        file=sys.stderr,
                    )
                    retrieval_exit = 124

        status_path = local_run_dir / "logs/status.txt"
        if status_path.is_file():
            first_status = read_first_line(status_path)
            if first_status.startswith("stage="):
                stage = first_status.split("=", 1)[1]
        finished = dt.datetime.now(dt.timezone.utc).isoformat()
        artifacts = [
            str(path.relative_to(local_run_dir))
            for path in sorted(local_run_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ]
        final_exit = retrieval_exit if retrieval_exit is not None else remote_exit
        write_manifest(
            manifest_path,
            args,
            remote_workspace,
            started,
            finished,
            final_exit,
            artifacts,
            stage=stage,
            kernel_exit_code=kernel_exit,
            retrieval_exit_code=retrieval_exit,
            error="artifact retrieval failed" if retrieval_exit is not None else None,
        )
        print(f"[remote-smoke] artifacts: {local_run_dir}")
        return final_exit
    finally:
        try:
            os.unlink(archive_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
