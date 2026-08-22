#!/usr/bin/env bash
# bench.sh — reproducible benchmark wrapper (must be run with sudo).
#
# Locks the GPU graphics clock to max frequency, runs the selected benchmark
# binary as the original (non-root) user, then resets clock/persistence on
# exit. Trap ensures reset even on Ctrl-C or crash (ARD §18).
#
# Usage:
#   sudo scripts/bench.sh                         # runs ./build/gemm_y
#   sudo scripts/bench.sh ./build/gemm_y
#
# Why sudo: nvidia-smi clock locking (-lgc) requires root. The binary
# itself runs as the original user so build artifacts aren't owned by root.
#
# No fan control: consumer cards (e.g. RTX 5070) lock the auto-fan curve
# to the driver — manual fan control (-fan-gpu) returns "Not Supported".
# The auto-fan curve is the only path that ever runs locally; relying on
# it is simpler than carrying an attempt-and-warn fallback that never
# fires. Datacenter cards (H100) support manual fan control but it is not
# orchestrated by this wrapper — the auto curve suffices for short sweeps.

set -euo pipefail

GPU_ID=${GPU_ID:-0}
BENCH_BINARY=${1:-./build/gemm_y}

# The original (pre-sudo) user, so we can drop privileges for the binary.
ORIG_USER="${SUDO_USER:-$USER}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "error: bench.sh must be run with sudo (nvidia-smi clock locking requires root)" >&2
    exit 1
fi

if [[ -z "${ORIG_USER}" || "${ORIG_USER}" == "root" ]]; then
    echo "error: could not determine original (non-root) user (SUDO_USER empty)" >&2
    exit 1
fi

echo "[bench] GPU ${GPU_ID}, running binary as '${ORIG_USER}'"

# --- reset helpers (called on EXIT/INT/TERM) --------------------------------
reset_state() {
    local rc=$?
    echo "[bench] resetting GPU state..."
    # Reset clock lock. -rgc may print "Not Supported" if no lock was set;
    # tolerate.
    nvidia-smi -i "${GPU_ID}" -rgc 2>/dev/null || true
    # Reset persistence mode (optional; leaving it on is harmless).
    nvidia-smi -i "${GPU_ID}" -pm 0 2>/dev/null || true
    echo "[bench] done (exit code ${rc})."
}
trap reset_state EXIT INT TERM

# --- setup ------------------------------------------------------------------
# 1. Persistence mode — keeps the GPU driver loaded between processes
#    (avoids the ~1s init latency on first CUDA call).
echo "[bench] enabling persistence mode..."
nvidia-smi -i "${GPU_ID}" -pm 1

# 2. Query max graphics clock.
MAX_GR=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader -i "${GPU_ID}" |
    head -n1 | tr -cd '0-9')
if [[ -z "${MAX_GR}" ]]; then
    echo "error: could not query max graphics clock" >&2
    exit 1
fi
echo "[bench] max graphics clock: ${MAX_GR} MHz"

# 3. Lock graphics clock to max frequency (within the GPU's validated boost
#    range — not overclocking).
echo "[bench] locking graphics clock to ${MAX_GR} MHz..."
if ! nvidia-smi -i "${GPU_ID}" -lgc "${MAX_GR},${MAX_GR}"; then
    echo "[bench] WARNING: could not lock graphics clock (continuing without lock)" >&2
fi

# 4. Pre-run thermal baseline.
TEMP_BEFORE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i "${GPU_ID}" | head -n1 | tr -d ' ')
echo "[bench] pre-run temperature: ${TEMP_BEFORE} C"

# 5. Drop privileges and run the binary as the original user.
if [[ ! -x "${BENCH_BINARY}" ]]; then
    echo "error: benchmark binary is not executable: ${BENCH_BINARY}" >&2
    exit 1
fi

# sudo -u sanitizes the dynamic-loader environment. Add the library directory
# belonging to the active CUDA installation explicitly. The binary may require
# a particular cuBLAS soname (for example libcublas.so.12), so validate that
# exact dependency instead of assuming the active toolkit major version.
CUDA_LIB_DIR=""
NVCC_PATH=$(command -v nvcc || true)
if [[ -n "${NVCC_PATH}" ]]; then
    NVCC_PATH=$(readlink -f "${NVCC_PATH}")
    CUDA_ROOT=$(dirname -- "$(dirname -- "${NVCC_PATH}")")
    if compgen -G "${CUDA_ROOT}/lib64/libcublas.so.*" >/dev/null; then
        CUDA_LIB_DIR="${CUDA_ROOT}/lib64"
    fi
fi
if [[ -z "${CUDA_LIB_DIR}" ]]; then
    for candidate in /usr/local/cuda/lib64 /usr/local/cuda-*/lib64 /opt/cuda/lib64; do
        if compgen -G "${candidate}/libcublas.so.*" >/dev/null; then
            CUDA_LIB_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -z "${CUDA_LIB_DIR}" ]]; then
    echo "error: could not locate a CUDA cuBLAS library directory" >&2
    exit 1
fi

BENCH_LD_LIBRARY_PATH="${CUDA_LIB_DIR}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    BENCH_LD_LIBRARY_PATH="${BENCH_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

# Check the dependency using the same library path that will be passed to
# sudo -u. This catches stale builds, such as a CUDA 12 binary on a CUDA 13
# host, before starting the benchmark.
MISSING_LIBRARY=$(LD_LIBRARY_PATH="${BENCH_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    ldd "${BENCH_BINARY}" | awk '/=> not found/ {print $1; exit}')
if [[ -n "${MISSING_LIBRARY}" ]]; then
    echo "error: ${BENCH_BINARY} requires ${MISSING_LIBRARY}, but it is not available" >&2
    echo "error: rebuild the binary with the active CUDA toolkit (${CUDA_ROOT:-unknown})" >&2
    exit 1
fi

echo "[bench] CUDA library directory: ${CUDA_LIB_DIR}"
echo "[bench] running ${BENCH_BINARY} as '${ORIG_USER}'..."
sudo -u "${ORIG_USER}" env "LD_LIBRARY_PATH=${BENCH_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${BENCH_BINARY}"

# 6. Post-run temperature (thermal delta sanity check).
TEMP_AFTER=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i "${GPU_ID}" | head -n1 | tr -d ' ')
echo "[bench] post-run temperature: ${TEMP_AFTER} C (delta: $(( TEMP_AFTER - TEMP_BEFORE )) C)"

# 7. Reset happens in the EXIT trap.
