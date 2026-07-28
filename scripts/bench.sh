#!/usr/bin/env bash
# bench.sh — reproducible benchmark wrapper (must be run with sudo).
#
# Locks the GPU graphics clock to max frequency, runs ./build/gemm_y as the
# original (non-root) user, then resets clock/persistence on exit. Trap
# ensures reset even on Ctrl-C or crash (ARD §18).
#
# Usage:
#   sudo scripts/bench.sh
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
MAX_GR=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader -i "${GPU_ID}" | head -n1 | tr -d ' ')
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
echo "[bench] running ./build/gemm_y as '${ORIG_USER}'..."
sudo -u "${ORIG_USER}" ./build/gemm_y

# 6. Post-run temperature (thermal delta sanity check).
TEMP_AFTER=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i "${GPU_ID}" | head -n1 | tr -d ' ')
echo "[bench] post-run temperature: ${TEMP_AFTER} C (delta: $(( TEMP_AFTER - TEMP_BEFORE )) C)"

# 7. Reset happens in the EXIT trap.
