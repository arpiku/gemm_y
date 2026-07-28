#!/usr/bin/env python3
"""Dash dashboard for gemm_y benchmark results.

Served at http://localhost:8050. Reads from db/gemm_y.db (the source of
truth); never from CSV. Run `ingest.py` first to populate the DB.

Layout: single page, sidebar + six tabs (Timing / Comparison / Accuracy /
Speedup / Diff / Run History). Sidebar filters: arch, dtype, kernel class
(Custom/cuBLAS), runs multi-select, Diff run B (single-select), scale
(log-log / linear), chart height. The Comparison tab plots `perf_pct` vs N
(ARD §15) with a parity line at 0; it ignores the scale toggle (always
linear-y, log-x).

The sidebar's run dropdown is populated once at startup. If you ingest new
runs while the server is running, restart the server to pick them up.

Usage:
    python scripts/server.py
    python scripts/server.py --port 8050
"""

from __future__ import annotations

import argparse
import sys

import dash
import dash.dash_table as dt
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
import db

# Okabe-Ito palette (colorblind-safe) for custom kernels.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
]
CUBLAS_COLOR = "#000000"  # black; cuBLAS is the reference line

# Significance ring colors for the Timing tab overlay (2G.6.1).
SIG_FASTER = "#009E73"   # green — custom is significantly faster
SIG_SLOWER = "#D55E00"   # vermillion — custom is significantly slower
SIG_INCONCLUSIVE = "#999999"  # gray — CIs overlap, inconclusive

# The 14-size square sweep (powers of 2 + midpoints). Used as log-x tickvals
SWEEP_SIZES = [32, 64, 96, 128, 192, 256, 384, 512, 768,
               1024, 1536, 2048, 3072, 4096]

PLOT_BG_COLOR = "#ffffff"   # white plot area
PAPER_BG_COLOR = "#f0f0f0"  # light gray page / margin

BASE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12

_GRID_BOLD = {"gridwidth": 1, "gridcolor": "rgba(0,0,0,0.35)", "tickwidth": 1}
_GRID_LIGHT = {"gridwidth": 1, "gridcolor": "rgba(0,0,0,0.15)", "tickwidth": 1}
_GRID_PRESETS = {"bold": _GRID_BOLD, "light": _GRID_LIGHT}

# user said the chart felt too short, especially the accuracy tab.
DEFAULT_CHART_HEIGHT = 640


def _axis_layout(log_x: bool, log_y: bool, zeroline_y: bool = False,
                 grid_weight: str = "bold") -> dict:
    """Shared per-axis grid + tick styling (TODO 2B.3.2 / 2B.3.6).

    - grid_weight: "bold" (timing, comparison — wide y-range) or "light"
      (accuracy — tiny y-range, gridwidth=2 merges into a solid block).
      tickwidth follows gridwidth so outside ticks match the grid weight.
    - Outside ticks (len=6) for a crisper read.
    - Log-x: tickvals at the 14 sweep sizes so every data point has a tick.
      Linear-x: leave Plotly auto (sweep sizes are not evenly spaced).
    - Log-y: dtick='D1' (every decade) — reads well for the 10ns–100ms range.
    - zeroline_y: when True (Comparison tab), the y=0 parity line is drawn
      bolder than the grid so it stands out.
    """
    grid = _GRID_PRESETS[grid_weight]
    xaxis = dict(
        showgrid=True,
        gridwidth=grid["gridwidth"],
        gridcolor=grid["gridcolor"],
        ticks="outside",
        tickwidth=grid["tickwidth"],
        ticklen=6,
        zeroline=False,
    )
    yaxis = dict(
        showgrid=True,
        gridwidth=grid["gridwidth"],
        gridcolor=grid["gridcolor"],
        ticks="outside",
        tickwidth=grid["tickwidth"],
        ticklen=6,
        zeroline=zeroline_y,
        zerolinewidth=2,
        zerolinecolor=grid["gridcolor"],
    )
    if log_x:
        xaxis["type"] = "log"
        xaxis["tickvals"] = SWEEP_SIZES
    else:
        xaxis["type"] = "linear"
    if log_y:
        yaxis["type"] = "log"
        yaxis["dtick"] = "D1"  # every decade
    else:
        yaxis["type"] = "linear"
    return {"xaxis": xaxis, "yaxis": yaxis}


def _base_layout(title: str, x_title: str, y_title: str,
                 log_x: bool, log_y: bool,
                 zeroline_y: bool = False,
                 grid_weight: str = "bold",
                 height: int = DEFAULT_CHART_HEIGHT) -> dict:
    """Shared layout dict: axes + theme + font + legend (TODO 2B.3.2 / 2B.3.3 /
    2B.3.6 / 2B.3.7).

    Legend font is one step smaller than the base font to keep the legend
    compact when many runs are selected. Height is runtime-configurable via
    the sidebar's chart-height control (threaded through render_tab).
    """
    layout = {
        "title": title,
        "xaxis_title": x_title,
        "yaxis_title": y_title,
        "plot_bgcolor": PLOT_BG_COLOR,
        "paper_bgcolor": PAPER_BG_COLOR,
        "font": dict(size=BASE_FONT_SIZE),
        "legend": dict(
            orientation="h", y=-0.2, x=0, xanchor="left",
            font=dict(size=LEGEND_FONT_SIZE),
        ),
        "margin": dict(l=60, r=20, t=50, b=80),
        "height": height,
    }
    layout.update(_axis_layout(log_x, log_y, zeroline_y=zeroline_y,
                               grid_weight=grid_weight))
    return layout


def _is_cublas(kernel_name: str) -> bool:
    return kernel_name == "cublas"


def _speedup(median: float, ref_median: float) -> float | None:
    """ref_median / median — >1 means custom is faster than cuBLAS."""
    if not median or median <= 0:
        return None
    return ref_median / median


def _perf_pct(row: dict) -> float | None:
    """Time-reduction % vs cuBLAS (ARD §15). +X = X% faster, -X = X% slower.

    None for cuBLAS rows (undefined for the self-reference)."""
    val = row.get("perf_pct")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _significance(row: dict) -> str | None:
    """Significance of custom vs cuBLAS at one point (2G.6.1, ARD §20).

    Returns 'faster', 'slower', 'inconclusive', or None (missing CI data).
    A point is significant (custom ≠ cuBLAS) if the custom CI and cuBLAS CI
    do not overlap vertically. Overlap -> inconclusive at 95%.
    """
    k_lo = row.get("kernel_ci_low_ns")
    k_hi = row.get("kernel_ci_high_ns")
    r_lo = row.get("ref_kernel_ci_low_ns")
    r_hi = row.get("ref_kernel_ci_high_ns")
    if None in (k_lo, k_hi, r_lo, r_hi):
        return None
    # No overlap -> significant. Custom is faster if its CI is entirely below.
    if k_hi < r_lo:
        return "faster"
    if k_lo > r_hi:
        return "slower"
    return "inconclusive"


def _propagated_ci_half(median: float, std: float,
                        ref_median: float, ref_std: float) -> float | None:
    """First-order Gaussian CI half-width for a ratio median/ref_median
    (2G.6.5 / 2G.6.8, ARD §20).

    ``ratio = median / ref_median``; the propagated relative SE is
    ``sqrt((σ/median)^2 + (σ_ref/median_ref)^2)`` and the CI half-width is
    ``ratio * 1.253 * relative_se / sqrt(n)`` (the 1.253 factor is the
    asymptotic SE of the median; n=50 is implicit — we fold it into the
    std which is already the sample std). For simplicity we use the
    asymptotic form: ``ratio * sqrt((std/median)^2 + (ref_std/ref_median)^2)``
    which is the first-order propagation without the 1.253/sqrt(n) factor
    (the std already reflects the sample size).
    """
    if not median or not ref_median or median <= 0 or ref_median <= 0:
        return None
    rel_k = (std / median) if std else 0.0
    rel_r = (ref_std / ref_median) if ref_std else 0.0
    return median / ref_median * (rel_k ** 2 + rel_r ** 2) ** 0.5


def _timing_figure(rows: list[dict], log_log: bool,
                  height: int = DEFAULT_CHART_HEIGHT) -> go.Figure:
    """kernel_median_ns vs N, one line per (run, kernel), with per-point
    95% CI error bars (ARD §19)."""
    fig = go.Figure()
    # Group rows by (run_id, kernel_name) so each gets its own line.
    series: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        series.setdefault((r["run_id"], r["kernel_name"]), []).append(r)

    # Sort series so cuBLAS is drawn first (reference), then custom by name.
    def series_key(item: tuple[tuple[int, str], list[dict]]) -> tuple:
        (run_id, kname), rs = item
        is_cublas = _is_cublas(kname)
        # cuBLAS first (0), then custom (1) by name; tie-break by run_id.
        return (0 if is_cublas else 1, kname, run_id)

    color_idx = 0
    for (run_id, kname), rs in sorted(series.items(), key=series_key):
        rs_sorted = sorted(rs, key=lambda r: r["n"])
        xs = [r["n"] for r in rs_sorted]
        ys = [r["kernel_median_ns"] for r in rs_sorted]
        # Per-point 95% CI half-width: median - ci_low (= ci_high - median,
        # since the CI is symmetric). None where stats are missing (old runs).
        def _ci_half(r: dict) -> float | None:
            med = r.get("kernel_median_ns")
            lo = r.get("kernel_ci_low_ns")
            if med is None or lo is None:
                return None
            return med - lo
        err_y = [_ci_half(r) for r in rs_sorted]
        # customdata carries the hover extras.
        customdata = [
            [
                r["arch"],
                r["dtype"],
                "cuBLAS" if _is_cublas(kname) else "custom",
                r["kernel_desc"],
                r["kernel_median_ns"],
                r["ref_kernel_median_ns"],
                _speedup(r["kernel_median_ns"], r["ref_kernel_median_ns"]),
                _perf_pct(r),
                r.get("kernel_std_ns"),
                r.get("kernel_p95_ns"),
                r.get("kernel_ci_low_ns"),
                r.get("kernel_ci_high_ns"),
                r.get("ref_kernel_std_ns"),
                r.get("ref_kernel_p95_ns"),
                r.get("ref_kernel_ci_low_ns"),
                r.get("ref_kernel_ci_high_ns"),
            ]
            for r in rs_sorted
        ]
        is_cublas = _is_cublas(kname)
        # Label includes run_id so multiple ingests of the same kernel
        # are distinguishable.
        label = f"{kname} (run {run_id})"
        # Shared hovertemplate — thousands separator on ns values.
        # customdata indices:
        #   [0] arch   [1] dtype   [2] class   [3] kernel_desc
        #   [4] kernel_median_ns   [5] ref_kernel_median_ns
        #   [6] speedup            [7] perf_pct
        #   [8] kernel_std_ns      [9] kernel_p95_ns
        #   [10] kernel_ci_low_ns  [11] kernel_ci_high_ns
        #   [12] ref_kernel_std_ns     [13] ref_kernel_p95_ns
        #   [14] ref_kernel_ci_low_ns  [15] ref_kernel_ci_high_ns
        #
        # perf_pct is None for cuBLAS rows; the %{customdata[7]:+.2f} format
        # renders 'nan' for None, so cuBLAS uses a separate hovertemplate.
        # Stat fields may be None for old (pre-2E) runs; format with a
        # fallback that renders '—' for None.
        hovertemplate = (
            "<b>%{fullData.name}</b><br>"
            "N=%{x}<br>"
            "median=%{y:,.0f} ns<br>"
            "arch=%{customdata[0]}<br>"
            "dtype=%{customdata[1]}<br>"
            "class=%{customdata[2]}<br>"
            "desc=%{customdata[3]}<br>"
            "std=%{customdata[8]:,.0f} ns<br>"
            "p95=%{customdata[9]:,.0f} ns<br>"
            "95% CI=[%{customdata[10]:,.0f}, %{customdata[11]:,.0f}] ns<br>"
            "ref_median=%{customdata[5]:,.0f} ns<br>"
            "ref_std=%{customdata[12]:,.0f} ns<br>"
            "ref_p95=%{customdata[13]:,.0f} ns<br>"
            "ref_95% CI=[%{customdata[14]:,.0f}, %{customdata[15]:,.0f}] ns<br>"
            "speedup=%{customdata[6]:.2f}x<br>"
            "perf=%{customdata[7]:+.2f}% vs cuBLAS (+ = faster)"
            "<extra></extra>"
        )
        cublas_hovertemplate = (
            "<b>%{fullData.name}</b><br>"
            "N=%{x}<br>"
            "median=%{y:,.0f} ns<br>"
            "arch=%{customdata[0]}<br>"
            "dtype=%{customdata[1]}<br>"
            "class=%{customdata[2]}<br>"
            "desc=%{customdata[3]}<br>"
            "std=%{customdata[8]:,.0f} ns<br>"
            "p95=%{customdata[9]:,.0f} ns<br>"
            "95% CI=[%{customdata[10]:,.0f}, %{customdata[11]:,.0f}] ns<br>"
            "ref_median=%{customdata[5]:,.0f} ns<br>"
            "speedup=%{customdata[6]:.2f}x<br>"
            "perf=— (cuBLAS reference)"
            "<extra></extra>"
        )
        # Per-point 95% CI error bars. Plotly renders a vertical line from
        # median - half to median + half at each point. If the cuBLAS and
        # custom error bars don't overlap at a given N, the difference is
        # statistically significant (ARD §19).
        err_bar = dict(
            type="data",
            array=err_y,
            thickness=2,
            width=4,
            visible=True,
        )
        if is_cublas:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=CUBLAS_COLOR, dash="dash"),
                    marker=dict(color=CUBLAS_COLOR),
                    opacity=0.6,  # semi-transparent so custom lines show through
                    customdata=customdata,
                    hovertemplate=cublas_hovertemplate,
                    error_y=err_bar,
                )
            )
        else:
            color = OKABE_ITO[color_idx % len(OKABE_ITO)]
            color_idx += 1
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color),
                    marker=dict(color=color),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    error_y=err_bar,
                )
            )
            # Significance overlay (2G.6.1): open-circle ring behind each
            # custom point, colored by whether the custom CI and cuBLAS CI
            # overlap. Skip if CI data is missing (old runs).
            sig_colors = []
            for r in rs_sorted:
                sig = _significance(r)
                if sig == "faster":
                    sig_colors.append(SIG_FASTER)
                elif sig == "slower":
                    sig_colors.append(SIG_SLOWER)
                else:
                    sig_colors.append(SIG_INCONCLUSIVE)
            if any(s is not None and r.get("kernel_ci_low_ns") is not None
                   for s, r in zip(sig_colors, rs_sorted)):
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=f"{label} (significance)",
                        marker=dict(
                            symbol="circle-open",
                            size=12,
                            color=sig_colors,
                            line=dict(width=2),
                        ),
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    log_x = log_y = bool(log_log)
    fig.update_layout(**_base_layout(
        title="GEMM timing (lower is better)",
        x_title="N",
        y_title="kernel_median_ns",
        log_x=log_x,
        log_y=log_y,
        height=height,
    ))
    return fig


def _speedup_figure(rows: list[dict],
                    height: int = DEFAULT_CHART_HEIGHT) -> go.Figure:
    """ref_median / kernel_median (speedup ratio) vs N (2G.6.5, ARD §20).

    >1 = custom faster than cuBLAS. Parity line at 1.0. Per-point CI via
    first-order Gaussian propagation. cuBLAS trace excluded (ratio = 1 by
    definition). Log-y default (ratios span 0.1–10×).
    """
    fig = go.Figure()
    custom_rows = [r for r in rows if not _is_cublas(r["kernel_name"])
                   and r.get("kernel_median_ns") and r.get("ref_kernel_median_ns")]
    series: dict[tuple[int, str], list[dict]] = {}
    for r in custom_rows:
        series.setdefault((r["run_id"], r["kernel_name"]), []).append(r)

    color_idx = 0
    for (run_id, kname), rs in sorted(series.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rs_sorted = sorted(rs, key=lambda r: r["n"])
        xs = [r["n"] for r in rs_sorted]
        ys = [r["ref_kernel_median_ns"] / r["kernel_median_ns"] for r in rs_sorted]
        err_y = [_propagated_ci_half(
            r["kernel_median_ns"], r.get("kernel_std_ns") or 0.0,
            r["ref_kernel_median_ns"], r.get("ref_kernel_std_ns") or 0.0)
            for r in rs_sorted]
        customdata = [[r["kernel_median_ns"], r["ref_kernel_median_ns"]] for r in rs_sorted]
        color = OKABE_ITO[color_idx % len(OKABE_ITO)]
        color_idx += 1
        label = f"{kname} (run {run_id})"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=label,
            line=dict(color=color), marker=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "N=%{x}<br>"
                "speedup=%{y:.2f}x<br>"
                "custom_median=%{customdata[0]:,.0f} ns<br>"
                "ref_median=%{customdata[1]:,.0f} ns"
                "<extra></extra>"
            ),
            error_y=dict(type="data", array=err_y, thickness=2, width=4,
                         visible=True),
        ))

    fig.add_hline(y=1.0, line=dict(color=CUBLAS_COLOR, width=1, dash="dash"),
                  annotation_text="parity (1.0x)", annotation_position="top left")
    fig.update_layout(**_base_layout(
        title="Speedup vs cuBLAS (>1 = faster)",
        x_title="N", y_title="ref_median / kernel_median",
        log_x=True, log_y=True, zeroline_y=True, height=height,
    ))
    return fig


def _diff_figure(rows: list[dict], run_a: int, run_b: int,
                 height: int = DEFAULT_CHART_HEIGHT) -> go.Figure:
    """Cross-run diff: median_A / median_B vs N (2G.6.8, ARD §20).

    >1 = run A is slower; <1 = run A is faster. Parity at 1.0. One line per
    kernel name common to both runs. Per-point CI via propagation.
    """
    fig = go.Figure()
    # Group by kernel_name within each run.
    by_run: dict[int, dict[str, list[dict]]] = {run_a: {}, run_b: {}}
    for r in rows:
        if r["run_id"] not in by_run:
            continue
        by_run[r["run_id"]].setdefault(r["kernel_name"], []).append(r)

    common_kernels = set(by_run[run_a].keys()) & set(by_run[run_b].keys())
    color_idx = 0
    for kname in sorted(common_kernels):
        rs_a = sorted(by_run[run_a][kname], key=lambda r: r["n"])
        rs_b = {r["n"]: r for r in by_run[run_b][kname]}
        xs, ys, err = [], [], []
        for r in rs_a:
            n = r["n"]
            if n not in rs_b:
                continue
            rb = rs_b[n]
            med_a = r["kernel_median_ns"]
            med_b = rb["kernel_median_ns"]
            if not med_a or not med_b:
                continue
            xs.append(n)
            ys.append(med_a / med_b)
            err.append(_propagated_ci_half(
                med_a, r.get("kernel_std_ns") or 0.0,
                med_b, rb.get("ref_kernel_std_ns") or 0.0))
        if not xs:
            continue
        color = OKABE_ITO[color_idx % len(OKABE_ITO)]
        color_idx += 1
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=f"{kname} (A={run_a}/B={run_b})",
            line=dict(color=color), marker=dict(color=color),
            error_y=dict(type="data", array=err, thickness=2, width=4,
                         visible=True),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "N=%{x}<br>"
                "ratio A/B=%{y:.3f}<br>"
                "(<1 = A faster, >1 = B faster)"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=1.0, line=dict(color=CUBLAS_COLOR, width=1, dash="dash"),
                  annotation_text="parity (1.0)", annotation_position="top left")
    fig.update_layout(**_base_layout(
        title=f"Cross-run diff: run {run_a} / run {run_b} (<1 = A faster)",
        x_title="N", y_title=f"median_{run_a} / median_{run_b}",
        log_x=True, log_y=True, zeroline_y=True, height=height,
    ))
    return fig



def _accuracy_figure(rows: list[dict], log_log: bool,
                    height: int = DEFAULT_CHART_HEIGHT) -> go.Figure:
    """max_rel_err vs N per kernel, with a tol line per run.

    When all max_rel_err values are 0 (deterministic fill produces
    bit-identical output), log-log is degenerate (log(0) = -inf). We
    clamp the displayed y to a small epsilon for display only — the
    underlying data is not mutated.
    """
    fig = go.Figure()
    # Group by (run_id, kernel_name).
    series: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        series.setdefault((r["run_id"], r["kernel_name"]), []).append(r)

    # Display clamp: if any y is 0, use epsilon for display only.
    _EPS = 1e-15

    color_idx = 0
    for (run_id, kname), rs in series.items():
        rs_sorted = sorted(rs, key=lambda r: r["n"])
        xs = [r["n"] for r in rs_sorted]
        # Clamp for display only; do not mutate the underlying data.
        ys = [max(r["max_rel_err"], _EPS) for r in rs_sorted]
        is_cublas = _is_cublas(kname)
        label = f"{kname} (run {run_id})"
        if is_cublas:
            color = CUBLAS_COLOR
            dash = "dash"
            opacity = 0.6
        else:
            color = OKABE_ITO[color_idx % len(OKABE_ITO)]
            color_idx += 1
            dash = "solid"
            opacity = 1.0
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=label,
                line=dict(color=color, dash=dash),
                marker=dict(color=color),
                opacity=opacity,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "N=%{x}<br>"
                    # 2-decimal scientific (TODO 2B.3.8): %.3e -> %.2e.
                    "max_rel_err=%{y:.2e}<extra></extra>"
                ),
            )
        )

    # Tolerance line(s): one per distinct tol value (TODO 2B.3.4).
    # Previously one add_hline per (run_id, tol) — with 11 runs and 2–3
    # distinct tols, the annotations stacked at the same corner. Now we
    # deduplicate by tol and build a combined label listing the dtypes that
    # share it (e.g. "tol=1e-02 (bf16) / 1e-03 (fp16, tf32)").
    tol_to_dtypes: dict[float, set[str]] = {}
    for r in rows:
        tol = r["tol"]
        if tol is None:
            continue
        tol_to_dtypes.setdefault(float(tol), set()).add(r["dtype"])
    # Stable order: descending tol so the largest (loosest) is annotated first.
    for tol in sorted(tol_to_dtypes, reverse=True):
        dtypes_sorted = sorted(tol_to_dtypes[tol])
        fig.add_hline(
            y=tol,
            line=dict(color="red", width=1, dash="dot"),
            annotation_text=f"tol={tol:g} ({', '.join(dtypes_sorted)})",
            annotation_position="top right",
        )

    fig.update_layout(**_base_layout(
        title="GEMM accuracy (max_rel_err; lower is better)",
        x_title="N",
        y_title="max_rel_err",
        log_x=log_log,
        log_y=log_log,
        grid_weight="light",  # TODO 2B.3.6: tiny y-range, bold grid merges
        height=height,
    ))
    return fig


def _comparison_figure(rows: list[dict],
                      height: int = DEFAULT_CHART_HEIGHT) -> go.Figure:
    """perf_pct vs N, one line per (run, custom kernel). cuBLAS rows are
    excluded (perf_pct is None for them). Horizontal parity line at 0.

    Default linear y-axis — the percentage is the point; log scale
    obscures it (ARD §15, TODO 2B.2.3).
    """
    fig = go.Figure()
    # Only custom rows have a defined perf_pct.
    custom_rows = [r for r in rows if _perf_pct(r) is not None]
    series: dict[tuple[int, str], list[dict]] = {}
    for r in custom_rows:
        series.setdefault((r["run_id"], r["kernel_name"]), []).append(r)

    color_idx = 0
    for (run_id, kname), rs in sorted(series.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rs_sorted = sorted(rs, key=lambda r: r["n"])
        xs = [r["n"] for r in rs_sorted]
        ys = [_perf_pct(r) for r in rs_sorted]
        customdata = [
            [
                r["arch"],
                r["dtype"],
                r["kernel_desc"],
                r["kernel_median_ns"],
                r["ref_kernel_median_ns"],
            ]
            for r in rs_sorted
        ]
        color = OKABE_ITO[color_idx % len(OKABE_ITO)]
        color_idx += 1
        label = f"{kname} (run {run_id})"
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=label,
                line=dict(color=color),
                marker=dict(color=color),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "N=%{x}<br>"
                    # 2-decimal perf (TODO 2B.3.8): %+.1f -> %+.2f.
                    "perf=%{y:+.2f}% vs cuBLAS (+ = faster)<br>"
                    "arch=%{customdata[0]}<br>"
                    "dtype=%{customdata[1]}<br>"
                    "desc=%{customdata[2]}<br>"
                    "median=%{customdata[3]:,.0f} ns<br>"
                    "ref_median=%{customdata[4]:,.0f} ns"
                    "<extra></extra>"
                ),
            )
        )

    # Parity line at 0: above = beating cuBLAS, below = slower.
    fig.add_hline(
        y=0,
        line=dict(color=CUBLAS_COLOR, width=1, dash="dash"),
        annotation_text="parity (0%)",
        annotation_position="top left",
    )

    fig.update_layout(**_base_layout(
        title="% perf vs cuBLAS (+ = faster; above parity = winning)",
        x_title="N",
        y_title="% vs cuBLAS (+ = faster)",
        log_x=True,   # N spans 32..4096; log x keeps small-N visible
        log_y=False,  # linear y — the percentage is the point (ARD §15)
        zeroline_y=True,  # parity line at 0 should be visually distinct
        height=height,
    ))
    return fig


def _runs_table() -> list[dict]:
    """Run history rows for the Dash table."""
    conn = db.connect()
    try:
        runs = db.list_runs(conn)
        # Per-run best custom perf_pct at N=4096 (largest common sweep size).
        # ARD §15 / TODO 2B.2.4: single-number summary per run.
        perf_by_run: dict[int, float | None] = {}
        for r in runs:
            perf_by_run[r["id"]] = db.best_custom_perf_pct_at_n(
                conn, r["id"], 4096
            )
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "ingested_at": r["ingested_at"],
            "git_sha": r["git_sha"] or "",
            "label": r["label"] or "",
            "arch": r["arch"],
            "dtype": r["dtype"],
            "kernel_count": r["kernel_count"],
            "median % vs cuBLAS @ N=4096": _format_perf_pct(
                perf_by_run.get(r["id"])
            ),
        }
        for r in runs
    ]


def _format_perf_pct(val: float | None) -> str:
    """Format perf_pct for the Run History table. Empty string when None
    (no custom kernel at N=4096, or cuBLAS missing/zero at N=4096)."""
    if val is None:
        return ""
    return f"{val:+.1f}%"


def build_app() -> dash.Dash:
    app = dash.Dash(__name__)
    app.title = "gemm_y dashboard"

    conn = db.connect()
    try:
        archs = db.distinct(conn, "arch")
        dtypes = db.distinct(conn, "dtype")
        runs = db.list_runs(conn)
    finally:
        conn.close()
    run_options = [
        {"label": f"#{r['id']} {r['arch']}/{r['dtype']}"
                  + (f" [{r['label']}]" if r["label"] else ""),
         "value": r["id"]}
        for r in runs
    ]

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "row", "gap": "16px",
               "padding": "16px", "fontFamily": "sans-serif",
               "fontSize": 15},  # TODO 2B.3.3: explicit sidebar/page font size
        children=[
            # Sidebar
            html.Div(
                id="sidebar",
                style={"width": "260px", "flexShrink": "0",
                       "borderRight": "1px solid #ccc", "paddingRight": "16px"},
                children=[
                    html.H3("gemm_y"),
                    html.Div([
                        html.Label("Arch", style={"fontSize": 14}),
                        dcc.RadioItems(
                            id="filter-arch",
                            options=[{"label": a, "value": a} for a in archs],
                            value=archs[0] if archs else None,
                            labelStyle={"display": "block"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    html.Div([
                        html.Label("Dtype", style={"fontSize": 14}),
                        dcc.Checklist(
                            id="filter-dtype",
                            options=[{"label": d, "value": d} for d in dtypes],
                            value=list(dtypes),
                            labelStyle={"display": "block"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    html.Div([
                        html.Label("Kernel class", style={"fontSize": 14}),
                        dcc.Checklist(
                            id="filter-class",
                            options=[
                                {"label": "Custom", "value": "custom"},
                                {"label": "cuBLAS", "value": "cublas"},
                            ],
                            value=["custom", "cublas"],
                            labelStyle={"display": "block"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    html.Div([
                        html.Label("Runs", style={"fontSize": 14}),
                        dcc.Dropdown(
                            id="filter-runs",
                            options=run_options,
                            value=[r["value"] for r in run_options],
                            multi=True,
                        ),
                    ], style={"marginBottom": "12px"}),
                    # Second run selector for the Diff tab (2G.6.8).
                    html.Div([
                        html.Label("Diff run B", style={"fontSize": 14}),
                        dcc.Dropdown(
                            id="filter-run-b",
                            options=run_options,
                            value=run_options[-1]["value"] if run_options else None,
                            multi=False,
                        ),
                    ], style={"marginBottom": "12px"}),
                    html.Div([
                        html.Label("Scale", style={"fontSize": 14}),
                        dcc.RadioItems(
                            id="filter-scale",
                            options=[
                                {"label": "log-log", "value": "log"},
                                {"label": "linear", "value": "linear"},
                            ],
                            value="log",
                            labelStyle={"display": "block"},
                        ),
                    ], style={"marginBottom": "12px"}),
                    # Chart height control (TODO 2B.3.7): runtime control over
                    # figure height without a server restart. Default bumped
                    # from 520 to 640 — the user said the chart felt too short.
                    html.Div([
                        html.Label("Chart height", style={"fontSize": 14}),
                        dcc.RadioItems(
                            id="filter-chart-height",
                            options=[
                                {"label": "S (640)", "value": 640},
                                {"label": "M (760)", "value": 760},
                                {"label": "L (900)", "value": 900},
                                {"label": "XL (1160)", "value": 1160},
                            ],
                            value=DEFAULT_CHART_HEIGHT,
                            labelStyle={"display": "block"},
                        ),
                    ], style={"marginBottom": "12px"}),
                ],
            ),
            # Main content
            html.Div(
                style={"flex": "1", "minWidth": "0"},
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="timing",
                        children=[
                            dcc.Tab(label="Timing", value="timing"),
                            dcc.Tab(label="Comparison", value="comparison"),
                            dcc.Tab(label="Accuracy", value="accuracy"),
                            dcc.Tab(label="Speedup", value="speedup"),
                            dcc.Tab(label="Diff", value="diff"),
                            dcc.Tab(label="Run History", value="runs"),
                        ],
                    ),
                    html.Div(id="tab-content", style={"marginTop": "16px"}),
                ],
            ),
        ],
    )

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("filter-arch", "value"),
        Input("filter-dtype", "value"),
        Input("filter-class", "value"),
        Input("filter-runs", "value"),
        Input("filter-scale", "value"),
        Input("filter-chart-height", "value"),
        Input("filter-run-b", "value"),
    )
    def render_tab(tab, arch, dtypes_sel, classes, run_ids, scale,
                   chart_height, run_b):
        if tab == "runs":
            rows = _runs_table()
            return dt.DataTable(
                columns=[
                    {"name": c, "id": c}
                    for c in ["id", "ingested_at", "git_sha", "label",
                              "arch", "dtype", "kernel_count",
                              "median % vs cuBLAS @ N=4096"]
                ],
                data=rows,
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "monospace", "fontSize": 12,
                            "padding": "4px 8px"},
                style_header={"fontWeight": "bold"},
            )

        conn = db.connect()
        try:
            archs_sel = [arch] if arch else None
            rows = db.fetch_measurements(
                conn,
                run_ids=run_ids or None,
                archs=archs_sel,
                dtypes=dtypes_sel or None,
                kernel_classes=classes or None,
            )
        finally:
            conn.close()
        log_log = scale == "log"
        # Chart height is runtime-configurable via the sidebar (TODO 2B.3.7).
        # Fall back to the default if the control is missing/None.
        height = chart_height if chart_height else DEFAULT_CHART_HEIGHT
        if tab == "timing":
            return dcc.Graph(figure=_timing_figure(rows, log_log, height=height))
        if tab == "comparison":
            # Comparison view is always linear-y (log obscures the %).
            # Sidebar filters (arch/dtype/runs/class) still apply.
            return dcc.Graph(figure=_comparison_figure(rows, height=height))
        if tab == "accuracy":
            return dcc.Graph(figure=_accuracy_figure(rows, log_log, height=height))
        if tab == "speedup":
            return dcc.Graph(figure=_speedup_figure(rows, height=height))
        if tab == "diff":
            # Diff uses run A = first selected run, run B = the sidebar's
            # Diff-run-B selector.
            run_a = (run_ids or [None])[0]
            if not run_a or not run_b:
                return html.Div("Select at least one run in 'Runs' and a run in 'Diff run B'.")
            return dcc.Graph(figure=_diff_figure(rows, run_a, run_b, height=height))
        return html.Div("unknown tab")

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = build_app()
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
