#!/usr/bin/env python3
"""Dash dashboard for gemm_y benchmark results.

Served at http://localhost:8050. Reads from db/gemm_y.db (the source of
truth); never from CSV. Run `ingest.py` first to populate the DB.

Layout: single page, sidebar + four tabs (Timing / Comparison / Accuracy /
Run History). Sidebar filters: arch, dtype, kernel class (Custom/cuBLAS),
runs multi-select, scale (log-log / linear). The Comparison tab plots
`perf_pct` vs N (ARD §15) with a parity line at 0; it ignores the scale
toggle (always linear-y, log-x).

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

# The 14-size square sweep (powers of 2 + midpoints). Used as log-x tickvals
# so every data point has a tick (AGENTS.md Benchmarking Protocol).
SWEEP_SIZES = [32, 64, 96, 128, 192, 256, 384, 512, 768,
               1024, 1536, 2048, 3072, 4096]

# Shared visual theme for all figures (TODO 2B.3.2 / 2B.3.3 / 2B.3.6 / 2B.3.7).
PLOT_BG_COLOR = "#f0f0f0"   # light gray plot area — visible boundary vs white page
PAPER_BG_COLOR = "#ffffff"  # white margin / page
BASE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12

# Grid weight presets (TODO 2B.3.6). The accuracy chart's y-range is tiny
# (1e-15 to ~1e-3); gridwidth=2 merges into a solid block, so it uses the
# "light" preset. Timing / Comparison use "bold" — wide y-range, bold reads fine.
_GRID_BOLD = {"gridwidth": 2, "gridcolor": "rgba(0,0,0,0.35)", "tickwidth": 2}
_GRID_LIGHT = {"gridwidth": 1, "gridcolor": "rgba(0,0,0,0.15)", "tickwidth": 1}
_GRID_PRESETS = {"bold": _GRID_BOLD, "light": _GRID_LIGHT}

# Chart height options (TODO 2B.3.7). Default bumped from 520 to 640 — the
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


def _timing_figure(rows: list[dict], log_log: bool,
                  height: int = DEFAULT_CHART_HEIGHT) -> go.Figure:
    """kernel_median_ns vs N, one line per (run, kernel)."""
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
            ]
            for r in rs_sorted
        ]
        is_cublas = _is_cublas(kname)
        # Label includes run_id so multiple ingests of the same kernel
        # are distinguishable.
        label = f"{kname} (run {run_id})"
        # Shared hovertemplate — thousands separator on ns values.
        # customdata indices (TODO 2B.3.1 — fixed from [4]/[5] to [5]/[6]):
        #   [0] arch   [1] dtype   [2] class   [3] kernel_desc
        #   [4] kernel_median_ns   [5] ref_kernel_median_ns
        #   [6] speedup            [7] perf_pct
        # Hover fractions rounded to a 2-decimal ceiling (TODO 2B.3.8):
        # speedup %.3f -> %.2f, perf_pct %+.1f -> %+.2f. Integer ns values
        # stay integer (no fraction to round).
        # perf_pct is None for cuBLAS rows; the %{customdata[7]:+.2f} format
        # renders 'nan' for None, so we use a conditional via a separate
        # cuBLAS hovertemplate below.
        hovertemplate = (
            "<b>%{fullData.name}</b><br>"
            "N=%{x}<br>"
            "median=%{y:,.0f} ns<br>"
            "arch=%{customdata[0]}<br>"
            "dtype=%{customdata[1]}<br>"
            "class=%{customdata[2]}<br>"
            "desc=%{customdata[3]}<br>"
            "ref_median=%{customdata[5]:,.0f} ns<br>"
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
            "ref_median=%{customdata[5]:,.0f} ns<br>"
            "speedup=%{customdata[6]:.2f}x<br>"
            "perf=— (cuBLAS reference)"
            "<extra></extra>"
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
                                {"label": "S (520)", "value": 520},
                                {"label": "M (640)", "value": 640},
                                {"label": "L (760)", "value": 760},
                                {"label": "XL (900)", "value": 900},
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
    )
    def render_tab(tab, arch, dtypes_sel, classes, run_ids, scale,
                   chart_height):
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
