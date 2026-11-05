"""manuscript_style.py — house figure style for the ARLP / manuscript figures.

Drop-in, self-contained matplotlib styling so every figure matches the
"Representative load shapes" reference figure: strict blue/grey palette, no
embedded figure name, enlarged fonts, bold markerless lines, and a single
shared legend at the bottom.

Usage
-----
    import manuscript_style as ms

    ms.apply_house_rcparams()
    fig, axes = plt.subplots(4, 3, figsize=(22, 24))
    for ax, (x, y, title) in zip(axes.ravel(), panels):
        ax.plot(x, y, **ms.PRIMARY_LINE)
        ms.style_axes(ax, xlabel="hour-ending",
                      ylabel="peak normalized load", title=title)
    ms.bottom_legend(fig, handles)          # one shared legend, no per-panel legends
    ms.save(fig, out_path)

The module has no project dependencies (only numpy + matplotlib), so it can be
copied next to any figure script.
"""
from __future__ import annotations

import numpy as np


# ── Palette (the ONLY colours permitted — blue/grey table swatches) ─────────
NAVY = "#1A2C3C"           # dark navy      — primary data curve, hatch lines
SLATE = "#676C80"          # slate-gray     — dash-dot reference line, band edges
BLUEGRAY = "#5C7999"       # blue-gray      — dashed reference line
BODY = "#44586C"           # dark blue body — secondary data curve
LIGHT_BLUEGRAY = "#D2D7DE"  # light blue-gray — primary (darker) fill band
LAVENDER = "#F5F2F6"       # very light lavender-gray — neutral (lighter) fill band
WHITE = "#FFFFFF"

PALETTE = {
    "navy": NAVY, "slate": SLATE, "blue_gray": BLUEGRAY, "body": BODY,
    "light_blue_gray": LIGHT_BLUEGRAY, "lavender": LAVENDER, "white": WHITE,
}

# Ordered cycle for multi-series charts: blues first, then greys. Never adds a
# colour outside the palette — use hatching (not a new hue) for emphasis.
CATEGORICAL = [NAVY, BLUEGRAY, BODY, SLATE, LIGHT_BLUEGRAY]


# ── Enlarged type scale ─────────────────────────────────────────────────────
FS_TITLE = 17     # panel subtitle (short identifier only — never a figure name)
FS_LABEL = 15     # axis names
FS_TICK = 12      # tick labels
FS_LEGEND = 15    # shared bottom legend

# ── Line / fill style kwargs — spread into plot()/bar()/Patch() ─────────────
# No markers anywhere: "just lines".
PRIMARY_LINE = dict(color=NAVY, lw=2.7, solid_capstyle="round")
SECONDARY_LINE = dict(color=BODY, lw=2.4, solid_capstyle="round")
REF_LINE_DASHED = dict(color=BLUEGRAY, lw=1.9, ls="--")
REF_LINE_DASHDOT = dict(color=SLATE, lw=1.9, ls="-.")

# Filled bands (e.g. TOU zone means). Darker band = LIGHT_BLUEGRAY, neutral =
# LAVENDER. A highlighted band uses diagonal hatching instead of a new colour.
FILL_PRIMARY = dict(color=LIGHT_BLUEGRAY, edgecolor=SLATE, linewidth=0.5)
FILL_NEUTRAL = dict(color=LAVENDER, edgecolor=SLATE, linewidth=0.5)
HIGHLIGHT_HATCH = dict(color=LAVENDER, edgecolor=NAVY, hatch="////", linewidth=0.6)

_HATCH_LW = 1.3   # matplotlib hatch stroke weight


def apply_house_rcparams():
    """Set global rcParams for the house style. Call once before plotting."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        # House font: Arial for all chart text (axis labels, values, titles,
        # legend); falls back to DejaVu Sans if Arial is unavailable.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "hatch.linewidth": _HATCH_LW,
        "savefig.dpi": 130,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def style_axes(ax, *, xlabel=None, ylabel=None, title=None, normalized=True):
    """Apply the house per-axes conventions.

    normalized=True clamps the y-axis to 0..1 (0..1.03 headroom, ticks 0..1.0)
    so nothing renders above 1. Set normalized=False for absolute-unit axes
    (e.g. a $/kWh secondary axis), which keep their own real range.
    """
    if normalized:
        ax.set_ylim(0.0, 1.03)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    ax.grid(True, axis="y", alpha=0.30)


def bottom_legend(fig, handles, *, ncol=4, reserve=0.05):
    """Reserve a bottom band and drop ONE shared legend into it.

    Removes the need for per-panel legends. Labels on ``handles`` must be
    generic (no per-panel numeric values). ``reserve`` is the fraction of
    figure height held for the legend by tight_layout.
    """
    fig.tight_layout(rect=(0, reserve, 1, 1.0))
    return fig.legend(handles=handles, loc="lower center", ncol=ncol,
                      fontsize=FS_LEGEND, frameon=False,
                      bbox_to_anchor=(0.5, reserve * 0.15))


def save(fig, out_path, *, dpi=130):
    """Save with the house defaults (dpi 130, tight bbox) and close the figure."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def contiguous_spans(kind_by_position):
    """Collapse a per-position list of category tokens into contiguous runs.

    Returns a list of (start, stop, kind) where a band spans positions
    [start, stop). Useful for drawing zone bands as one rectangle per run
    (no internal seams). ``kind_by_position[i]`` is the token for unit interval
    [i, i+1).
    """
    runs, x0 = [], 0
    cur = kind_by_position[0]
    for i in range(1, len(kind_by_position)):
        if kind_by_position[i] != cur:
            runs.append((x0, i, cur))
            x0, cur = i, kind_by_position[i]
    runs.append((x0, len(kind_by_position), cur))
    return runs
