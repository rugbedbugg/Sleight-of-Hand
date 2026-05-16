"""Shared matplotlib styling so every figure in the report reads as one
system: a fixed categorical color order, a single-hue sequential ramp for
magnitude (the win-rate heatmap), and muted chart chrome. Palette values
per the project's dataviz color formula (light mode, static report
figures)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; scripts save figures to disk, they don't show()

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]  # red -> neutral -> blue


def sequential_cmap():
    return LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)


def diverging_cmap():
    return LinearSegmentedColormap.from_list("div_red_blue", DIVERGING)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def new_figure(figsize=(7, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    style_axes(ax)
    return fig, ax
