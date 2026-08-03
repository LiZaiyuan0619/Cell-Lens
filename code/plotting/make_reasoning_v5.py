#!/usr/bin/env python3
"""Reasoning robustness v5: a zero-baseline line chart using four-model data.

Outputs
-------
``reasoning_v5_zero_baseline_lines`` is a zero-anchored line chart designed to
test the main claim directly: the gain is positive at every effort. The
bubble-matrix comparison function is retained but disabled by default.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nmi_style import NMI, apply_style  # noqa: E402


HERE = Path(__file__).parent

# ========================= CONFIG（常用微调都在这里）=========================
# ---- 1. 数据与导出 ----
SOURCE_DATA = HERE / "reasoning_v5_source_data.csv"
EFFORTS = ("None", "Low", "Medium", "High")
MODEL_ORDER = ("GPT-5.4 Mini", "GPT-5.5", "Gemini 3.5 Flash", "Claude Opus 4.8")
OUT_BUBBLE = HERE / "reasoning_v5_bubble_matrix"
OUT_LINES = HERE / "reasoning_v5_zero_baseline_lines"
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
PNG_DPI = 600
# Set True only when regenerating the earlier bubble-matrix comparison.
DRAW_COMPARISON_BUBBLE = False

# ---- 2. 最终折线图：常用微调参数（优先改这里）----
# Final single-column page size (inches).  It matches the manuscript's
# \linewidth and preserves this figure's previous final placed height.
LINE_FIG_SIZE = (3.3125, 3.319)
# 全局字体、轴线与标题字号。
FONT_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
BASE_FONT_SIZE = 12
AXIS_LINE_WIDTH = 0.9
TITLE_SIZE = 12
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 12
END_LABEL_SIZE = 12
# 标题、坐标轴与数据范围。
LINE_TITLE = "Ours helps at every effort setting"
LINE_Y_LIM = (-2.0, 26.5)
LINE_Y_TICKS = (0, 5, 10, 15, 20, 25)
# Reserve the right-hand portion of the axes for direct model labels.  Keeping
# that area inside xlim makes the positive band and zero baseline cover the
# labels rather than ending at the High data point.
LINE_X_LIM = (-0.12, 4.35)
# 线条、点和右侧模型直标。
LINE_WIDTH = 3
MARKER_SIZE = 8
END_LABEL_OFFSETS = {
    "GPT-5.4 Mini": (6, -12),
    "GPT-5.5": (6, 3),
    "Gemini 3.5 Flash": (6, -3),
    "Claude Opus 4.8": (6, 12),
}
# 正向区域：保持淡绿色纯色，不用深浅渐变，以免与 y 轴的增益编码重复。
POSITIVE_BAND_COLOR = "#EAF5EC"
POSITIVE_BAND_LABEL_X = 4.10
POSITIVE_BAND_LABEL_Y = 0.62
POSITIVE_BAND_LABEL_SIZE = 12
ZERO_LINE_COLOR = NMI["neutral_dark"]
GRID = "#E8EEE8"
INK = NMI["ink"]

# ---- 3. 方案 A：更新的 v4 气泡矩阵 ----
BUBBLE_FIG_SIZE = (4.70, 4.10)
BUBBLE_TITLE = "Ours improves at every reasoning-effort"
BUBBLE_DOT_COLOR = "#77B98B"
BUBBLE_DOT_EDGE = "#4F9670"
BUBBLE_DOT_SIZE_MIN = 165
BUBBLE_DOT_SIZE_MAX = 700
BUBBLE_DOT_SIZE_POWER = 0.62
BUBBLE_VALUE_SIZE = 8.0
MODEL_LABEL_SIZE = 7.7

# ---- 4. 模型身份（折线图）----
# Muted model-identity colors; green is reserved for the positive-effect band.
MODEL_STYLES = {
    "GPT-5.4 Mini": {"color": "#5666A7", "marker": "o", "label": "GPT-5.4\nMini"},
    "GPT-5.5": {"color": "#8A6FB0", "marker": "s", "label": "GPT-5.5"},
    "Gemini 3.5 Flash": {"color": "#C86AA0", "marker": "^", "label": "Gemini\n3.5 Flash"},
    "Claude Opus 4.8": {"color": "#A87545", "marker": "D", "label": "Claude\nOpus 4.8"},
}
# ============================================================================


def apply_paper_font() -> None:
    """Set a Times-compatible family while retaining editable vector text."""
    for font_path in fm.findSystemFonts():
        filename = os.path.basename(font_path).lower()
        if "nimbusrom" in filename or "nimbus_roman" in filename or filename.startswith("times"):
            try:
                fm.fontManager.addfont(font_path)
            except RuntimeError:
                pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def load_data() -> tuple[list[str], np.ndarray]:
    """Load gains in the declared model and effort order."""
    with SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_model = {row["model"]: row for row in rows}
    if set(by_model) != set(MODEL_ORDER):
        raise ValueError("Source-data model names do not match MODEL_ORDER")
    gains = np.array([[float(by_model[model][effort.lower()]) for effort in EFFORTS]
                      for model in MODEL_ORDER])
    if not np.all(gains > 0):
        raise ValueError("This figure is defined for strictly positive gains only")
    return list(MODEL_ORDER), gains


def scaled_dot_sizes(values: np.ndarray) -> np.ndarray:
    """Map gains to discriminable bubble areas without exaggerating extremes."""
    low, high = float(values.min()), float(values.max())
    normalized = (values - low) / (high - low)
    return BUBBLE_DOT_SIZE_MIN + (BUBBLE_DOT_SIZE_MAX - BUBBLE_DOT_SIZE_MIN) * normalized**BUBBLE_DOT_SIZE_POWER


def save_figure(fig: plt.Figure, out_base: Path) -> list[str]:
    outputs = []
    for extension in EXPORT_FORMATS:
        output = out_base.with_suffix(f".{extension}")
        # The final PDF page must equal LINE_FIG_SIZE exactly.  Tight cropping
        # silently changes that page size and causes LaTeX to rescale the figure.
        fig.savefig(output, dpi=PNG_DPI)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def draw_bubble_matrix(models: list[str], gains: np.ndarray) -> list[str]:
    """Version A: retain the v4 visual language, now with all four models."""
    figure, ax = plt.subplots(figsize=BUBBLE_FIG_SIZE)
    rows = np.arange(len(models))[::-1]
    columns = np.arange(len(EFFORTS))
    x_values, y_values = np.meshgrid(columns, rows)
    ax.scatter(x_values.ravel(), y_values.ravel(), s=scaled_dot_sizes(gains.ravel()),
               color=BUBBLE_DOT_COLOR, edgecolor=BUBBLE_DOT_EDGE, linewidth=0.9, alpha=0.92, zorder=3)
    for row_index, y in enumerate(rows):
        for column_index, x in enumerate(columns):
            ax.annotate(f"+{gains[row_index, column_index]:.1f}", (x, y), xytext=(9, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=BUBBLE_VALUE_SIZE,
                        color=INK, zorder=4)
    ax.set_xticks(columns, EFFORTS, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Reasoning effort", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_yticks(rows, [model.replace(" ", "\n", 1) for model in models], fontsize=MODEL_LABEL_SIZE)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_xlim(-0.38, 3.58)
    ax.set_ylim(-0.48, len(models) - 0.52)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(BUBBLE_TITLE, loc="center", fontsize=TITLE_SIZE, fontweight="bold", color=INK, pad=8)
    figure.tight_layout(pad=0.45)
    return save_figure(figure, OUT_BUBBLE)


def draw_zero_baseline_lines(models: list[str], gains: np.ndarray) -> list[str]:
    """Version B: make the all-positive-at-every-effort claim directly testable."""
    figure, ax = plt.subplots(figsize=LINE_FIG_SIZE)
    x = np.arange(len(EFFORTS))
    ax.axhspan(0, LINE_Y_LIM[1], color=POSITIVE_BAND_COLOR, zorder=0)
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.05, zorder=2)
    ax.text(POSITIVE_BAND_LABEL_X, POSITIVE_BAND_LABEL_Y, "Ours helps", ha="right", va="bottom",
            fontsize=POSITIVE_BAND_LABEL_SIZE, color="#4F9670", fontweight="bold")
    for model, values in zip(models, gains):
        style = MODEL_STYLES[model]
        ax.plot(x, values, color=style["color"], marker=style["marker"], markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        dx, dy = END_LABEL_OFFSETS[model]
        ax.annotate(style["label"], (x[-1], values[-1]), xytext=(dx, dy), textcoords="offset points",
                    ha="left", va="center", fontsize=END_LABEL_SIZE, color=style["color"], fontweight="bold")
    ax.set_xlim(*LINE_X_LIM)
    ax.set_ylim(*LINE_Y_LIM)
    ax.set_xticks(x, EFFORTS, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Reasoning effort", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_yticks(LINE_Y_TICKS)
    ax.set_ylabel("Ours gain", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(LINE_TITLE, loc="center", x=0.45, fontsize=TITLE_SIZE,
                 fontweight="bold", color=INK, pad=4)
    figure.tight_layout(pad=0.45)
    return save_figure(figure, OUT_LINES)


def main() -> None:
    apply_style(font_size=BASE_FONT_SIZE, axes_linewidth=AXIS_LINE_WIDTH)
    apply_paper_font()
    models, gains = load_data()
    outputs = draw_zero_baseline_lines(models, gains)
    if DRAW_COMPARISON_BUBBLE:
        outputs = draw_bubble_matrix(models, gains) + outputs
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
