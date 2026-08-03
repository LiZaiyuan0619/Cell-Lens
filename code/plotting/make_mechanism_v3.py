#!/usr/bin/env python3
"""Mechanism figure v3: missing evidence → marker-level gain → program-level control."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nmi_style import NMI, apply_style  # noqa: E402


HERE = Path(__file__).parent

# ==================== 高频微调区（改图时优先只改此处）====================
# 说明：所有距离以 pt（points）计，除非注释另写；坐标范围使用数据单位。

# ---- 1. 总图尺寸与三栏布局 ----
# 双栏最终页面尺寸（英寸）；不要改动宽度 7.0。高度与论文现有占位一致。
FIG_SIZE = (7.0, 2.91)
# 总图使用显式坐标，避免自动布局因 y 轴文字长度而错位。
# 每项为 (left, bottom, width, height)，均为相对整张图的 0--1 坐标。
# 调整 width 可直接扩/缩对应子图；调整 left 可整体左右移动子图。
COMBINED_PLOT_POSITIONS = {
    "a": (0.075, 0.235, 0.240, 0.655),
    "b": (0.390, 0.235, 0.250, 0.655),
    "c": (0.735, 0.235, 0.230, 0.655),
}
# 每个子图下方图例区的位置；与上方子图独立，避免图例反过来压缩绘图区。
COMBINED_LEGEND_POSITIONS = {
    "a": (0.000, 0.010, 0.370, 0.115),
    "b": (0.350, 0.010, 0.310, 0.115),
    "c": (0.650, 0.010, 0.350, 0.115),
}
# 仅在总图中绘制 a/b/c；三个编号同一 y 坐标，x 与各自子图最左文字边界对齐。
COMBINED_PANEL_TAG_POSITIONS = {
    "a": (0.005, 0.992), "b": (0.327, 0.992), "c": (0.655, 0.992),
}

# ---- 2. 子图编号、标题、通用字体与轴线 ----
# 标题从绘图区左端开始；总图编号由 COMBINED_PANEL_TAG_POSITIONS 单独控制。
PANEL_TITLE_LOC = "left"
PANEL_TITLE_Y = 1.030                # 标题垂直位置；调小可使标题下移
# 标题相对各绘图区左端的水平位置；a 左移以视觉居中包含 y 轴文字的整张子图。
PANEL_TITLE_X_BY_PANEL = {"a": -0.15, "b": 0.0, "c": 0.0}
PANEL_LABEL_SIZE = 12                # 总图 a/b/c 编号字号
BASE_FONT_SIZE = 10                  # Matplotlib 基准字号
AXIS_LINE_WIDTH = 0.8
AXIS_LABEL_PAD = 2.0                 # 轴标题与轴线距离
TICK_LABEL_PAD = 2.0                 # 刻度文字与轴线距离
GRID_LINE_WIDTH = 0.65
GRID_COLOR = "#E9E9E9"

# ---- 3. a：marker 可见性柱图 ----
MARKER_BAR_WIDTH = 0.36              # 每根柱的宽度（数据坐标）
MARKER_YLIM = (0, 112)
MARKER_YTICKS = (0, 25, 50, 75, 100)
MARKER_XTICK_SIZE = 9.5              # marker / 细胞类型标签字号
MARKER_YTICK_SIZE = 10
MARKER_YLABEL_SIZE = 10
MARKER_VALUE_FONT_SIZE = 10
MARKER_VALUE_Y_OFFSET = 2.0          # 柱顶数字相对柱顶的上移量
A_TITLE = "Protein carries missing markers"
A_TITLE_SIZE = 10

# ---- 4. b：散点图的大小、坐标与文字位置 ----
SCATTER_XLIM = (-6, 90)
SCATTER_YLIM = (-22, 77)
SCATTER_XTICKS = (0, 20, 40, 60, 80)
SCATTER_YTICKS = (-20, 0, 20, 40, 60)
BURIAL_ZONE_RIGHT = 15               # x < 15% 的淡紫 "buried" 区域右边界
SCATTER_SIZE = 120                   # 散点面积（pt²）
SCATTER_EDGE_WIDTH = 0.85
SCATTER_XTICK_SIZE = 10
SCATTER_XLABEL_SIZE = 10.5
SCATTER_YTICK_SIZE = 10
SCATTER_YLABEL_SIZE = 10.5
B_POINT_LABEL_SIZE = 10.0
ZERO_LINE_WIDTH = 0.9
# 点标签偏移：(向右为正, 向上为正)，单位 pt。单独改某一行即可避开重叠。
CELL_LABEL_OFFSETS = {
    "Naive CD8 T": (5, -10), "Eff CD8 T": (5, 5), "Eff CD4 T": (5, -13),
    "Naive CD4 T": (5, 0), "NK": (-9, 8), "B": (9, -8), "Mono": (5, 0),
}
B_TITLE = "Gains on buried markers"
B_TITLE_SIZE = 10

# ---- 5. c：dumbbell 图的范围、大小与数值标签 ----
PROGRAM_XLIM = (-18.0, 16.0)         # 左右各留数据边距，平衡负值与 +10.0 标签空间
PROGRAM_XTICKS = (-15, -10, -5, 0, 5, 10, 15)
PROGRAM_MODEL_TICK_SIZE = 9.5
PROGRAM_XTICK_SIZE = 10
PROGRAM_XLABEL_SIZE = 10.5
PROGRAM_VALUE_FONT_SIZE = 10
PROGRAM_DOT_SIZE = 135               # 圆点面积（pt²）
PROGRAM_LINE_WIDTH = 3.0
PROGRAM_ZERO_LINE_WIDTH = 1.2
PROGRAM_VALUE_X_OFFSET = 5           # 正/负数值标签离圆点的水平距离（pt）
# 指定某个数值标签的额外偏移：(水平, 垂直)，单位 pt；无需要可保持不动。
PROGRAM_VALUE_OFFSETS = {
    ("GPT 5.6 Sol", "readable"): (2, 8),
    ("Gemini 3.5 Flash", "readable"): (15, 15),
    ("Claude Opus 4.8", "readable"): (2, 8),  # -10.6 右上移，避开圆点和模型标签
}
C_TITLE = "Buried programs yield gains"
C_TITLE_SIZE = 10

# ---- 6. 图例尺寸与间距 ----
COMBINED_LEGEND_NCOL = 2             # 每个 panel 下方的图例列数
COMBINED_LEGEND_FONT_SIZE = 9.5
LEGEND_HANDLE_LENGTH = 1.15
LEGEND_HANDLE_TEXT_PAD = 0.35
LEGEND_COLUMN_SPACING = 0.90
LEGEND_MARKER_SIZE = 8
PROGRAM_LEGEND_MARKER_SIZE = 8.5

# ---- 7. 单 panel 审阅图（不影响主文总图）----
SINGLE_FIG_SIZE_A = (3.45, 3.25)
SINGLE_FIG_SIZE_B = (3.60, 3.35)
SINGLE_FIG_SIZE_C = (4.0, 4.0)
SINGLE_HEIGHT_RATIOS = (3.0, 0.32)
SINGLE_HSPACE = 0.08
SINGLE_LEGEND_FONT_SIZE = 10

# ---- 8. 固定输出设置 ----
OUT_A = HERE / "mechanism_v3_a"
OUT_B = HERE / "mechanism_v3_b"
OUT_C = HERE / "mechanism_v3_c"
OUT_COMBINED = HERE / "mechanism_v3"
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
PNG_DPI = 600

# ======================== 数据与配色（通常无需修改）========================
MARKERS = [
    ("CD8A\n(CD8 T)", 7, 98),
    ("MS4A1\n(B)", 60, 100),
    ("NKG7\n(NK)", 81, 95),
    ("CD14\n(Mono)", 36, 100),
]
CELL_TYPES = [  # (细胞类型, marker 在文本中的可见率%, Ours 增益 pts)
    ("Naive CD8 T", 7, 71.6), ("Eff CD8 T", 9, 17.8), ("Eff CD4 T", 0, 52.1),
    ("Naive CD4 T", 0, 7.7), ("NK", 81, 13.5), ("B", 60, -2.5), ("Mono", 36, -15.8),
]
MODELS = ["GPT-5.4 Mini", "Qwen 3.7 Plus", "DeepSeek V4 Flash", "Gemini 3.5 Flash", "GPT-5.6 Sol", "Claude Opus 4.8"]
BURIED_PROGRAM_GAIN = [1.7, 6.0, 9.2, 10.0, 5.9, 6.7]
READABLE_PROGRAM_GAIN = [-5.3, -1.6, -2.3, -12.6, -9.0, -10.6]

# 优先 Times New Roman；本机未安装时自动使用 Times 度量兼容的 Nimbus Roman。
FONT_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
RNA_COLOR = NMI["baseline_mid"]
PROTEIN_COLOR = NMI["ours_base"]
BURIED_COLOR = NMI["ours_deep"]
READABLE_COLOR = NMI["neutral_mid"]
GAIN_UP_COLOR = NMI["delta_up"]
GAIN_DOWN_COLOR = NMI["delta_down"]
BURIAL_ZONE_COLOR = NMI["bg_lilac"]
BURIAL_ZONE_ALPHA = 0.50
INK = NMI["ink"]
BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 0.75


def apply_paper_font() -> None:
    """Configure editable, Times-compatible vector text."""
    for font_path in fm.findSystemFonts():
        filename = os.path.basename(font_path).lower()
        if "nimbusrom" in filename or "nimbus_roman" in filename or filename.startswith("times"):
            try:
                fm.fontManager.addfont(font_path)
            except RuntimeError:
                pass
    plt.rcParams.update({
        "font.family": "serif", "font.serif": FONT_SERIF,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def model_label(model: str) -> str:
    return {
        "DeepSeek V4 Flash": "DeepSeek\nV4 Flash",
        "GPT-5.4 Mini": "GPT\n5.4 Mini",
        "Gemini 3.5 Flash": "Gemini\n3.5 Flash",
        "Qwen 3.7 Plus": "Qwen\n3.7 Plus",
        "GPT-5.6 Sol": "GPT\n5.6 Sol",
        "Claude Opus 4.8": "Claude\nOpus 4.8",
    }[model]


def add_combined_panel_tags(fig: plt.Figure) -> None:
    """Add a/b/c only once, in fixed figure coordinates for reliable alignment."""
    for label, (x, y) in COMBINED_PANEL_TAG_POSITIONS.items():
        fig.text(x, y, label, transform=fig.transFigure, fontsize=PANEL_LABEL_SIZE,
                 fontweight="bold", ha="left", va="top", color=INK)


def draw_marker_panel(ax: plt.Axes) -> None:
    x = np.arange(len(MARKERS))
    rna = np.array([row[1] for row in MARKERS])
    protein = np.array([row[2] for row in MARKERS])
    ax.bar(x - MARKER_BAR_WIDTH / 2, rna, MARKER_BAR_WIDTH, color=RNA_COLOR,
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    ax.bar(x + MARKER_BAR_WIDTH / 2, protein, MARKER_BAR_WIDTH, color=PROTEIN_COLOR,
           edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    for index, (rna_value, protein_value) in enumerate(zip(rna, protein)):
        ax.text(index - MARKER_BAR_WIDTH / 2, rna_value + MARKER_VALUE_Y_OFFSET, f"{rna_value:.0f}",
                ha="center", va="bottom", fontsize=MARKER_VALUE_FONT_SIZE, color=INK)
        ax.text(index + MARKER_BAR_WIDTH / 2, protein_value + MARKER_VALUE_Y_OFFSET, f"{protein_value:.0f}",
                ha="center", va="bottom", fontsize=MARKER_VALUE_FONT_SIZE, color=INK)
    ax.set_ylabel("Visible in cell (%)", fontsize=MARKER_YLABEL_SIZE, labelpad=AXIS_LABEL_PAD)
    ax.set_ylim(*MARKER_YLIM)
    ax.set_yticks(MARKER_YTICKS)
    ax.set_xticks(x, [row[0] for row in MARKERS], fontsize=MARKER_XTICK_SIZE)
    ax.tick_params(axis="y", labelsize=MARKER_YTICK_SIZE, pad=TICK_LABEL_PAD)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.set_title(A_TITLE, loc=PANEL_TITLE_LOC, fontsize=A_TITLE_SIZE, fontweight="bold",
                 color=INK, x=PANEL_TITLE_X_BY_PANEL["a"], y=PANEL_TITLE_Y, pad=0)


def draw_scatter_panel(ax: plt.Axes) -> None:
    ax.axvspan(SCATTER_XLIM[0], BURIAL_ZONE_RIGHT, color=BURIAL_ZONE_COLOR,
               alpha=BURIAL_ZONE_ALPHA, zorder=0)
    ax.axhline(0, color=NMI["neutral_dark"], linewidth=ZERO_LINE_WIDTH, linestyle="--", zorder=1)
    for name, visibility, gain in CELL_TYPES:
        color = GAIN_UP_COLOR if gain > 0 else GAIN_DOWN_COLOR
        ax.scatter(visibility, gain, s=SCATTER_SIZE, color=color, edgecolor="white",
                   linewidth=SCATTER_EDGE_WIDTH, zorder=3)
        dx, dy = CELL_LABEL_OFFSETS[name]
        ax.annotate(name, (visibility, gain), xytext=(dx, dy), textcoords="offset points",
                    fontsize=B_POINT_LABEL_SIZE, color=INK)
    ax.set_xlim(*SCATTER_XLIM)
    ax.set_ylim(*SCATTER_YLIM)
    ax.set_xticks(SCATTER_XTICKS)
    ax.set_yticks(SCATTER_YTICKS)
    ax.tick_params(axis="x", labelsize=SCATTER_XTICK_SIZE, pad=TICK_LABEL_PAD)
    ax.tick_params(axis="y", labelsize=SCATTER_YTICK_SIZE, pad=TICK_LABEL_PAD)
    ax.set_xlabel("Marker visibility in text (%)", fontsize=SCATTER_XLABEL_SIZE, labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel("Gain from Ours (pts)", fontsize=SCATTER_YLABEL_SIZE, labelpad=AXIS_LABEL_PAD)
    ax.grid(color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.set_title(B_TITLE, loc=PANEL_TITLE_LOC, fontsize=B_TITLE_SIZE, fontweight="bold",
                 color=INK, x=PANEL_TITLE_X_BY_PANEL["b"], y=PANEL_TITLE_Y, pad=0)


def draw_program_panel(ax: plt.Axes) -> None:
    y = np.arange(len(MODELS))
    for index, (buried, readable) in enumerate(zip(BURIED_PROGRAM_GAIN, READABLE_PROGRAM_GAIN)):
        ax.plot([readable, buried], [index, index], color=NMI["neutral_light"],
                linewidth=PROGRAM_LINE_WIDTH, solid_capstyle="round", zorder=1)
        ax.scatter(readable, index, s=PROGRAM_DOT_SIZE, color=READABLE_COLOR,
                   edgecolor="white", linewidth=SCATTER_EDGE_WIDTH, zorder=3)
        ax.scatter(buried, index, s=PROGRAM_DOT_SIZE, color=BURIED_COLOR,
                   edgecolor="white", linewidth=SCATTER_EDGE_WIDTH, zorder=3)
        for series, value in (("readable", readable), ("buried", buried)):
            direction = 1 if value >= 0 else -1
            extra_x, extra_y = PROGRAM_VALUE_OFFSETS.get((MODELS[index], series), (0, 0))
            ax.annotate(f"{value:+.1f}", (value, index),
                        xytext=(direction * PROGRAM_VALUE_X_OFFSET + extra_x, extra_y), textcoords="offset points",
                        ha="left" if direction > 0 else "right", va="center",
                        fontsize=PROGRAM_VALUE_FONT_SIZE, color=INK)
    ax.axvline(0, color=NMI["neutral_dark"], linewidth=PROGRAM_ZERO_LINE_WIDTH,
               linestyle="--", zorder=0)
    ax.set_xlabel("Gains from labeling (pts)", fontsize=PROGRAM_XLABEL_SIZE, labelpad=AXIS_LABEL_PAD)
    ax.set_xlim(*PROGRAM_XLIM)
    ax.set_xticks(PROGRAM_XTICKS)
    ax.set_yticks(y, [model_label(model) for model in MODELS], fontsize=PROGRAM_MODEL_TICK_SIZE)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=PROGRAM_XTICK_SIZE, pad=TICK_LABEL_PAD)
    ax.tick_params(axis="y", length=0, pad=TICK_LABEL_PAD)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    ax.set_axisbelow(True)
    ax.set_title(C_TITLE, loc=PANEL_TITLE_LOC, fontsize=C_TITLE_SIZE, fontweight="bold",
                 color=INK, x=PANEL_TITLE_X_BY_PANEL["c"], y=PANEL_TITLE_Y, pad=0)


def legend_handles() -> dict[str, list]:
    """Return semantic legend groups for the three panels and the combined figure."""
    handles_a = [
        Patch(facecolor=RNA_COLOR, edgecolor="none", label="RNA marker in text"),
        Patch(facecolor=PROTEIN_COLOR, edgecolor="none", label="Protein present"),
    ]
    handles_b = [
        Line2D([0], [0], marker="o", linestyle="none", color=GAIN_UP_COLOR, markersize=LEGEND_MARKER_SIZE, label="Ours helps"),
        Line2D([0], [0], marker="o", linestyle="none", color=GAIN_DOWN_COLOR, markersize=LEGEND_MARKER_SIZE, label="Ours hurts"),
    ]
    handles_c = [
        Line2D([0], [0], marker="o", linestyle="none", color=BURIED_COLOR,
               markerfacecolor=BURIED_COLOR, markeredgecolor="white", markeredgewidth=SCATTER_EDGE_WIDTH,
               markersize=PROGRAM_LEGEND_MARKER_SIZE, label="Buried program"),
        Line2D([0], [0], marker="o", linestyle="none", color=READABLE_COLOR,
               markerfacecolor=READABLE_COLOR, markeredgecolor="white", markeredgewidth=SCATTER_EDGE_WIDTH,
               markersize=PROGRAM_LEGEND_MARKER_SIZE, label="Readable program"),
    ]
    return {"a": handles_a, "b": handles_b, "c": handles_c, "combined": handles_a + handles_b + handles_c}


def add_legend(ax: plt.Axes, handles: list, ncol: int, fontsize: float) -> None:
    ax.legend(handles=handles, ncol=ncol, loc="center", fontsize=fontsize,
              handlelength=LEGEND_HANDLE_LENGTH, handletextpad=LEGEND_HANDLE_TEXT_PAD,
              columnspacing=LEGEND_COLUMN_SPACING)
    ax.set_axis_off()


def save_figure(fig: plt.Figure, out_base: Path) -> list[str]:
    outputs = []
    for extension in EXPORT_FORMATS:
        output = out_base.with_suffix(f".{extension}")
        # Preserve FIG_SIZE exactly; tight cropping would make LaTeX rescale it.
        fig.savefig(output, dpi=PNG_DPI)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def draw_single_panel(draw_panel, size: tuple[float, float], out_base: Path, handles: list) -> list[str]:
    """Export one enlarged, self-contained panel for direct visual review and tuning."""
    apply_style(font_size=BASE_FONT_SIZE, axes_linewidth=AXIS_LINE_WIDTH)
    apply_paper_font()
    fig = plt.figure(figsize=size, layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=SINGLE_HEIGHT_RATIOS, hspace=SINGLE_HSPACE)
    axis = fig.add_subplot(grid[0])
    legend_axis = fig.add_subplot(grid[1])
    draw_panel(axis)
    add_legend(legend_axis, handles, ncol=len(handles), fontsize=SINGLE_LEGEND_FONT_SIZE)
    return save_figure(fig, out_base)


def draw_combined_figure() -> list[str]:
    apply_style(font_size=BASE_FONT_SIZE, axes_linewidth=AXIS_LINE_WIDTH)
    apply_paper_font()
    # Manual axes positions deliberately decouple the combined layout from
    # each panel's y-axis labels and multi-line model names.
    fig = plt.figure(figsize=FIG_SIZE)
    ax_a = fig.add_axes(COMBINED_PLOT_POSITIONS["a"])
    ax_b = fig.add_axes(COMBINED_PLOT_POSITIONS["b"])
    ax_c = fig.add_axes(COMBINED_PLOT_POSITIONS["c"])
    ax_legend_a = fig.add_axes(COMBINED_LEGEND_POSITIONS["a"])
    ax_legend_b = fig.add_axes(COMBINED_LEGEND_POSITIONS["b"])
    ax_legend_c = fig.add_axes(COMBINED_LEGEND_POSITIONS["c"])
    draw_marker_panel(ax_a)
    draw_scatter_panel(ax_b)
    draw_program_panel(ax_c)
    handles = legend_handles()
    add_legend(ax_legend_a, handles["a"], ncol=COMBINED_LEGEND_NCOL, fontsize=COMBINED_LEGEND_FONT_SIZE)
    add_legend(ax_legend_b, handles["b"], ncol=COMBINED_LEGEND_NCOL, fontsize=COMBINED_LEGEND_FONT_SIZE)
    add_legend(ax_legend_c, handles["c"], ncol=COMBINED_LEGEND_NCOL, fontsize=COMBINED_LEGEND_FONT_SIZE)
    add_combined_panel_tags(fig)
    return save_figure(fig, OUT_COMBINED)


if __name__ == "__main__":
    all_outputs = []
    handles = legend_handles()
    all_outputs.extend(draw_single_panel(draw_marker_panel, SINGLE_FIG_SIZE_A, OUT_A, handles["a"]))
    all_outputs.extend(draw_single_panel(draw_scatter_panel, SINGLE_FIG_SIZE_B, OUT_B, handles["b"]))
    all_outputs.extend(draw_single_panel(draw_program_panel, SINGLE_FIG_SIZE_C, OUT_C, handles["c"]))
    all_outputs.extend(draw_combined_figure())
    for output in all_outputs:
        print(output)
