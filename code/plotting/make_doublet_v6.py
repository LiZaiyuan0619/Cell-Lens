#!/usr/bin/env python3
"""Doublet v6: compact five-model vertical auditability ladder.

Each Baseline/Ours bar partitions genotype-truth doublets into missed, partial
attribution, and full attribution.  Detection-accuracy gains are reported as
concise direct labels above each model pair. Claude Sonnet 5 is retained in the
source-data file but intentionally excluded from this primary figure because
its small effect would obscure the cross-model pattern.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nmi_style import NMI, apply_style  # noqa: E402


HERE = Path(__file__).parent

# ========================= CONFIG（常用微调都在这里）=========================
# ---- 1. 数据、画布与导出（最常用：整体宽高）----
SOURCE_DATA = HERE / "doublet_v6_source_data.csv"
OUT_BASE = HERE / "doublet_v6_auditability_ladder"
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
PNG_DPI = 600
MODEL_ORDER = [
    "GPT-5.4 Mini",
    "Qwen 3.7 Plus",
    "DeepSeek V4 Flash",
    "Gemini 3.5 Flash",
    "GPT-5.6 Sol",
]
# 最终单栏 PDF 页面尺寸（英寸）。高度保持论文中当前的实际占位，避免增页。
FIG_SIZE = (3.3125, 3.058)
# 主绘图区 / 图例区的高度比例；想给图例更多空间时提高第二个值。
PANEL_HEIGHT_RATIOS = (3.23, 0.30)
PANEL_HSPACE = 0.04
FIG_TITLE = "Ours makes doublet calls more verifiable"
GAIN_SUBTITLE = "Green means accuracy gain(%)"

# ---- 2. 字体与坐标轴（最常用：各部分字号）----
FONT_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
# 全局默认字号；单独字号未指定时由它控制。
BASE_FONT_SIZE = 10
AXIS_LINE_WIDTH = 1.0
# 标题、模型名、刻度、轴标题的字号。单栏内所有可读文本不低于 9.5 pt。
PANEL_TITLE_SIZE = 11
GAIN_SUBTITLE_SIZE = 10
FIG_TITLE_Y = 1.080
GAIN_SUBTITLE_Y = 1.015
MODEL_LABEL_SIZE = 9.5
TICK_LABEL_SIZE = 10
AXIS_LABEL_SIZE = 10.5

# ---- 3. 语义配色 ----
MISS_COLOR = "#E7DCD2"
INCOMPLETE_COLOR = NMI["baseline_soft"]
AUDITABLE_COLOR = NMI["ours_base"]
OURS_LEGEND_COLOR = "#EEEEEE"
ACCURACY_COLOR = NMI["delta_up"]
INK = NMI["ink"]
GRID = "#E8E8E8"
BASELINE_HATCH = "////"
BAR_EDGE_COLOR = "white"
BAR_EDGE_WIDTH = 0.75

# ---- 4. 几何与文字（最常用：柱宽、两柱间距、顶部标签位置）----
# 单根柱宽；增大后柱更粗，过大则同一模型的两柱会挤在一起。
BAR_WIDTH = 0.32
# 同一模型 Baseline/Ours 相对中心的左右偏移；增大可拉开两根柱。
PAIR_OFFSET = 0.19
# 左轴显示范围与刻度；上限需高于 100，给顶部 Acc. 标签留空间。
Y_LIM = (0, 108)
Y_TICKS = (0, 25, 50, 75, 100)
# 顶部 accuracy-gain 标签：caption 已定义度量，因此只显示紧凑的增益数值。
# 可在此调整全局基线、字号与每个模型的 (x, y) 偏移；顺序与 MODEL_ORDER 相同。
ACCURACY_GAIN_Y = 101.0
ACCURACY_GAIN_FONT_SIZE = 9.5
ACCURACY_GAIN_OFFSETS = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
ACCURACY_GAIN_TEMPLATE = "+{gain:.1f}"
# 图例：单行项目数、字号、色块宽度、项目间距。
# 图例：标签使用 caption 已定义的紧凑术语；下列四项可独立微调。
LEGEND_NCOL = 5
LEGEND_FONT_SIZE = 9.5
LEGEND_HANDLE_LENGTH = 0.60
LEGEND_HANDLE_TEXT_PAD = 0.15
LEGEND_COLUMN_SPACING = 0.16
# 五组的 x 位置和显示名。增大相邻差值会拉开该两组柱；改 X_LIM 可控制两端留白。
MODEL_X = (0.0, 1.0, 2.0, 3.0, 4.0)
MODEL_X_LIM = (-0.58, 4.58)
MODEL_LABELS = {
    "GPT-5.4 Mini": "GPT-5.4\nMini",
    "Qwen 3.7 Plus": "Qwen\n3.7 Plus",
    "DeepSeek V4 Flash": "DeepSeek\nV4 Flash",
    "Gemini 3.5 Flash": "Gemini\n3.5 Flash",
    "GPT-5.6 Sol": "GPT-5.6\nSol",
}
# ============================================================================


def apply_paper_font() -> None:
    """Use a Times-compatible serif family while retaining editable vector text."""
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


def load_data() -> list[dict[str, float | str]]:
    """Load five selected models; source CSV retains the excluded Claude run."""
    required = {
        "model", "baseline_missed", "baseline_incomplete", "baseline_auditable",
        "ours_missed", "ours_incomplete", "ours_auditable",
        "baseline_accuracy", "ours_accuracy", "n_true_doublets",
    }
    with SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{SOURCE_DATA.name} must contain: {', '.join(sorted(required))}")
        parsed_rows = []
        for row in reader:
            parsed: dict[str, float | str] = {"model": row["model"]}
            for key in required - {"model"}:
                parsed[key] = float(row[key])
            for arm in ("baseline", "ours"):
                values = [float(parsed[f"{arm}_{part}"]) for part in ("missed", "incomplete", "auditable")]
                if not np.isclose(sum(values), 100.0, atol=0.11):
                    raise ValueError(f"{row['model']} {arm} composition must sum to 100, got {sum(values):.2f}")
            parsed_rows.append(parsed)
    by_model = {str(row["model"]): row for row in parsed_rows}
    missing = set(MODEL_ORDER) - set(by_model)
    if missing:
        raise ValueError(f"Missing selected model rows: {', '.join(sorted(missing))}")
    return [by_model[model] for model in MODEL_ORDER]


def model_label(model: str) -> str:
    return MODEL_LABELS[model]


def add_legend(ax: plt.Axes) -> None:
    ax.legend(
        handles=[
            Patch(facecolor="#FFFFFF", edgecolor=NMI["neutral_dark"], hatch=BASELINE_HATCH, label="Baseline"),
            Patch(facecolor=OURS_LEGEND_COLOR, edgecolor="none", label="Ours"),
            Patch(facecolor=MISS_COLOR, edgecolor="none", label="Missed"),
            Patch(facecolor=INCOMPLETE_COLOR, edgecolor="none", label="Partial"),
            Patch(facecolor=AUDITABLE_COLOR, edgecolor="none", label="Full"),
        ],
        ncol=LEGEND_NCOL, loc="center", fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH, handletextpad=LEGEND_HANDLE_TEXT_PAD,
        columnspacing=LEGEND_COLUMN_SPACING,
    )
    ax.set_axis_off()


def save_figure(fig: plt.Figure) -> list[str]:
    outputs = []
    for extension in EXPORT_FORMATS:
        output = OUT_BASE.with_suffix(f".{extension}")
        # Preserve FIG_SIZE exactly: tight cropping would trigger LaTeX scaling.
        fig.savefig(output, dpi=PNG_DPI)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def draw_figure() -> list[str]:
    rows = load_data()
    apply_style(font_size=BASE_FONT_SIZE, axes_linewidth=AXIS_LINE_WIDTH)
    apply_paper_font()
    fig = plt.figure(figsize=FIG_SIZE, layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=PANEL_HEIGHT_RATIOS, hspace=PANEL_HSPACE)
    ax = fig.add_subplot(grid[0])
    ax_key = fig.add_subplot(grid[1])
    x = np.asarray(MODEL_X)

    for index, row in enumerate(rows):
        x_positions = {"baseline": x[index] - PAIR_OFFSET, "ours": x[index] + PAIR_OFFSET}
        for arm, x_position, hatch in (("baseline", x_positions["baseline"], BASELINE_HATCH),
                                       ("ours", x_positions["ours"], None)):
            missed = float(row[f"{arm}_missed"])
            incomplete = float(row[f"{arm}_incomplete"])
            auditable = float(row[f"{arm}_auditable"])
            for value, bottom, color in (
                (missed, 0, MISS_COLOR),
                (incomplete, missed, INCOMPLETE_COLOR),
                (auditable, missed + incomplete, AUDITABLE_COLOR),
            ):
                ax.bar(x_position, value, BAR_WIDTH, bottom=bottom, color=color,
                       edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, hatch=hatch, zorder=3)
        baseline_accuracy = float(row["baseline_accuracy"])
        ours_accuracy = float(row["ours_accuracy"])
        accuracy_gain = ours_accuracy - baseline_accuracy
        gain_dx, gain_dy = ACCURACY_GAIN_OFFSETS[index]
        ax.text(x[index] + gain_dx, ACCURACY_GAIN_Y + gain_dy,
                ACCURACY_GAIN_TEMPLATE.format(gain=accuracy_gain), ha="center", va="bottom",
                fontsize=ACCURACY_GAIN_FONT_SIZE, color=ACCURACY_COLOR, fontweight="bold", zorder=9)

    ax.set_xlim(*MODEL_X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_ylabel("Genotype-truth doublets (%)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, zorder=11)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(x, [model_label(str(row["model"])) for row in rows], fontsize=MODEL_LABEL_SIZE)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["top"].set_visible(False)
    # 白色柱边会覆盖 y=0 附近的默认 spine；在所有柱上方重画连续 x 轴，避免断裂。
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color="#444444", linewidth=AXIS_LINE_WIDTH, zorder=12, clip_on=False)
    ax.tick_params(axis="x", zorder=12)
    ax.set_title(FIG_TITLE, loc="center", y=FIG_TITLE_Y, fontsize=PANEL_TITLE_SIZE,
                 color=INK, fontweight="bold", pad=0)
    ax.text(0.5, GAIN_SUBTITLE_Y, GAIN_SUBTITLE, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=GAIN_SUBTITLE_SIZE,
            color=ACCURACY_COLOR, clip_on=False)
    add_legend(ax_key)
    return save_figure(fig)


if __name__ == "__main__":
    for output in draw_figure():
        print(output)
