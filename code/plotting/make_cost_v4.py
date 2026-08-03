#!/usr/bin/env python3
"""Cost figure v4: single-column token decomposition and accuracy dumbbells.

The upper panel shows the counterintuitive budget trade-off: prompt/answer
tokens rise, yet reasoning and total tokens fall. The lower panel independently
shows the paired accuracy gain for each model, without implying a cross-model
line trend.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nmi_style import NMI, apply_style  # noqa: E402


HERE = Path(__file__).parent
DATA_FILE = HERE / "cost_source_data.tsv"
OUT_BASE = HERE / "cost_v4_token_budget_accuracy"

# Display order follows the manuscript comparison: Qwen, DeepSeek Flash,
# DeepSeek Pro, then Gemini.
MODELS = [
    "Qwen 3.7 Plus",
    "DeepSeek V4 Flash",
    "DeepSeek V4 Pro",
    "Gemini 3.5 Flash",
]
# Evenly spaced groups, with a slightly wider interval than the original
# layout so the two DeepSeek labels remain distinguishable at 10 pt.
MODEL_X = (0.0, 1.15, 2.30, 3.45)
MODEL_X_LIM = (-0.55, 3.95)
# NMI pastel families, following the project-wide convention:
# Baseline = blue-violet; Ours = pink; green is reserved for beneficial deltas.
BASELINE_REASONING = NMI["baseline_mid"]
BASELINE_NONREASON = NMI["baseline_soft"]
BASELINE_ACCENT = NMI["baseline_dark"]
OURS_REASONING = NMI["ours_deep"]
OURS_NONREASON = NMI["ours_soft"]
OURS_ACCENT = NMI["ours_deep"]
DELTA_UP = NMI["delta_up"]
INK = NMI["ink"]
GRID = "#E8E8E8"

# ========================= CONFIG（常用微调都在这里）=========================
# ---- 1. 单栏画布与总体布局 ----
# Exact final single-column page size.  The height preserves the figure's
# existing placed height in the manuscript, avoiding any page-count change.
FIG_SIZE = (3.3125, 3.837)       # 宽、高单位为英寸
# 三个数依次控制 a 图高度 / a 图底端横向图例 / b 图高度；调第 2 个数可改变图例留白。
PANEL_HEIGHT_RATIOS = (2.45, 0.24, 1.28)
PANEL_HSPACE = 0.08
PNG_DPI = 600

# ---- 2. 字体、坐标轴与图例 ----
# 本图使用 Times New Roman 度量兼容的 Nimbus Roman；SVG/PDF 文本保持可编辑。
FONT_SERIF = ["Nimbus Roman No9 L", "Nimbus Roman", "Times New Roman",
              "Liberation Serif", "DejaVu Serif"]
BASE_FONT_SIZE = 10
AXIS_LINE_WIDTH = 1
PANEL_TITLE_SIZE = 11
PANEL_LABEL_SIZE = 15
TICK_LABEL_SIZE = 10
AXIS_LABEL_SIZE = 10
# Panel letters sit in the same alignment band as the y-axis labels, rather
# than occupying a separate far-left column.
PANEL_LABEL_X = -0.18
# 一行四项图例，位于 a 图底端；若仍觉得拥挤，优先减小 LEGEND_FONT_SIZE，再调列间距。
LEGEND_NCOL = 4
LEGEND_FONT_SIZE = 10
LEGEND_COLUMN_SPACING = 0.5
LEGEND_HANDLE_LENGTH = 0.9
LEGEND_HANDLE_TEXT_PAD = 0.20
TOKEN_TITLE = "Lower reasoning-token and total-token usage"
ACCURACY_TITLE = "Accuracy improves for every model"

# ---- 3. a 图：token 堆叠柱与降幅文字 ----
BAR_WIDTH = 0.35
BAR_OFFSET = 0.18
BAR_EDGE_COLOR = "white"          # 段与段之间的分界；底部 x 轴会单独置顶保证连续
BAR_EDGE_WIDTH = 0.7
TOKEN_YLIM = (0, 4050)
TOKEN_YTICKS = (0, 1000, 2000, 3000, 4000)
TOKEN_YTICK_LABELS = ("0", "1k", "2k", "3k", "4k")
TOTAL_DELTA_Y_OFFSET = 105
TOTAL_DELTA_FONT_SIZE = 10        # 柱顶：只显示 total 降幅
# 在同一模型的两根柱之间横向居中标注 reasoning 降幅；文字可跨越两种颜色，不受单根柱宽限制。
REASONING_DELTA_TEMPLATE = "Reasoning\n ↓{percent}%"
REASONING_DELTA_FONT_SIZE = 10.5
REASONING_DELTA_X_OFFSET = 0.0
# 相对较低的整根柱顶端下移；使用两根柱中较低的顶端，可确保文字仍落在两根实心 reasoning 段内。
REASONING_DELTA_Y_OFFSET = -700
REASONING_DELTA_COLOR = INK        # 深色文字在粉色实心段与相邻留白上都保持可读

# ---- 4. b 图：accuracy dumbbell / 箭头 / 增益文字 ----
ACCURACY_YLIM = (35, 85)
ACCURACY_YTICKS = (40, 60, 80)
ACCURACY_ARROW_LW = 2.5
ACCURACY_ARROW_MUTATION_SCALE = 25
ACCURACY_BASE_MARKER_SIZE = 95
ACCURACY_OURS_MARKER_SIZE = 95
ACCURACY_MARKER_EDGE_WIDTH = 0.7
ACCURACY_DELTA_FONT_SIZE = 10
# Place every gain consistently at the upper left of its Ours marker.
# Adjust an individual item here as (Qwen, DeepSeek Flash, DeepSeek Pro, Gemini).
ACCURACY_DELTA_X_OFFSETS = (-0.18, -0.1, -0.1, -0.18)
ACCURACY_DELTA_ALIGNMENTS = ("center", "right", "right", "right")
MODEL_LABEL_SIZE = 10
# 对应 Qwen / DeepSeek Flash / DeepSeek Pro / Gemini；Gemini 单独下移，避免 +16.5 贴近图顶。
ACCURACY_DELTA_Y_OFFSETS = (2, 1.5, 1.5, -4.0)
# ============================================================================


def apply_paper_font() -> None:
    """Use the manuscript's Times-compatible serif font after shared styling."""
    for font_path in fm.findSystemFonts():
        font_name = os.path.basename(font_path).lower()
        if "nimbusrom" in font_name or "nimbus_roman" in font_name or font_name.startswith("times"):
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


def load_data(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load all supplied rows; no observations are filtered or transformed."""
    data: dict[str, dict[str, dict[str, float]]] = {model: {} for model in MODELS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            model = row["model"]
            if model not in data:
                raise ValueError(f"Unexpected model in source data: {model}")
            condition = row["condition"]
            if condition in data[model]:
                raise ValueError(f"Duplicate row: {model} / {condition}")
            data[model][condition] = {
                key: float(row[key])
                for key in ("accuracy_pct", "reasoning_tokens", "input_output_tokens", "total_tokens")
            }
    for model, rows in data.items():
        if set(rows) != {"Baseline", "Ours"}:
            raise ValueError(f"Expected Baseline and Ours rows for {model}; found {sorted(rows)}")
        for condition, values in rows.items():
            reconstructed_total = values["reasoning_tokens"] + values["input_output_tokens"]
            if not np.isclose(reconstructed_total, values["total_tokens"]):
                raise ValueError(f"Token components do not sum to total for {model} / {condition}")
    return data


def model_label(model: str) -> str:
    return {
        "Qwen 3.7 Plus": "Qwen\n3.7 Plus",
        "DeepSeek V4 Flash": "DeepSeek\nV4 Flash",
        "DeepSeek V4 Pro": "DeepSeek\nV4 Pro",
        "Gemini 3.5 Flash": "Gemini\n3.5 Flash",
    }[model]


def add_encoding_key(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0], [0], color=BASELINE_ACCENT, marker="o", markerfacecolor=BASELINE_ACCENT,
            markeredgecolor="white", markeredgewidth=0.5, linestyle="none", markersize=5.5,
            label="Baseline",
        ),
        Line2D(
            [0], [0], color=OURS_ACCENT, marker="o", markerfacecolor=OURS_ACCENT,
            markeredgecolor="white", markeredgewidth=0.5, linestyle="none", markersize=5.5,
            label="Ours",
        ),
        Patch(facecolor="white", edgecolor=NMI["neutral_mid"], hatch="//", label="Input + output"),
        Patch(facecolor=NMI["neutral_mid"], edgecolor="none", label="Reasoning"),
    ]
    ax.legend(
        handles=handles,
        ncol=LEGEND_NCOL,
        loc="center",
        columnspacing=LEGEND_COLUMN_SPACING,
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        fontsize=LEGEND_FONT_SIZE,
    )


def draw_figure(data: dict[str, dict[str, dict[str, float]]]) -> plt.Figure:
    apply_style(font_size=BASE_FONT_SIZE, axes_linewidth=AXIS_LINE_WIDTH)
    apply_paper_font()
    # 89-mm single-column figure: a compact legend strip plus vertically stacked evidence.
    fig = plt.figure(figsize=FIG_SIZE, layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=PANEL_HEIGHT_RATIOS, hspace=PANEL_HSPACE)
    ax_tokens = fig.add_subplot(grid[0])
    ax_key = fig.add_subplot(grid[1])
    ax_accuracy = fig.add_subplot(grid[2], sharex=ax_tokens)
    ax_key.set_axis_off()
    add_encoding_key(ax_key)

    x = np.asarray(MODEL_X)
    x_baseline = x - BAR_OFFSET
    x_ours = x + BAR_OFFSET

    for i, model in enumerate(MODELS):
        baseline = data[model]["Baseline"]
        ours = data[model]["Ours"]
        for position, values, nonreason_color, reasoning_color in (
            (x_baseline[i], baseline, BASELINE_NONREASON, BASELINE_REASONING),
            (x_ours[i], ours, OURS_NONREASON, OURS_REASONING),
        ):
            ax_tokens.bar(
                position,
                values["input_output_tokens"],
                width=BAR_WIDTH,
                color=nonreason_color,
                hatch="//",
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
            )
            ax_tokens.bar(
                position,
                values["reasoning_tokens"],
                width=BAR_WIDTH,
                bottom=values["input_output_tokens"],
                color=reasoning_color,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
            )

        # Direct evidence for the central claim, derived from the source values.
        total_reduction = round(100 * (1 - ours["total_tokens"] / baseline["total_tokens"]))
        reasoning_reduction = round(100 * (1 - ours["reasoning_tokens"] / baseline["reasoning_tokens"]))
        ax_tokens.text(
            x[i],
            max(baseline["total_tokens"], ours["total_tokens"]) + TOTAL_DELTA_Y_OFFSET,
            f"total ↓{total_reduction}%",
            ha="center", va="bottom", fontsize=TOTAL_DELTA_FONT_SIZE, color=DELTA_UP,
            fontweight="bold", zorder=6,
        )
        ax_tokens.text(
            x[i] + REASONING_DELTA_X_OFFSET,
            min(baseline["total_tokens"], ours["total_tokens"]) + REASONING_DELTA_Y_OFFSET,
            REASONING_DELTA_TEMPLATE.format(percent=reasoning_reduction),
            ha="center", va="center", fontsize=REASONING_DELTA_FONT_SIZE,
            color=REASONING_DELTA_COLOR, fontweight="bold", zorder=6,
        )

        # One within-model dumbbell: no connecting line is drawn across model categories.
        baseline_accuracy = baseline["accuracy_pct"]
        ours_accuracy = ours["accuracy_pct"]
        ax_accuracy.annotate(
            "", xy=(x[i], ours_accuracy), xytext=(x[i], baseline_accuracy),
            arrowprops=dict(arrowstyle="->", color=DELTA_UP, lw=ACCURACY_ARROW_LW,
                            mutation_scale=ACCURACY_ARROW_MUTATION_SCALE),
            zorder=2,
        )
        ax_accuracy.scatter(x[i], baseline_accuracy, s=ACCURACY_BASE_MARKER_SIZE,
                            color=BASELINE_ACCENT, edgecolor="white",
                            linewidth=ACCURACY_MARKER_EDGE_WIDTH, zorder=4)
        ax_accuracy.scatter(x[i], ours_accuracy, s=ACCURACY_OURS_MARKER_SIZE,
                            color=OURS_ACCENT, edgecolor="white",
                            linewidth=ACCURACY_MARKER_EDGE_WIDTH, zorder=5)
        ax_accuracy.text(
            x[i] + ACCURACY_DELTA_X_OFFSETS[i], ours_accuracy + ACCURACY_DELTA_Y_OFFSETS[i],
            f"+{ours_accuracy - baseline_accuracy:.1f}",
            fontsize=ACCURACY_DELTA_FONT_SIZE, color=DELTA_UP, fontweight="bold",
            ha=ACCURACY_DELTA_ALIGNMENTS[i], va="bottom",
        )

    ax_tokens.set_ylabel("Tokens per query", fontsize=AXIS_LABEL_SIZE)
    ax_tokens.set_ylim(*TOKEN_YLIM)
    ax_tokens.set_yticks(TOKEN_YTICKS, labels=TOKEN_YTICK_LABELS)
    ax_tokens.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax_tokens.grid(axis="y", color=GRID, linewidth=0.7)
    ax_tokens.set_axisbelow(True)
    ax_tokens.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_tokens.set_title(TOKEN_TITLE, loc="center", fontsize=PANEL_TITLE_SIZE,
                        color=INK, fontweight="bold", pad=3)
    ax_tokens.text(PANEL_LABEL_X, 1.02, "a", transform=ax_tokens.transAxes,
                   fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="bottom")
    # Bars use white edges for component separation; draw the axis spine above them.
    ax_tokens.spines["bottom"].set_zorder(10)

    ax_accuracy.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_SIZE)
    ax_accuracy.set_ylim(*ACCURACY_YLIM)
    ax_accuracy.set_yticks(ACCURACY_YTICKS)
    ax_accuracy.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax_accuracy.grid(axis="y", color=GRID, linewidth=0.7)
    ax_accuracy.set_axisbelow(True)
    ax_accuracy.set_xticks(x, [model_label(model) for model in MODELS])
    ax_accuracy.tick_params(axis="x", length=0, pad=4, labelsize=MODEL_LABEL_SIZE)
    ax_accuracy.set_title(ACCURACY_TITLE, loc="center", fontsize=PANEL_TITLE_SIZE,
                          color=INK, fontweight="bold", pad=3)
    ax_accuracy.text(PANEL_LABEL_X, 1.02, "b", transform=ax_accuracy.transAxes,
                     fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="bottom")
    ax_tokens.set_xlim(*MODEL_X_LIM)

    return fig


def save_figure(fig: plt.Figure) -> list[Path]:
    svg_path = HERE / "cost_v4_token_budget_accuracy.svg"
    pdf_path = HERE / "cost_v4_token_budget_accuracy.pdf"
    png_path = HERE / "cost_v4_token_budget_accuracy.png"
    tiff_path = HERE / "cost_v4_token_budget_accuracy.tiff"
    # Preserve FIG_SIZE exactly so LaTeX does not rescale the final figure.
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=PNG_DPI)
    fig.savefig(tiff_path, dpi=PNG_DPI)
    output_paths = [svg_path, pdf_path, png_path, tiff_path]
    return output_paths


if __name__ == "__main__":
    records = load_data(DATA_FILE)
    figure = draw_figure(records)
    for output in save_figure(figure):
        print(output)
    plt.close(figure)
