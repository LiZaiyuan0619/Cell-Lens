"""
Shared Nature/NMI figure style for the CellCode paper (adapted from the project's
nature-figure skill). Import from any figs/*/vN/ script:  sys.path append figs/.

Palette policy: unified families -- Baseline = blue-violet family, Ours = pink
family; green/red reserved for delta (gain/drop). Editable SVG+PDF text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- NMI pastel palette (families, not per-item hues) ---
NMI = {
    "baseline_dark": "#484878", "baseline_mid": "#7884B4", "baseline_soft": "#B4C0E4",
    "ours_soft": "#E4CCD8", "ours_base": "#E0A8C4", "ours_deep": "#C86AA0",
    "bg_lilac": "#E0E0F0", "bg_aqua": "#E0F0F0", "bg_peach": "#F0E0D0",
    "neutral_light": "#D8D8D8", "neutral_mid": "#A8A8A8", "neutral_dark": "#606060",
    "ink": "#272727", "delta_up": "#2E9E44", "delta_down": "#E53935",
}
BASELINE_SHADES = [NMI["baseline_dark"], NMI["baseline_mid"], NMI["baseline_soft"]]
OURS_SHADES     = [NMI["ours_deep"], NMI["ours_base"], NMI["ours_soft"]]

def apply_style(font_size=8, axes_linewidth=0.9):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",     # editable SVG text
        "pdf.fonttype": 42,         # editable PDF text
        "font.size": font_size,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": axes_linewidth, "axes.edgecolor": "#444444",
        "legend.frameon": False, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    })

def panel_label(ax, s, x=-0.12, y=1.04, fs=11):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="bottom", ha="left")

def finalize(fig, out_base, formats=("svg", "pdf", "png"), dpi=600, pad=0.4):
    fig.tight_layout(pad=pad)
    base = Path(out_base); os.makedirs(base.parent, exist_ok=True)
    saved = []
    for f in formats:
        p = f"{base}.{f}"; fig.savefig(p, dpi=dpi, bbox_inches="tight"); saved.append(p)
    plt.close(fig)
    return saved
