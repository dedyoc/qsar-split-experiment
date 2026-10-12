"""Figure - how the three split strategies assign compounds to train/test.

Three rows, one per split in splits.py:
  Random   : schematic; each compound is drawn individually, so the same
             scaffold (grey box) ends up in both train and test.
  Scaffold : schematic; whole scaffold groups go to test.
  Time     : real data; compounds per first-publication year, train < cutoff
             <= test (cutoff and dropped-year count from split_info.csv).
Each row is annotated with that split's real results from bang3.csv.
Labels are in Vietnamese to match the report.

Output: results/split_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import config

TRAIN, TEST = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated palette)
INK, MUTED, BOX = "#0b0b0b", "#52514e", "#e6e5e0"

GROUP_SIZES = [7, 5, 6, 4, 6]       # schematic scaffold groups A-E
SCAFFOLD_TEST_GROUPS = {1, 4}       # B and E go to test as whole groups
RANDOM_TEST = {(0, 2), (1, 0), (2, 4), (3, 1), (4, 3), (0, 5)}  # (group, member) ~20%


def draw_groups(ax, is_test):
    """Scaffold boxes with one dot per compound; is_test(g, m) -> bool."""
    x0 = 0.0
    for g, n in enumerate(GROUP_SIZES):
        cols = int(np.ceil(n / 2))
        w = cols * 0.5
        ax.add_patch(FancyBboxPatch((x0 - 0.3, -0.35), w + 0.1, 1.2,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=BOX, ec="none"))
        for m in range(n):
            ax.scatter(x0 + (m // 2) * 0.5, (m % 2) * 0.5, s=110, zorder=3,
                       color=TEST if is_test(g, m) else TRAIN, edgecolor="white", lw=1.5)
        ax.text(x0 - 0.05 + (cols - 1) * 0.25, -0.62, f"Scaffold {'ABCDE'[g]}",
                ha="center", va="center", fontsize=9, color=MUTED)
        x0 += w + 0.6
    ax.set(xlim=(-0.5, x0 - 0.4), ylim=(-0.85, 1.0))
    ax.axis("off")


def annotate(ax, title, how, row):
    ax.axis("off")
    ax.text(0, 0.95, title, fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0, 0.72, how, fontsize=9.5, color=MUTED, va="top", linespacing=1.5)
    stats = (f"n test = {row.n_test}\n"
             f"Tanimoto trung vị test→train = {row['Độ tương đồng trung vị test–train'].split(' ')[0]}\n"
             f"RMSE = {row['RMSE (trung bình ± SD)']}\n"
             f"R² = {row['R² (trung bình ± SD)']}")
    ax.text(0, 0.30, stats, fontsize=9.5, color=INK, va="top", linespacing=1.5)


def main():
    bang3 = pd.read_csv(config.RESULTS_DIR / "bang3.csv").set_index("Cách chia")
    info = pd.read_csv(config.RESULTS_DIR / "split_info.csv").iloc[0]
    years = pd.read_csv(config.CLEAN_FILE).year.dropna().astype(int)
    cutoff = int(info.cutoff_year)

    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1], hspace=0.45, wspace=0.05,
                          left=0.07, right=0.98, top=0.90, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    draw_groups(ax, lambda g, m: (g, m) in RANDOM_TEST)
    annotate(fig.add_subplot(gs[0, 1]), "Random split",
             "Từng hợp chất được gán ngẫu nhiên.\nCùng một scaffold xuất hiện ở cả\ntrain và test → nội suy.",
             bang3.loc["Random"])

    ax = fig.add_subplot(gs[1, 0])
    draw_groups(ax, lambda g, m: g in SCAFFOLD_TEST_GROUPS)
    annotate(fig.add_subplot(gs[1, 1]), "Scaffold split (Bemis–Murcko)",
             "Cả nhóm scaffold vào cùng một tập.\nScaffold của test chưa từng có trong\ntrain → ngoại suy khung mới.",
             bang3.loc["Scaffold"])

    ax = fig.add_subplot(gs[2, 0])
    counts = years.value_counts().sort_index()
    yrs = np.arange(counts.index.min(), counts.index.max() + 1)
    counts = counts.reindex(yrs, fill_value=0)
    ax.bar(yrs, counts, width=0.8, color=[TEST if y >= cutoff else TRAIN for y in yrs])
    ax.axvline(cutoff - 0.5, color=INK, lw=1.2, ls="--")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.text(cutoff - 0.5, counts.max() * 1.13, f"mốc {cutoff}", ha="center", va="top",
            fontsize=9, color=INK, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_ylabel("Số hợp chất", fontsize=9, color=MUTED)
    ax.set_xlabel("Năm công bố đầu tiên (document_year)", fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    annotate(fig.add_subplot(gs[2, 1]), "Time split",
             f"Train: năm < {cutoff}; test: năm ≥ {cutoff}.\n"
             f"Bỏ {int(info.n_missing_year_dropped)} hợp chất thiếu năm.\n"
             "Mô phỏng dự đoán tiến cứu.",
             bang3.loc["Time"])

    for color, label, x in [(TRAIN, "Train", 0.04), (TEST, "Test", 0.12)]:
        fig.patches.append(FancyBboxPatch((x, 0.945), 0.014, 0.018, transform=fig.transFigure,
                                          boxstyle="round,pad=0,rounding_size=0.003",
                                          fc=color, ec="none"))
        fig.text(x + 0.02, 0.954, label, fontsize=10, color=INK, va="center")
    fig.text(0.98, 0.954, f"EGFR (CHEMBL203), {len(pd.read_csv(config.CLEAN_FILE))} hợp chất · "
             "RMSE/R²: trung bình ± SD qua 5 seed", fontsize=9, color=MUTED,
             ha="right", va="center")

    out = config.RESULTS_DIR / "split_diagram.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"[plot_split_diagram] -> {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
