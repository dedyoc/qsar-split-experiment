"""Figure: per-seed RMSE / R2 / median similarity + RMSE by similarity bin -> results/seed_results.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

COLORS = {"random": "#2a78d6", "scaffold": "#eb6834", "time": "#1baf7a"}
LABELS = {"random": "Random", "scaffold": "Scaffold", "time": "Time"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEED_OFFSETS = np.linspace(-0.16, 0.16, len(config.SEEDS))


def style(ax, ylabel, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def per_seed(ax, metrics, col, ylabel, title, fmt):
    for i, split in enumerate(config.SPLITS):
        m = metrics[metrics.split == split].sort_values("seed")
        mean, sd = m[col].mean(), m[col].std()
        ax.bar(i, mean, width=0.6, color=COLORS[split], alpha=0.22)
        ax.errorbar(i, mean, yerr=sd, color=INK, capsize=5, lw=1.2, zorder=4)
        ax.scatter(i + SEED_OFFSETS, m[col], s=40, color=COLORS[split],
                   edgecolor="white", lw=1.2, zorder=5)
        pad = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        if mean >= 0:  # label goes below if mean is negative
            y, va = max(mean + sd, m[col].max()) + pad, "bottom"
        else:
            y, va = min(mean - sd, m[col].min()) - pad, "top"
        ax.text(i, y, f"{mean:{fmt}} ± {sd:{fmt}}", ha="center", va=va, fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(config.SPLITS)), [LABELS[s] for s in config.SPLITS])
    style(ax, ylabel, title)


def main():
    metrics = pd.read_csv(config.RESULTS_DIR / "metrics_per_run.csv")
    bins = pd.read_csv(config.RESULTS_DIR / "rmse_by_similarity_bin.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (a, b), (c, d) = axes

    a.set_ylim(0, 1.6)
    per_seed(a, metrics, "rmse", "RMSE (pIC50)", "A. RMSE theo từng seed", ".3f")
    b.set_ylim(-0.5, 0.95)
    per_seed(b, metrics, "r2", "R²", "B. R² theo từng seed", ".3f")
    c.set_ylim(0, 1.0)
    per_seed(c, metrics, "median_max_sim", "Tanimoto (ECFP4)",
             "C. Độ tương đồng trung vị test→train", ".2f")

    x = np.arange(len(config.SIM_BIN_LABELS))
    for k, split in enumerate(config.SPLITS):
        s = bins[bins.split == split].set_index("bin").reindex(config.SIM_BIN_LABELS)
        d.errorbar(x + (k - 1) * 0.06, s.rmse_mean, yerr=s.rmse_sd, color=COLORS[split], lw=2,
                   marker="o", ms=7, capsize=4, mec="white", mew=1.2, label=LABELS[split])
    d.set_xticks(x, config.SIM_BIN_LABELS)
    d.set_xlim(-0.3, x[-1] + 0.3)
    d.set_xlabel("Tanimoto lớn nhất với tập train", fontsize=9, color=MUTED)
    style(d, "RMSE (pIC50)", "D. RMSE theo khoảng độ tương đồng")
    d.legend(frameon=False, fontsize=9, loc="upper right")

    n = metrics.groupby("split")[["n_train", "n_test"]].first()
    fig.text(0.01, 0.99,
             "Kết quả qua 5 seed (0–4) · EGFR CHEMBL203 · ECFP4 + RandomForest (500 cây)\n"
             "train/test: " + " · ".join(f"{LABELS[s]} {n.loc[s, 'n_train']}/{n.loc[s, 'n_test']}"
                                          for s in config.SPLITS),
             fontsize=9, color=MUTED, va="top", linespacing=1.5)
    fig.text(0.01, 0.012, "Chấm = từng seed; cột = trung bình; râu = ± SD. Time split là tất định "
             "nên 5 chấm gần trùng nhau (chỉ seed của random forest thay đổi).",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=2.5, w_pad=2.5)
    out = config.RESULTS_DIR / "seed_results.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"[plot_seed_results] -> {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
