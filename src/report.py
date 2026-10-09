"""Step 7 - Aggregate runs into Bảng 3 (report Section 6.3).

Columns (as in the report):
  Cách chia | RMSE (mean ± SD) | R² (mean ± SD) |
  Độ tương đồng trung vị test–train | Chênh lệch RMSE so với random

- mean ± SD are over the 5 seeds (sample SD, ddof=1).
- "Độ tương đồng trung vị" = per-run median of max Tanimoto test->train,
  then mean ± SD over seeds.
- ΔRMSE = mean RMSE(split) - mean RMSE(random); the % is relative to random.

Outputs results/bang3.csv, results/bang3.md (table + notes ready to paste),
results/split_info.csv and results/rmse_r2_by_split.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from featurize import load
from similarity import COLORS
from splits import time_split_info

LABELS = {"random": "Random", "scaffold": "Scaffold", "time": "Time"}


def pm(mean, sd, nd=3):
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def build_table(metrics):
    s = metrics.groupby("split").agg(["mean", "std"]).reindex(config.SPLITS)
    base = s.loc["random", ("rmse", "mean")]
    rows = []
    for split in config.SPLITS:
        r = s.loc[split]
        d = r[("rmse", "mean")] - base
        rows.append({
            "Cách chia": LABELS[split],
            "RMSE (trung bình ± SD)": pm(r[("rmse", "mean")], r[("rmse", "std")]),
            "R² (trung bình ± SD)": pm(r[("r2", "mean")], r[("r2", "std")]),
            "Độ tương đồng trung vị test–train": pm(r[("median_max_sim", "mean")],
                                                     r[("median_max_sim", "std")], 2),
            "Chênh lệch RMSE so với random": "—" if split == "random"
                                             else f"{d:+.3f} ({d / base:+.0%})",
            "n_train": int(r[("n_train", "mean")]), "n_test": int(r[("n_test", "mean")]),
        })
    return pd.DataFrame(rows)


def plot_metrics(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, col, label in zip(axes, ["rmse", "r2"], ["Test RMSE (pIC50)", "Test R²"]):
        for i, split in enumerate(config.SPLITS):
            v = metrics[metrics.split == split][col]
            ax.bar(i, v.mean(), yerr=v.std(), capsize=4, color=COLORS[split])
            ax.scatter([i] * len(v), v, color="k", s=10, zorder=3)
        ax.set(xticks=range(3), xticklabels=[LABELS[s] for s in config.SPLITS], ylabel=label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    metrics = pd.read_csv(config.RESULTS_DIR / "metrics_per_run.csv")
    log = pd.read_csv(config.RESULTS_DIR / "cleaning_log.csv")
    df, _, _ = load()
    info = time_split_info(df)
    pd.DataFrame([info]).to_csv(config.RESULTS_DIR / "split_info.csv", index=False)

    table = build_table(metrics)
    table.to_csv(config.RESULTS_DIR / "bang3.csv", index=False)
    plot_metrics(metrics, config.RESULTS_DIR / "rmse_r2_by_split.png")

    n_total = len(df)
    notes = [
        f"- Dữ liệu: ChEMBL {config.TARGET_CHEMBL_ID} (EGFR), IC50 '=' nM → pIC50; "
        f"{int(log.n_remaining.iloc[0])} bản ghi thô → {n_total} hợp chất duy nhất (gộp trung vị).",
        f"- Random/Scaffold dùng toàn bộ {n_total} hợp chất; Time split bỏ "
        f"{info['n_missing_year_dropped']} hợp chất thiếu document_year, "
        f"train < {info['cutoff_year']} ≤ test (năm {info['year_min']}–{info['year_max']}).",
        f"- ECFP4 (r=2, {config.FP_NBITS} bit); RandomForestRegressor "
        f"{config.RF_PARAMS}, giống nhau cho mọi cách chia; 5 seed {config.SEEDS}.",
        "- Time split là tất định: SD của Time chỉ phản ánh seed của random forest, "
        "không phản ánh biến thiên của cách chia.",
        "- Chênh lệch RMSE = RMSE trung bình (cách chia) − RMSE trung bình (Random).",
    ]
    bins = config.RESULTS_DIR / "rmse_by_similarity_bin.csv"
    if bins.exists():
        b = pd.read_csv(bins).set_index(["split", "bin"]).rmse_mean
        top = config.SIM_BIN_LABELS[-1]
        notes.append(
            f"- RMSE theo bin {top}: Random {b['random', top]:.2f}, Scaffold {b['scaffold', top]:.2f}, "
            f"Time {b['time', top]:.2f} — ở Time, sai số cao ngay cả khi test rất giống train, nên gap "
            "của Time không chỉ do độ tương đồng thấp (xem results/rmse_by_similarity_bin.csv).")
    diag = config.RESULTS_DIR / "time_diag_by_period.csv"
    if diag.exists():
        p = pd.read_csv(diag).set_index("period")
        notes.append(
            f"- Lưu ý (gây nhiễu cho Time split): CHEMBL203 gộp cả assay EGFR đột biến; tỉ lệ bản ghi "
            f"đột biến tăng từ {p.frac_mutant.iloc[0]:.0%} (≤2010) lên {p.frac_mutant.iloc[-1]:.0%} "
            "(2022+), nên tập test của Time một phần là endpoint khác (xem src/diagnose_time.py).")
    cols = [c for c in table.columns if not c.startswith("n_")]
    md = ("**Bảng 3. Kết quả thí nghiệm**\n\n" + table[cols].to_markdown(index=False)
          + "\n\n" + "\n".join(notes) + "\n")
    (config.RESULTS_DIR / "bang3.md").write_text(md, encoding="utf-8")
    print("[report]\n" + md)


if __name__ == "__main__":
    main()
