"""Step 6 - Test-to-train similarity analysis.

max_tanimoto(): for each test compound, the highest Tanimoto similarity
(ECFP4) to any training compound, i.e. similarity to its nearest training
neighbour. Called from train_eval.py for every run, so each prediction in
results/predictions.csv carries its own max_sim.

main(): from predictions.csv computes, per split,
  - median max_sim of the test set (mean ± SD over seeds)
  - RMSE in max_sim bins 0-0.3, 0.3-0.5, 0.5-0.7, 0.7-1.0 (per run, then
    mean ± SD over seeds; bins with < 5 compounds in a run are not scored
    for that run)
and draws the distribution and RMSE-by-bin plots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs

import config

MIN_PER_BIN = 5
COLORS = {"random": "#4C72B0", "scaffold": "#DD8452", "time": "#55A868"}


def max_tanimoto(test_fps, train_fps):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in test_fps])


def assign_bins(max_sim):
    edges = config.SIM_BINS[:-1] + [config.SIM_BINS[-1] + 1e-9]  # include 1.0 in last bin
    return pd.cut(max_sim, edges, right=False, labels=config.SIM_BIN_LABELS)


def rmse_by_bin(pred):
    pred = pred.assign(bin=assign_bins(pred.max_sim))
    per_run = (pred.groupby(["split", "seed", "bin"], observed=False)
                   .apply(lambda g: pd.Series({
                       "n": len(g),
                       "rmse": np.sqrt(np.mean((g.y_true - g.y_pred) ** 2))
                               if len(g) >= MIN_PER_BIN else np.nan}),
                          include_groups=False)
                   .reset_index())
    summary = (per_run.groupby(["split", "bin"], observed=False)
                      .agg(n_mean=("n", "mean"), rmse_mean=("rmse", "mean"),
                           rmse_sd=("rmse", "std"), runs_scored=("rmse", "count"))
                      .reset_index())
    return per_run, summary


def plot_distribution(pred, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in config.SPLITS:
        s = pred[pred.split == split].max_sim
        ax.hist(s, bins=np.linspace(0, 1, 41), histtype="step", lw=2, density=True,
                color=COLORS[split], label=f"{split} (median {s.median():.2f})")
    for edge in config.SIM_BINS[1:-1]:
        ax.axvline(edge, color="grey", ls=":", lw=1)
    ax.set(xlabel="Max Tanimoto similarity of test compound to training set",
           ylabel="Density (pooled over 5 seeds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rmse_by_bin(summary, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(config.SIM_BIN_LABELS))
    w = 0.27
    for i, split in enumerate(config.SPLITS):
        s = summary[summary.split == split].set_index("bin").reindex(config.SIM_BIN_LABELS)
        ax.bar(x + (i - 1) * w, s.rmse_mean, w, yerr=s.rmse_sd, capsize=3,
               color=COLORS[split], label=split)
    ax.set(xticks=x, xticklabels=config.SIM_BIN_LABELS,
           xlabel="Max Tanimoto similarity to training set", ylabel="Test RMSE (pIC50)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    pred = pd.read_csv(config.RESULTS_DIR / "predictions.csv")
    per_run, summary = rmse_by_bin(pred)
    per_run.to_csv(config.RESULTS_DIR / "rmse_by_similarity_bin_per_run.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "rmse_by_similarity_bin.csv", index=False)
    plot_distribution(pred, config.RESULTS_DIR / "maxsim_distribution.png")
    plot_rmse_by_bin(summary, config.RESULTS_DIR / "rmse_by_simbin.png")
    print("[similarity]\n" + summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
