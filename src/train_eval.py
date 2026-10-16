"""Train RF for each (split, seed), save metrics + predictions.
Same RF params for every split."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import config
from featurize import load
from similarity import max_tanimoto
from splits import SPLITTERS


def run_one(df, X, fps, split, seed):
    train, test = SPLITTERS[split](df, seed)
    y = df.pIC50.to_numpy()
    model = RandomForestRegressor(**config.RF_PARAMS, random_state=seed)
    model.fit(X[train], y[train])
    y_pred = model.predict(X[test])
    max_sim = max_tanimoto([fps[i] for i in test], [fps[i] for i in train])
    metrics = {"split": split, "seed": seed, "n_train": len(train), "n_test": len(test),
               "rmse": np.sqrt(mean_squared_error(y[test], y_pred)),
               "r2": r2_score(y[test], y_pred),
               "median_max_sim": float(np.median(max_sim))}
    pred = pd.DataFrame({"split": split, "seed": seed, "smiles": df.smiles.to_numpy()[test],
                         "y_true": y[test], "y_pred": y_pred, "max_sim": max_sim})
    return metrics, pred


def main():
    df, X, fps = load()
    metrics, preds = [], []
    n_runs = len(config.SPLITS) * len(config.SEEDS)
    for split in config.SPLITS:
        for seed in config.SEEDS:
            t0 = time.time()
            m, p = run_one(df, X, fps, split, seed)
            metrics.append(m)
            preds.append(p)
            # save every run so a crash doesn't lose finished ones
            pd.DataFrame(metrics).to_csv(config.RESULTS_DIR / "metrics_per_run.csv", index=False)
            pd.concat(preds).to_csv(config.RESULTS_DIR / "predictions.csv", index=False)
            print(f"[train_eval] run {len(metrics)}/{n_runs} {split:8s} seed={seed} "
                  f"n_train={m['n_train']} n_test={m['n_test']} RMSE={m['rmse']:.3f} "
                  f"R2={m['r2']:.3f} median_maxsim={m['median_max_sim']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("[train_eval] done", flush=True)


if __name__ == "__main__":
    main()
