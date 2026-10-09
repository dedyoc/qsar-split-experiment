"""Supplementary - why does the time split fail so badly? (not part of the fixed design)

CHEMBL203 includes assays on mutant EGFR (L858R, T790M, C797S, del19, ...)
as well as cell-based assays. Their share grows over time, so the time-split
test set (>= cutoff year) is partly a different endpoint from most of train.

This script fetches assay annotations once (cached in data/raw/), flags a
record as "mutant" if ChEMBL's variant_sequence names a mutation OR the
assay description mentions a common EGFR mutation, and writes:
  results/time_diag_by_period.csv  share of mutant / non-binding records
                                   and mean pIC50, per publication period
  results/time_diag_shift.csv      train vs test pIC50 shift and prediction
                                   bias for the time split (seed 0)
"""
import numpy as np
import pandas as pd
import requests

import config
from splits import time_cutoff

ASSAY_FILE = config.RAW_DIR / f"{config.TARGET_CHEMBL_ID.lower()}_assays.csv"
MUT_PATTERN = r"T790M|L858R|C797S|del19|exon 19|L861Q|G719|mutant"


def load_assays(assay_ids):
    if ASSAY_FILE.exists():
        return pd.read_csv(ASSAY_FILE)
    rows = []
    for i in range(0, len(assay_ids), 200):
        r = requests.get(f"{config.CHEMBL_API}/assay.json", timeout=120, params={
            "assay_chembl_id__in": ",".join(assay_ids[i:i + 200]),
            "only": "assay_chembl_id,variant_sequence,description,assay_type", "limit": 1000})
        r.raise_for_status()
        for a in r.json()["assays"]:
            v = a.get("variant_sequence") or {}
            rows.append({"assay_chembl_id": a["assay_chembl_id"], "mutation": v.get("mutation"),
                         "assay_type": a.get("assay_type"), "description": a.get("description")})
    df = pd.DataFrame(rows)
    df.to_csv(ASSAY_FILE, index=False)
    return df


def main():
    raw = pd.read_csv(config.RAW_FILE)
    d = raw[(raw.standard_type == config.STANDARD_TYPE) & (raw.standard_relation == "=")
            & (raw.standard_units == "nM") & (raw.standard_value > 0)
            & raw.data_validity_comment.isna()]
    d = d.merge(load_assays(sorted(d.assay_chembl_id.unique())), on="assay_chembl_id", how="left")
    d["mutant"] = ((d.mutation.notna() & (d.mutation != "UNDEFINED MUTATION"))
                   | d.description.str.contains(MUT_PATTERN, case=False, na=False))
    d["non_binding"] = d.assay_type != "B"
    d["pIC50"] = 9 - np.log10(d.standard_value)
    d["period"] = pd.cut(d.document_year, [0, 2010, 2015, 2018, 2021, 2100],
                         labels=["<=2010", "2011-15", "2016-18", "2019-21", "2022+"])
    by_period = (d.groupby("period", observed=True)
                  .agg(n_records=("pIC50", "size"), frac_mutant=("mutant", "mean"),
                       frac_non_binding=("non_binding", "mean"), mean_pIC50=("pIC50", "mean"))
                  .round(3).reset_index())
    by_period.to_csv(config.RESULTS_DIR / "time_diag_by_period.csv", index=False)

    clean = pd.read_csv(config.CLEAN_FILE).dropna(subset=["year"])
    cutoff = time_cutoff(clean.year.to_numpy())
    pred = pd.read_csv(config.RESULTS_DIR / "predictions.csv").query("split == 'time' and seed == 0")
    shift = pd.DataFrame([{
        "cutoff_year": cutoff,
        "train_mean_pIC50": clean[clean.year < cutoff].pIC50.mean(),
        "test_mean_pIC50": clean[clean.year >= cutoff].pIC50.mean(),
        "mean_bias_pred_minus_true": (pred.y_pred - pred.y_true).mean(),
        "pearson_r": np.corrcoef(pred.y_true, pred.y_pred)[0, 1],
        "rmse_after_removing_bias": np.sqrt((((pred.y_pred - pred.y_pred.mean())
                                              - (pred.y_true - pred.y_true.mean())) ** 2).mean()),
    }]).round(3)
    shift.to_csv(config.RESULTS_DIR / "time_diag_shift.csv", index=False)
    print("[diagnose_time]\n" + by_period.to_string(index=False) + "\n\n" + shift.T.to_string(header=False))


if __name__ == "__main__":
    main()
