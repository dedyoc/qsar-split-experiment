# QSAR split experiment — random vs scaffold vs time split

This repo holds the code for Section 6 of the report *"Hiệu năng giảm khi chuyển từ random split sang
scaffold/time split nói lên điều gì?"* (`docs/`). It measures the **generalization gap** of one fixed
QSAR model (ECFP4 + random forest) on EGFR IC50 data from ChEMBL under three ways of splitting the
data, and fills in **Bảng 3**.

## Quick start

```bash
conda env create -f environment.yml      # first time only
conda activate qsar
python src/run_all.py                    # full pipeline
```

Each step can also be run alone in order, e.g. `python src/train_eval.py`. The ChEMBL download is
cached in `data/raw/`. Running `python src/fetch.py --force` downloads it again.

## Pipeline

| Step | Script | Input → output |
|---|---|---|
| 1. Fetch | `src/fetch.py` | ChEMBL REST `/activity` (CHEMBL203, IC50) → `data/raw/chembl203_ic50.csv` |
| 2. Clean | `src/clean.py` | raw → `data/processed/egfr_clean.csv`, `results/cleaning_log.csv` |
| 3. Featurize | `src/featurize.py` | adds a `scaffold` column, writes ECFP4 matrix `data/processed/fps.npz` |
| 4. Split | `src/splits.py` | library: `random_split`, `scaffold_split`, `time_split` |
| 5. Train/eval | `src/train_eval.py` | 3 splits × 5 seeds → `results/metrics_per_run.csv`, `results/predictions.csv` |
| 6. Similarity | `src/similarity.py` | → `results/rmse_by_similarity_bin*.csv`, `maxsim_distribution.png`, `rmse_by_simbin.png` |
| 7. Report | `src/report.py` | → `results/bang3.csv`, `results/bang3.md`, `split_info.csv`, `rmse_r2_by_split.png` |

All fixed parameters are in `src/config.py`.

## Methods

**Data.** All ChEMBL activity records for target CHEMBL203 (EGFR, human) with `standard_type = IC50`.
A record is kept if `standard_relation` is `=`, the units are nM, the value is > 0, and ChEMBL has
set no `data_validity_comment`. Censored values (`>`, `<`) are removed. `cleaning_log.csv` gives the
number of records left after each filter.

**Structure standardization.** RDKit `rdMolStandardize`: Cleanup → largest fragment (removes
salts/solvents) → neutralize → canonical SMILES. Tautomers are not canonicalized.

**Target.** pIC50 = 9 − log10(IC50 [nM]). Measurements of the same standardized structure are
merged into one compound using the **median** pIC50. The compound's `year` is the **earliest**
`document_year` among its measurements, i.e. when it first appeared in the literature.

**Features.** Morgan fingerprint, radius 2, 2048 bits (ECFP4), as binary bits.

**Model.** `RandomForestRegressor(n_estimators=500)` with every other setting at the scikit-learn
default. The model is identical for every split, and the seed only sets `random_state`. No
hyperparameters are tuned for any split.

**Splits** (target 80/20, seeds 0–4):

- **Random:** compounds are shuffled one by one.
- **Scaffold:** compounds are grouped by Bemis–Murcko scaffold. The groups are shuffled with the
  seed, and each group is added to test only if it still fits under 20% of the data. Everything
  left goes to train, so no scaffold is ever in both sets. As a result, very large scaffolds tend
  to end up in train. Acyclic compounds share the empty scaffold and form one group.
- **Time:** only compounds that have a `year`. Test holds every compound first published in or
  after a cutoff year Y, where Y is the year that makes the test fraction closest to 20%. This
  split is **deterministic**, so its SD across the 5 seeds reflects only the random forest's
  randomness. Random and scaffold use the full dataset. The number of compounds dropped for a
  missing year, and the cutoff year, are in `results/split_info.csv` and the notes under Bảng 3.

**Metrics** (on the test set, mean ± sample SD over 5 seeds):

- RMSE and R² of the pIC50 predictions.
- **Max Tanimoto test→train:** for each test compound, its ECFP4 Tanimoto similarity to the
  closest training compound. Each run reports the median over its test set.
- **RMSE by similarity bin:** RMSE for test compounds in the max-similarity bins [0, 0.3),
  [0.3, 0.5), [0.5, 0.7), [0.7, 1.0], computed per run and then averaged. A bin with fewer than 5
  compounds in a run is not scored for that run (`runs_scored` gives the count).
- **ΔRMSE vs random:** mean RMSE of the split − mean RMSE of the random split.

## Caveats

- This uses one target, one featurization and one model. Treat it as an illustration of the
  mechanism, not as a general conclusion (see the report's FAQ).
- Because the time split is deterministic, its SD is not comparable with the SDs of the other two
  splits.
- The time split is evaluated on a slightly smaller dataset (only compounds with a year).
