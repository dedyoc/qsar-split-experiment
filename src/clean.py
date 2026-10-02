"""Step 2 - Filter, standardize and aggregate the raw ChEMBL activities.

Filters (in order, each logged to results/cleaning_log.csv):
  1. missing SMILES or standard_value
  2. standard_type == IC50, standard_relation == "=", standard_units == nM
  3. standard_value > 0 (needed for the log)
  4. ChEMBL data_validity_comment set (e.g. "Outside typical range",
     "Potential transcription error") -> dropped as unreliable
  5. RDKit cannot parse / standardize the structure

Structure standardization (RDKit rdMolStandardize): Cleanup -> keep largest
fragment (strips salts/solvents) -> Uncharger -> canonical isomeric SMILES.
Tautomers are not canonicalized (slow, and Morgan bits are mostly robust to
the common cases).

pIC50 = 9 - log10(IC50 in nM). Duplicates are grouped by the standardized
SMILES; the compound gets the MEDIAN pIC50 and the EARLIEST document_year
(when the compound first appeared in the literature). Compounds without any
year keep year = NaN; only the time split drops them.
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

import config
from fetch import load_raw

RDLogger.DisableLog("rdApp.*")

_cleanup = rdMolStandardize.CleanupParameters()
_largest = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smi):
    """Return a canonical standardized SMILES, or None if RDKit fails."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol, _cleanup)
        mol = _largest.choose(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def clean(raw):
    log = [("raw activity records", len(raw))]
    df = raw.dropna(subset=["canonical_smiles", "standard_value"])
    log.append(("has SMILES and value", len(df)))

    df = df[(df.standard_type == config.STANDARD_TYPE)
            & (df.standard_relation == "=")
            & (df.standard_units == "nM")]
    log.append(("IC50, relation '=', units nM", len(df)))

    df = df[df.standard_value.astype(float) > 0]
    log.append(("value > 0", len(df)))

    df = df[df.data_validity_comment.isna()]
    log.append(("no ChEMBL validity flag", len(df)))

    df = df.assign(smiles=df.canonical_smiles.map(standardize_smiles)).dropna(subset=["smiles"])
    log.append(("RDKit-standardizable", len(df)))

    df = df.assign(pIC50=9 - np.log10(df.standard_value.astype(float)))
    agg = (df.groupby("smiles")
             .agg(pIC50=("pIC50", "median"),
                  pIC50_sd=("pIC50", "std"),
                  n_measurements=("pIC50", "size"),
                  year=("document_year", "min"),  # NaN-skipping
                  molecule_chembl_ids=("molecule_chembl_id",
                                       lambda s: ";".join(sorted(set(s)))))
             .reset_index())
    log.append(("unique compounds (median-aggregated)", len(agg)))
    log.append(("  of which with document_year", int(agg.year.notna().sum())))
    return agg, pd.DataFrame(log, columns=["step", "n_remaining"])


def main():
    agg, log = clean(load_raw())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(config.CLEAN_FILE, index=False)
    log.to_csv(config.RESULTS_DIR / "cleaning_log.csv", index=False)
    print("[clean]\n" + log.to_string(index=False))
    return agg


if __name__ == "__main__":
    main()
