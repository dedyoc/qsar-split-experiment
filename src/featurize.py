"""ECFP4 fingerprints (r=2, 2048 bits) + Bemis-Murcko scaffolds.
Acyclic molecules get scaffold "" -> one group in the scaffold split."""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

import config

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=config.FP_RADIUS,
                                                 fpSize=config.FP_NBITS)


def bitvects(smiles):
    return [_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def to_numpy(fps):
    X = np.zeros((len(fps), config.FP_NBITS), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def scaffold(smi):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi))


def main():
    df = pd.read_csv(config.CLEAN_FILE)
    df["scaffold"] = df.smiles.map(scaffold)
    X = to_numpy(bitvects(df.smiles))
    df.to_csv(config.CLEAN_FILE, index=False)
    np.savez_compressed(config.FP_FILE, X=X)
    print(f"[featurize] X={X.shape}, {df.scaffold.nunique()} unique scaffolds "
          f"({(df.scaffold == '').sum()} acyclic compounds)")


def load():
    df = pd.read_csv(config.CLEAN_FILE, keep_default_na=False,
                     na_values={"year": [""], "pIC50_sd": [""]})
    X = np.load(config.FP_FILE)["X"]
    return df, X, bitvects(df.smiles)


if __name__ == "__main__":
    main()
