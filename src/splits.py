"""Step 4 - Train/test splitting strategies.

Every function has the signature  split(df, seed) -> (train_idx, test_idx)
with positional indices into df, and targets TEST_FRACTION (20%) test.

random   : compounds shuffled individually (sklearn train_test_split).

scaffold : Bemis-Murcko scaffold groups are shuffled with `seed`, then
           greedily assigned to test as long as the group still fits under
           the 20% target; everything else goes to train. A scaffold is
           never in both sets. Greedy "fits under" (instead of "add until
           over") stops one very large scaffold from blowing up the test
           set when it is drawn early; large scaffolds therefore end up in
           train more often - the usual behaviour of balanced scaffold
           splits (cf. chemprop scaffold_balanced).

time     : compounds with a document_year only. Test = all compounds first
           published in year >= Y, where the cutoff Y is the year whose
           test fraction is closest to 20%. Deterministic: `seed` is
           ignored, so across the 5 runs only the RF seed varies.
           Compounds with missing year are dropped (count reported).
"""
import numpy as np
from sklearn.model_selection import train_test_split

import config


def random_split(df, seed):
    idx = np.arange(len(df))
    train, test = train_test_split(idx, test_size=config.TEST_FRACTION, random_state=seed)
    return np.sort(train), np.sort(test)


def scaffold_split(df, seed):
    groups = df.groupby("scaffold").indices  # scaffold -> array of positions
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    target = int(round(config.TEST_FRACTION * len(df)))
    test = []
    for k in keys:
        if len(test) + len(groups[k]) <= target:
            test.extend(groups[k])
        if len(test) == target:
            break
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(df)), test)
    return train, test


def time_cutoff(years):
    """Year Y such that fraction(year >= Y) is closest to TEST_FRACTION."""
    candidates = np.unique(years)
    fracs = np.array([(years >= y).mean() for y in candidates])
    return int(candidates[np.argmin(np.abs(fracs - config.TEST_FRACTION))])


def time_split(df, seed=None):
    has_year = df.year.notna().to_numpy()
    years = df.year.to_numpy()
    cutoff = time_cutoff(years[has_year])
    idx = np.arange(len(df))
    train = idx[has_year & (years < cutoff)]
    test = idx[has_year & (years >= cutoff)]
    return train, test


def time_split_info(df):
    """Facts about the time split for the report."""
    has_year = df.year.notna()
    cutoff = time_cutoff(df.year[has_year].to_numpy())
    return {"cutoff_year": cutoff,
            "n_missing_year_dropped": int((~has_year).sum()),
            "year_min": int(df.year.min()), "year_max": int(df.year.max())}


SPLITTERS = {"random": random_split, "scaffold": scaffold_split, "time": time_split}
