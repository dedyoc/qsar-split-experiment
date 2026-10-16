"""split(df, seed) -> (train_idx, test_idx), ~20% test.

scaffold: shuffle scaffold groups, put a group in test only if it still fits
under 20%, so one huge scaffold can't blow up the test set.
time: test = year >= cutoff, cutoff chosen so test is closest to 20%.
Deterministic (seed ignored). Rows without year are dropped.
"""
import numpy as np
from sklearn.model_selection import train_test_split

import config


def random_split(df, seed):
    idx = np.arange(len(df))
    train, test = train_test_split(idx, test_size=config.TEST_FRACTION, random_state=seed)
    return np.sort(train), np.sort(test)


def scaffold_split(df, seed):
    groups = df.groupby("scaffold").indices
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
    has_year = df.year.notna()
    cutoff = time_cutoff(df.year[has_year].to_numpy())
    return {"cutoff_year": cutoff,
            "n_missing_year_dropped": int((~has_year).sum()),
            "year_min": int(df.year.min()), "year_max": int(df.year.max())}


SPLITTERS = {"random": random_split, "scaffold": scaffold_split, "time": time_split}
