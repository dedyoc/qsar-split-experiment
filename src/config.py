"""Single source of truth for the experiment's fixed design.

Every module imports its parameters from here so that all three splits are
run with exactly the same data, features and model. Do not tune anything
per split.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# --- Data -------------------------------------------------------------------
TARGET_CHEMBL_ID = "CHEMBL203"  # EGFR (human)
STANDARD_TYPE = "IC50"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
RAW_FILE = RAW_DIR / f"{TARGET_CHEMBL_ID.lower()}_ic50.csv"
CLEAN_FILE = PROCESSED_DIR / "egfr_clean.csv"
FP_FILE = PROCESSED_DIR / "fps.npz"

# --- Features: Morgan ECFP4 -------------------------------------------------
FP_RADIUS = 2
FP_NBITS = 2048

# --- Model: identical for every split ----------------------------------------
RF_PARAMS = dict(n_estimators=500, n_jobs=-1)  # all else = sklearn defaults

# --- Evaluation -------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.2
SPLITS = ["random", "scaffold", "time"]
SIM_BINS = [0.0, 0.3, 0.5, 0.7, 1.0]
SIM_BIN_LABELS = ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]
