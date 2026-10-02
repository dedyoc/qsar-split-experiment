"""Step 1 - Download EGFR (CHEMBL203) IC50 activities from the ChEMBL REST API.

The raw table is cached in data/raw/; if the file exists it is reused and no
request is made. Pass --force to re-download.

Only the columns needed downstream are requested. `document_year` is the
publication year of the source document and is what the time split uses.
"""
import argparse
import time

import pandas as pd
import requests

import config

FIELDS = [
    "activity_id", "molecule_chembl_id", "canonical_smiles",
    "standard_type", "standard_relation", "standard_value", "standard_units",
    "assay_chembl_id", "document_chembl_id", "document_year",
    "data_validity_comment",
]
PAGE_SIZE = 1000


def _get(url, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def download():
    """Page through /activity and return all records as a DataFrame."""
    url = f"{config.CHEMBL_API}/activity.json"
    params = {
        "target_chembl_id": config.TARGET_CHEMBL_ID,
        "standard_type": config.STANDARD_TYPE,
        "only": ",".join(FIELDS),
        "limit": PAGE_SIZE,
        "offset": 0,
    }
    rows = []
    page = _get(url, params)
    total = page["page_meta"]["total_count"]
    while True:
        rows.extend(page["activities"])
        print(f"  fetched {len(rows)}/{total}")
        nxt = page["page_meta"]["next"]
        if not nxt:
            break
        page = _get("https://www.ebi.ac.uk" + nxt)  # `next` is a path incl. query
    return pd.DataFrame(rows, columns=FIELDS)


def load_raw(force=False):
    """Return the raw activity table, downloading only if not cached."""
    if config.RAW_FILE.exists() and not force:
        print(f"[fetch] using cached {config.RAW_FILE.relative_to(config.ROOT)}")
        return pd.read_csv(config.RAW_FILE)
    print(f"[fetch] downloading {config.TARGET_CHEMBL_ID} {config.STANDARD_TYPE} from ChEMBL")
    df = download()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_FILE, index=False)
    print(f"[fetch] saved {len(df)} rows -> {config.RAW_FILE.relative_to(config.ROOT)}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    load_raw(force=ap.parse_args().force)
