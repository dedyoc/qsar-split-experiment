"""Run the whole pipeline: fetch -> clean -> featurize -> train/eval -> similarity -> report.

Usage (from the repo root):
    conda activate qsar
    python src/run_all.py

The ChEMBL download is cached in data/raw/, so reruns do not hit the API.
"""
import clean
import featurize
import report
import similarity
import train_eval

if __name__ == "__main__":
    clean.main()        # calls fetch.load_raw(), cached
    featurize.main()
    train_eval.main()
    similarity.main()
    report.main()
