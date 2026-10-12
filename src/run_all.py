"""Run the whole pipeline: fetch -> clean -> featurize -> train/eval -> similarity
-> (supplementary) time-split diagnostics -> report.

Usage (from the repo root):
    conda activate qsar
    python src/run_all.py

The ChEMBL download is cached in data/raw/, so reruns do not hit the API.
"""
import clean
import diagnose_time
import featurize
import plot_seed_results
import plot_split_diagram
import report
import similarity
import train_eval

if __name__ == "__main__":
    clean.main()        # calls fetch.load_raw(), cached
    featurize.main()
    train_eval.main()
    similarity.main()
    diagnose_time.main()  # supplementary; cached assay annotations
    report.main()
    plot_split_diagram.main()  # figures for the report (need bang3.csv)
    plot_seed_results.main()
