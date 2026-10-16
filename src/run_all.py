"""Whole pipeline. Run from repo root: python src/run_all.py"""
import clean
import diagnose_time
import featurize
import plot_seed_results
import plot_split_diagram
import report
import similarity
import train_eval

if __name__ == "__main__":
    clean.main()
    featurize.main()
    train_eval.main()
    similarity.main()
    diagnose_time.main()  # extra check
    report.main()
    plot_split_diagram.main()  # needs bang3.csv
    plot_seed_results.main()
