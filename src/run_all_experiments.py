"""
run_all_experiments.py

Runs train_one_experiment() (from train.py) on all three experiment folders
back to back. For each one, this produces inside that folder:
    - loss_curve.png
    - pr_curve.png
    - results_summary.csv
    - best_model.pt

Then prints a combined comparison table across all three so you can see the
baseline vs. raw-domain-drift vs. physics-domain-drift numbers side by side.

Usage:
    python run_all_experiments.py
"""

import pandas as pd

from M_Train import train_one_experiment

EXPERIMENT_DIRS = [
    "../experiment_1_baseline",
    "../experiment_2_raw_features",
    "../experiment_3_physics_features",
]


def main():
    all_results = []
    for exp_dir in EXPERIMENT_DIRS:
        print(f"\n{'='*70}\nRunning {exp_dir}\n{'='*70}")
        results = train_one_experiment(exp_dir)
        all_results.append(results)

    summary_df = pd.DataFrame(all_results)
    print(f"\n{'='*70}\nCombined comparison across all 3 experiments\n{'='*70}")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("../all_experiments_summary.csv", index=False)
    print("\nSaved combined summary -> ../all_experiments_summary.csv")


if __name__ == "__main__":
    main()