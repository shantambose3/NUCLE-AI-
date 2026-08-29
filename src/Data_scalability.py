"""
data_scalability.py

Reproduces the "Baseline (No Physics) vs. Constrained (Physics)" scalability
comparison: trains both arms repeatedly on increasing subsets of the
training data, with multiple random seeds per size, and plots mean +/- std.

Arms (already built by Split.py):
    Baseline (No Physics)  -> experiment_2_raw_features/
    Constrained (Physics)  -> experiment_3_physics_features/

Both arms share the SAME domain-drift shot split (train = low toroidal_B_field
shots, test = high toroidal_B_field shots) -- only the feature columns
differ. The test set is fixed and untouched throughout the whole sweep; only
the TRAINING subset size changes.

Total runs = len(SIZES) x len(SEEDS) x 2 arms = 7 x 5 x 2 = 70 short
training runs. Uses CUDA automatically if available (same as train.py).

Outputs (written to ../ project root):
    scalability_auc_pr.png
    scalability_f1.png
    scalability_summary.csv

Usage:
    python data_scalability.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from M_Train import ShotDataset, FocalLoss, run_epoch, compute_metrics, get_device
from CORE_MODEL import PlasmaMCAT

ARMS = {
    "Baseline (No Physics)": "../experiment_2_raw_features",
    "Constrained (Physics)": "../experiment_3_physics_features",
}

SIZES = [40, 80, 150, 250, 400, 600, 770]   # training-shot counts to sweep (bounded by smaller arm's train pool)
SEEDS = [0, 1, 2, 3, 4]                     # 5 seeds per size, per arm
SWEEP_EPOCHS = 15                           # shorter than the full 30-epoch run -- this is a sweep, not a final result
BATCH_SIZE = 16
LR = 1e-3


def stratified_shot_subset(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Pick k shots out of df's shots, preserving the ratio of shots with at
    least one forecast-positive row where possible. Falls back to a plain
    random sample if k is too small to stratify."""
    shot_labels = df.groupby("shot_id")["forecast_label"].max()
    all_ids = shot_labels.index.to_numpy()

    if k >= len(all_ids):
        return df  # asking for more than we have -- just use everything

    try:
        subset_ids, _ = train_test_split(
            all_ids, train_size=k, random_state=seed, stratify=shot_labels.values
        )
    except ValueError:
        # not enough of one class to stratify at this size -- fall back to random
        rng = np.random.RandomState(seed)
        subset_ids = rng.choice(all_ids, size=k, replace=False)

    return df[df["shot_id"].isin(subset_ids)]


def train_one_run(train_df: pd.DataFrame, test_ds: ShotDataset, seed: int, device) -> dict:
    """Trains a fresh model on train_df for SWEEP_EPOCHS, evaluates on the
    fixed test_ds, and returns the best AUC-PR/F1 seen during this run."""
    torch.manual_seed(seed)

    train_ds = ShotDataset(df=train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PlasmaMCAT(
        sensor_channels=train_ds.sensor_channels,
        domain_channels=train_ds.domain_channels,
    ).to(device)
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best = {"auc_pr": -1.0, "best_f1": 0.0}
    for _ in range(SWEEP_EPOCHS):
        run_epoch(model, train_loader, criterion, optimizer, device)
        _, test_logits, test_labels = run_epoch(model, test_loader, criterion, None, device)
        metrics = compute_metrics(test_logits, test_labels)
        if metrics["auc_pr"] > best["auc_pr"]:
            best = metrics

    return best


def main():
    device = get_device()
    records = []
    run_count = 0
    total_runs = len(ARMS) * len(SIZES) * len(SEEDS)

    for arm_name, exp_dir in ARMS.items():
        train_df = pd.read_csv(os.path.join(exp_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(exp_dir, "test.csv"))
        test_ds = ShotDataset(df=test_df)  # fixed test set for this arm, reused across every size/seed

        for size in SIZES:
            for seed in SEEDS:
                run_count += 1
                subset_df = stratified_shot_subset(train_df, size, seed)
                print(f"[{run_count}/{total_runs}] arm={arm_name} size={size} seed={seed} "
                      f"(actual shots={subset_df['shot_id'].nunique()})")

                metrics = train_one_run(subset_df, test_ds, seed, device)
                records.append({
                    "arm": arm_name,
                    "size": size,
                    "seed": seed,
                    "auc_pr": metrics["auc_pr"],
                    "f1": metrics["best_f1"],
                })

    results_df = pd.DataFrame(records)
    results_df.to_csv("../scalability_raw_results.csv", index=False)

    # aggregate mean +/- std across seeds, per arm, per size
    summary = results_df.groupby(["arm", "size"]).agg(
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    ).reset_index()
    summary.to_csv("../scalability_summary.csv", index=False)
    print("\nSaved -> ../scalability_summary.csv, ../scalability_raw_results.csv")

    _plot_metric(summary, "auc_pr_mean", "auc_pr_std", "AUC-PR", "../scalability_auc_pr.png")
    _plot_metric(summary, "f1_mean", "f1_std", "F1 Score", "../scalability_f1.png")


def _plot_metric(summary: pd.DataFrame, mean_col: str, std_col: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Baseline (No Physics)": "tab:blue", "Constrained (Physics)": "tab:orange"}
    markers = {"Baseline (No Physics)": "o", "Constrained (Physics)": "s"}

    for arm_name in summary["arm"].unique():
        sub = summary[summary["arm"] == arm_name].sort_values("size")
        ax.errorbar(
            sub["size"], sub[mean_col], yerr=sub[std_col],
            label=arm_name, color=colors.get(arm_name), marker=markers.get(arm_name),
            capsize=3, markersize=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Data Size (samples)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Data Scalability: {ylabel} vs Training Set Size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()