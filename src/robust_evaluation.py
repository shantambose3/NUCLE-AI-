"""
robust_evaluation.py (v2 -- k-fold cross-validation)

Outputs (written to ../ project root):
    robust_eval_summary.csv        -- mean/std test AUC-PR/F1 AND mean val-test gap, per (arm, horizon)
    robust_eval_raw_results.csv    -- every individual fold run, including its own val/test gap
    robust_eval_auc_pr.png         -- test AUC-PR vs horizon, both arms, error bars
    robust_eval_gap.png            -- val/test AUC-PR gap vs horizon, both arms -- the reliability check

Usage:
    python robust_evaluation.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats

from Physics_featureM import add_physics_features
from forecast_label import make_forecast_label
from Split import shot_level_summary, make_domain_split
from M_Train import ShotDataset, FocalLoss, PlasmaMCAT, run_epoch, compute_metrics, get_device

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
INPUT_PATH = "../DATA/CLEANED/dl_dataframe_clean.csv"
HORIZONS_STEPS = [10, 20, 50, 100]   # 100ms / 200ms / 500ms / 1000ms at 10ms per step
N_FOLDS = 5
MAX_EPOCHS = 30
PATIENCE = 10           # raised from 5 -- avoid stopping on noisy small-validation-set fluctuations
BATCH_SIZE = 16
LR = 1e-3

ARMS = ["raw_features", "physics_features"]  # matches experiment_2 / experiment_3 feature choices


def build_arm_columns(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Selects the right columns for each arm, same logic as Split.py."""
    if arm == "raw_features":
        cols = ["shot_id", "time", "forecast_label",
                "density", "elongation", "minor_radius",
                "plasma_current", "toroidal_B_field", "triangularity"]
    elif arm == "physics_features":
        cols = ["shot_id", "time", "forecast_label",
                "elongation", "toroidal_B_field", "triangularity", "f_G"]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return df[cols]


def make_shot_kfold_splits(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = 0):
    """Stratified k-fold over shot_id (not rows), stratified by whether the
    shot has any positive forecast row. Returns a list of (train_ids, val_ids)
    tuples -- every shot appears in exactly one val fold across the list.
    """
    shot_pos = df.groupby("shot_id")["forecast_label"].max()
    all_ids = shot_pos.index.to_numpy()
    labels = shot_pos.values

    # StratifiedKFold needs at least n_folds examples of the minority class.
    n_pos = int(labels.sum())
    effective_folds = min(n_folds, max(2, n_pos))
    if effective_folds < n_folds:
        print(f"  WARNING: only {n_pos} positive shots available -- reducing to {effective_folds}-fold "
              f"(requested {n_folds}-fold) so every fold has at least one positive validation shot.")

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(all_ids, labels):
        splits.append((all_ids[train_idx], all_ids[val_idx]))
    return splits


def train_one_fold(train_df, val_df, test_ds, fold_seed, device):
    """Trains with early stopping on this fold's validation AUC-PR, then
    evaluates the selected checkpoint on the test set. Returns metrics
    including the val/test gap, which is the reliability signal to watch."""
    torch.manual_seed(fold_seed)

    train_ds = ShotDataset(df=train_df)
    val_ds = ShotDataset(df=val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PlasmaMCAT(
        sensor_channels=train_ds.sensor_channels,
        domain_channels=train_ds.domain_channels,
    ).to(device)
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc_pr = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(model, train_loader, criterion, optimizer, device)
        _, val_logits, val_labels = run_epoch(model, val_loader, criterion, None, device)
        val_metrics = compute_metrics(val_logits, val_labels)

        if val_metrics["auc_pr"] > best_val_auc_pr:
            best_val_auc_pr = val_metrics["auc_pr"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                break  # early stop -- validation hasn't improved in PATIENCE epochs

    model.load_state_dict(best_state)
    _, test_logits, test_labels = run_epoch(model, test_loader, criterion, None, device)
    test_metrics = compute_metrics(test_logits, test_labels)

    gap = best_val_auc_pr - test_metrics["auc_pr"]

    return {
        "best_epoch": best_epoch,
        "val_auc_pr": best_val_auc_pr,
        "test_auc_pr": test_metrics["auc_pr"],
        "test_f1": test_metrics["best_f1"],
        "val_test_gap": gap,  # large gap = val was an unreliable early-stopping signal for this fold
    }


def main():
    device = get_device()

    raw_df = pd.read_csv(INPUT_PATH)
    raw_df = add_physics_features(raw_df)
    summary = shot_level_summary(raw_df)  # computed once from the ORIGINAL disruptive column

    records = []
    baseline_records = []  # naive f_G-threshold baseline -- no training, no model, just the raw feature
    total_runs = len(HORIZONS_STEPS) * len(ARMS) * N_FOLDS
    run_count = 0

    for horizon in HORIZONS_STEPS:
        df_forecast = make_forecast_label(raw_df, horizon_steps=horizon, label_col="disruptive")

        # same domain-drift split (low B-field train, high B-field test) at this horizon
        train_full, test_full = make_domain_split(df_forecast, summary)

        # --- naive baseline: how well does f_G alone, with zero training, rank
        # the SAME test set the trained models are evaluated on? This is the
        # bar a trained model needs to clear to justify using it at all. ---
        baseline_auc_pr = average_precision_score(test_full["forecast_label"], test_full["f_G"])
        baseline_auc_roc = roc_auc_score(test_full["forecast_label"], test_full["f_G"])
        baseline_records.append({
            "horizon_ms": horizon * 10, "baseline_auc_pr": baseline_auc_pr, "baseline_auc_roc": baseline_auc_roc,
        })
        print(f"\n=== horizon={horizon} ({horizon*10}ms): naive f_G-threshold baseline on test set -- "
              f"AUC-PR={baseline_auc_pr:.4f}, AUC-ROC={baseline_auc_roc:.4f} (no training, no model) ===")

        for arm in ARMS:
            train_full_arm = build_arm_columns(train_full, arm)
            test_arm = build_arm_columns(test_full, arm)
            test_ds = ShotDataset(df=test_arm)  # fixed test set for this (horizon, arm)

            print(f"\n--- horizon={horizon} ({horizon*10}ms) arm={arm}: building {N_FOLDS}-fold CV splits ---")
            fold_splits = make_shot_kfold_splits(train_full_arm, n_folds=N_FOLDS, seed=0)

            for fold_idx, (train_ids, val_ids) in enumerate(fold_splits):
                run_count += 1
                train_part = train_full_arm[train_full_arm["shot_id"].isin(train_ids)]
                val_part = train_full_arm[train_full_arm["shot_id"].isin(val_ids)]

                print(f"[{run_count}/{total_runs}] horizon={horizon} arm={arm} fold={fold_idx} "
                      f"(train_shots={len(train_ids)}, val_shots={len(val_ids)})")

                result = train_one_fold(train_part, val_part, test_ds, fold_seed=fold_idx, device=device)
                result.update({"horizon_steps": horizon, "horizon_ms": horizon * 10, "arm": arm, "fold": fold_idx})
                result["beats_baseline"] = result["test_auc_pr"] > baseline_auc_pr
                records.append(result)
                print(f"  -> best_epoch={result['best_epoch']} val_auc_pr={result['val_auc_pr']:.4f} "
                      f"test_auc_pr={result['test_auc_pr']:.4f} val_test_gap={result['val_test_gap']:+.4f} "
                      f"beats_naive_baseline={result['beats_baseline']}")

    baseline_df = pd.DataFrame(baseline_records)
    baseline_df.to_csv("../naive_baseline_summary.csv", index=False)
    print("\nSaved -> ../naive_baseline_summary.csv")

    results_df = pd.DataFrame(records)
    results_df.to_csv("../robust_eval_raw_results.csv", index=False)

    summary_df = results_df.groupby(["arm", "horizon_ms"]).agg(
        test_auc_pr_mean=("test_auc_pr", "mean"), test_auc_pr_std=("test_auc_pr", "std"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        val_test_gap_mean=("val_test_gap", "mean"), val_test_gap_std=("val_test_gap", "std"),
    ).reset_index()
    summary_df = summary_df.merge(baseline_df, on="horizon_ms", how="left")
    summary_df["beats_baseline_by"] = summary_df["test_auc_pr_mean"] - summary_df["baseline_auc_pr"]
    summary_df.to_csv("../robust_eval_summary.csv", index=False)
    print("\nSaved -> ../robust_eval_summary.csv, ../robust_eval_raw_results.csv")

    print("\n=== Reliability check: mean val/test AUC-PR gap per (arm, horizon) ===")
    print("(large positive gap = validation was overly optimistic vs. test for that setting)")
    print(summary_df[["arm", "horizon_ms", "val_test_gap_mean", "val_test_gap_std"]].to_string(index=False))

    print("\n=== Trained model vs. naive f_G-threshold baseline (no training at all) ===")
    print("(positive 'beats_baseline_by' = the trained model earns its complexity; "
          "negative = a bare threshold on f_G alone would have done as well or better)")
    print(summary_df[["arm", "horizon_ms", "test_auc_pr_mean", "baseline_auc_pr", "beats_baseline_by"]]
          .to_string(index=False))

    # --- paired significance test: physics vs raw, matched by fold within each horizon ---
    sig_records = []
    for horizon in sorted(results_df["horizon_ms"].unique()):
        raw_vals = results_df[(results_df.arm == "raw_features") & (results_df.horizon_ms == horizon)] \
            .sort_values("fold")["test_auc_pr"].values
        phys_vals = results_df[(results_df.arm == "physics_features") & (results_df.horizon_ms == horizon)] \
            .sort_values("fold")["test_auc_pr"].values

        if len(raw_vals) != len(phys_vals) or len(raw_vals) < 2:
            continue  # can't pair reliably if fold counts differ (e.g. reduced-fold horizons)

        t_stat, t_p = stats.ttest_rel(phys_vals, raw_vals)
        try:
            w_stat, w_p = stats.wilcoxon(phys_vals, raw_vals)
        except ValueError:
            w_p = float("nan")  # happens if all differences are zero/identical

        sig_records.append({
            "horizon_ms": horizon,
            "mean_diff_physics_minus_raw": float(np.mean(phys_vals - raw_vals)),
            "n_folds_wins_for_physics": int((phys_vals > raw_vals).sum()),
            "n_folds_total": len(raw_vals),
            "paired_ttest_p": t_p,
            "wilcoxon_p": w_p,
        })

    sig_df = pd.DataFrame(sig_records)
    sig_df.to_csv("../paired_significance_test.csv", index=False)
    print("\n=== Paired significance test: physics vs. raw, matched by fold ===")
    print("(p < 0.05 = statistically supported advantage at that horizon; note n=5 folds limits power)")
    print(sig_df.to_string(index=False))
    print("Saved -> ../paired_significance_test.csv")

    _plot_metric_vs_horizon(summary_df, "test_auc_pr_mean", "test_auc_pr_std",
                             "Test AUC-PR", "Early-Warning Performance vs. Forecast Horizon",
                             "../robust_eval_auc_pr.png", baseline_df=baseline_df)
    _plot_metric_vs_horizon(summary_df, "val_test_gap_mean", "val_test_gap_std",
                             "Val - Test AUC-PR Gap", "Validation Reliability vs. Forecast Horizon",
                             "../robust_eval_gap.png")


def _plot_metric_vs_horizon(summary_df: pd.DataFrame, mean_col: str, std_col: str,
                             ylabel: str, title: str, out_path: str, baseline_df: pd.DataFrame = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"raw_features": "tab:blue", "physics_features": "tab:orange"}
    labels = {"raw_features": "Baseline (No Physics)", "physics_features": "Constrained (Physics)"}

    for arm in summary_df["arm"].unique():
        sub = summary_df[summary_df["arm"] == arm].sort_values("horizon_ms")
        ax.errorbar(
            sub["horizon_ms"], sub[mean_col], yerr=sub[std_col],
            label=labels.get(arm, arm), color=colors.get(arm), marker="o", capsize=3,
        )

    if baseline_df is not None:
        base_sorted = baseline_df.sort_values("horizon_ms")
        ax.plot(base_sorted["horizon_ms"], base_sorted["baseline_auc_pr"],
                 label="Naive f_G threshold (no training)", color="gray", linestyle="--", marker="x")

    if "gap" in out_path:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)  # zero gap = perfectly reliable validation

    ax.set_xlabel("Forecast Horizon (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()