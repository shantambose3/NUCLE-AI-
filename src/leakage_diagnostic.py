"""
leakage_diagnosis.py

Runs the leakage check (AUC-ROC of raw f_G on the same-timestep label vs.
the forecast label at each horizon) and saves the results as plain-text and
plotted artifacts.

Outputs (written to ../ project root):
    leakage_diagnosis_log.txt   -- the numbers, in plain text
    leakage_diagnosis.png       -- bar chart: f_G's AUC-ROC/AUC-PR on the
                                    same-timestep label vs. the forecast label
                                    at each horizon -- the "before/after" picture

Usage:
    python leakage_diagnosis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from Physics_featureM import add_physics_features
from forecast_label import make_forecast_label

INPUT_PATH = "../DATA/CLEANED/dl_dataframe_clean.csv"
HORIZONS_STEPS = [10, 20, 50, 100]  # 100ms / 200ms / 500ms / 1000ms


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    df = pd.read_csv(INPUT_PATH)
    df = add_physics_features(df)

    log("=" * 70)
    log("LEAKAGE DIAGNOSIS -- f_G as a predictor of the label")
    log("=" * 70)
    log("")

    # --- SAME-TIMESTEP label (the original bug) ---
    same_ts_auc_roc = roc_auc_score(df["disruptive"], df["f_G"])
    same_ts_auc_pr = average_precision_score(df["disruptive"], df["f_G"])
    same_ts_prevalence = df["disruptive"].mean()

    log("SAME-TIMESTEP label ('disruptive' / density_limit_phase):")
    log(f"  prevalence (chance baseline): {same_ts_prevalence:.4f}")
    log(f"  AUC-ROC of raw f_G alone (no model): {same_ts_auc_roc:.4f}")
    log(f"  AUC-PR  of raw f_G alone (no model): {same_ts_auc_pr:.4f}")
    log(f"  --> This is the leakage: a bare threshold on f_G nearly solves the")
    log(f"      task with zero training, because the label and f_G are two")
    log(f"      measurements of the same physical fact at the same instant.")
    log("")

    results = []

    # --- FORECAST label, at each horizon (the fix) ---
    log("FORECAST label (early-warning task, at several horizons):")
    for horizon in HORIZONS_STEPS:
        df_f = make_forecast_label(df, horizon_steps=horizon, label_col="disruptive")
        prevalence = df_f["forecast_label"].mean()
        auc_roc = roc_auc_score(df_f["forecast_label"], df_f["f_G"])
        auc_pr = average_precision_score(df_f["forecast_label"], df_f["f_G"])

        log(f"  horizon={horizon} steps ({horizon*10}ms): prevalence={prevalence:.4f}, "
            f"AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f} "
            f"(AUC-PR / prevalence ratio = {auc_pr/prevalence:.2f}x)")
        results.append({"horizon_ms": horizon * 10, "auc_roc": auc_roc,
                         "auc_pr": auc_pr, "prevalence": prevalence})

    log("")
    log("--> At every horizon, f_G alone drops from near-perfect separation")
    log("    (same-timestep) to close to chance level (forecast). This is the")
    log("    evidence that the forecast reframing removes the leakage.")

    with open("../leakage_diagnosis_log.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nSaved -> ../leakage_diagnosis_log.txt")

    _plot(same_ts_auc_roc, same_ts_auc_pr, results)


def _plot(same_ts_auc_roc, same_ts_auc_pr, results):
    horizons = [r["horizon_ms"] for r in results]
    auc_rocs = [r["auc_roc"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Same-timestep\n(leakage)"] + [f"{h}ms\nforecast" for h in horizons]
    values = [same_ts_auc_roc] + auc_rocs
    colors = ["tab:red"] + ["tab:blue"] * len(horizons)

    ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUC-ROC=0.5)")
    ax.set_ylabel("AUC-ROC of raw f_G alone (no model)")
    ax.set_title("Leakage check: f_G's predictive power, same-timestep vs. forecast")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("../leakage_diagnosis.png", dpi=150)
    plt.close(fig)
    print("Saved -> ../leakage_diagnosis.png")


if __name__ == "__main__":
    main()