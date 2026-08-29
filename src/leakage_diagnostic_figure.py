"""
leakage_diagnostic_figure.py

Produces the leakage-diagnostic figure used in the accompanying report:
the discriminative power (AUC-ROC) of a bare, untrained threshold on raw
f_G, measured against two different labels:

  1. The dataset's native same-timestep `disruptive` column (the leakage
     case). Because f_G = n / (Ip / pi*a^2) and the density-limit phase is
     defined as a region of high f_G, thresholding f_G alone at the SAME
     timestep as the label is close to circular -- expect AUC-ROC near
     0.98.

  2. The genuine forecasting label (forecast_label.make_forecast_label) at
     each horizon in {100, 200, 500, 1000} ms. Because this label asks
     "does the shot enter the density-limit phase in the NEXT H ms", and
     f_G is only measured at the CURRENT timestep, a bare f_G threshold can
     no longer trivially satisfy the label -- but if f_G is a genuine
     physical precursor, it should still carry real signal above chance,
     just less than the same-timestep case, and decaying as the horizon
     lengthens (the further out you're asked to forecast, the less any
     single current-timestep reading can tell you).

This script computes both, tabulates them, and produces the bar-chart
figure.

Usage:
    python leakage_diagnostic_figure.py
    python leakage_diagnostic_figure.py --horizons 10 20 50 100

Outputs (written to project root, i.e. one level up from CODE/):
    leakage_diagnostic_results.csv
    fig1_leakage_diagnostic.png
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from Physics_featureM import add_physics_features
from forecast_label import make_forecast_label

INPUT_PATH = "../DATA/CLEANED/dl_dataframe_clean.csv"
DEFAULT_HORIZONS_STEPS = [10, 20, 50, 100]  # -> 100/200/500/1000 ms at 10ms/step
FEATURE_COL = "f_G"


def compute_leakage_auc_roc(input_path: str, horizons_steps: list,
                             feature_col: str = FEATURE_COL) -> pd.DataFrame:
    raw_df = pd.read_csv(input_path)
    raw_df = add_physics_features(raw_df)

    records = []

    # --- Case 1: same-timestep label (the leakage case) ---
    same_timestep_auc = roc_auc_score(raw_df["disruptive"], raw_df[feature_col])
    records.append({
        "label_type": "same_timestep_leakage",
        "horizon_ms": 0,
        "auc_roc": same_timestep_auc,
        "n_rows": len(raw_df),
        "positive_rate": raw_df["disruptive"].mean(),
    })
    print(f"[leakage case] same-timestep AUC-ROC of raw {feature_col}: {same_timestep_auc:.4f} "
          f"(n={len(raw_df)}, positive_rate={raw_df['disruptive'].mean():.4f})")

    # --- Case 2: genuine forecasting label at each horizon ---
    for horizon_steps in horizons_steps:
        horizon_ms = horizon_steps * 10
        df_forecast = make_forecast_label(raw_df, horizon_steps=horizon_steps, label_col="disruptive")

        auc = roc_auc_score(df_forecast["forecast_label"], df_forecast[feature_col])
        records.append({
            "label_type": "forecast",
            "horizon_ms": horizon_ms,
            "auc_roc": auc,
            "n_rows": len(df_forecast),
            "positive_rate": df_forecast["forecast_label"].mean(),
        })
        print(f"[forecast, {horizon_ms}ms] AUC-ROC of raw {feature_col}: {auc:.4f} "
              f"(n={len(df_forecast)}, positive_rate={df_forecast['forecast_label'].mean():.4f})")

    return pd.DataFrame(records)


def plot_leakage_diagnostic(results_df: pd.DataFrame, out_path: str) -> None:
    same_ts = results_df[results_df.label_type == "same_timestep_leakage"].iloc[0]
    forecast_rows = results_df[results_df.label_type == "forecast"].sort_values("horizon_ms")

    labels = ["Same-timestep\n(leakage)"] + [f"{int(h)}ms\nforecast" for h in forecast_rows["horizon_ms"]]
    values = [same_ts["auc_roc"]] + forecast_rows["auc_roc"].tolist()
    colors = ["tab:red"] + ["tab:blue"] * len(forecast_rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUC-ROC=0.5)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC-ROC of raw f_G alone (no model)")
    ax.set_title("Leakage check: f_G's predictive power, same-timestep vs. forecast")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS_STEPS,
                         help="Horizons in TIMESTEPS (10 steps = 100ms at this dataset's 10ms resolution). "
                              "Default: 10 20 50 100 -> 100/200/500/1000ms.")
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--out-csv", default="../leakage_diagnostic_results.csv")
    parser.add_argument("--out-fig", default="../fig1_leakage_diagnostic.png")
    args = parser.parse_args()

    results_df = compute_leakage_auc_roc(args.input, args.horizons)
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved -> {args.out_csv}")

    plot_leakage_diagnostic(results_df, args.out_fig)

    print("\n=== Summary ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
