"""
forecast_label.py

Fixes the label-leakage problem: 'disruptive' (density_limit_phase) is a
same-timestep measurement of density vs. the Greenwald limit -- and f_G is
literally that same ratio. Predicting 'disruptive' from 'f_G' at the same
timestep is close to circular: a bare threshold on f_G alone hits
AUC-ROC 0.976 with zero training (see leakage_diagnostic.py).

This module replaces that with an early-warning FORECASTING label:

    forecast_label[t] = 1  if the shot enters density_limit_phase at any
                            point in (t, t + horizon_steps]
                       = 0  otherwise

Rows that are ALREADY inside density_limit_phase are dropped -- there's
nothing to forecast once the event has already started. Only pre-phase
timesteps remain, and they're labeled by what happens next, not by what's
true right now.

This dataset is sampled at 10ms per timestep, so:
    horizon_steps=10  -> 100ms lead time
    horizon_steps=20  -> 200ms lead time
    horizon_steps=50  -> 500ms lead time

Usage:
    from forecast_label import make_forecast_label
    df_forecast = make_forecast_label(df, horizon_steps=20)
"""

import numpy as np
import pandas as pd


def make_forecast_label(df: pd.DataFrame, horizon_steps: int, label_col: str = "disruptive") -> pd.DataFrame:
    """
    Args:
        df: per-timestep dataframe with columns shot_id, time, and label_col
        horizon_steps: how many timesteps ahead to look for phase onset
        label_col: the same-timestep column to forecast (default 'disruptive')

    Returns:
        New dataframe with a 'forecast_label' column added, and all rows
        where label_col == 1 (already inside the phase) removed.
    """
    df = df.sort_values(["shot_id", "time"]).reset_index(drop=True)
    out_frames = []

    for shot_id, g in df.groupby("shot_id", sort=False):
        g = g.reset_index(drop=True)
        in_phase = g[label_col].to_numpy()
        n = len(g)

        forecast = np.zeros(n, dtype=int)
        for i in range(n):
            window = in_phase[i + 1: i + 1 + horizon_steps]
            forecast[i] = int(window.any())

        g = g.copy()
        g["forecast_label"] = forecast
        g = g[in_phase == 0]  # drop rows already inside the phase -- nothing to forecast there
        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    return result


def sanity_check_no_leakage(df: pd.DataFrame, feature_col: str = "f_G") -> None:
    """Quick gut-check: confirm a bare threshold on `feature_col` no longer
    trivially separates forecast_label the way it did for the same-timestep
    label. Run this every time you change the horizon or add a new feature."""
    from sklearn.metrics import average_precision_score

    if "forecast_label" not in df.columns:
        raise ValueError("Run make_forecast_label() first.")

    prevalence = df["forecast_label"].mean()
    ap = average_precision_score(df["forecast_label"], df[feature_col])

    print(f"Forecast label prevalence (chance baseline): {prevalence:.4f}")
    print(f"AUC-PR of raw '{feature_col}' alone (no model) on forecast_label: {ap:.4f}")

    if ap > 5 * prevalence:
        print(f"WARNING: '{feature_col}' alone still separates forecast_label suspiciously well "
              f"({ap:.4f} vs chance {prevalence:.4f}). Leakage may not be fully resolved -- "
              f"consider a longer horizon or check other features.")
    else:
        print(f"Looks OK: '{feature_col}' alone is close to chance-level on the forecasting task. "
              f"A trained model will need to learn a real precursor pattern, not restate a threshold.")


if __name__ == "__main__":
    # Quick standalone check against the cleaned dataframe + physics features.
    import sys
    sys.path.insert(0, ".")
    from Physics_featureM import add_physics_features

    df = pd.read_csv("../DATA/CLEANED/dl_dataframe_clean.csv")
    df = add_physics_features(df)

    for horizon in [1, 10, 20, 50]:
        print(f"\n--- horizon_steps={horizon} ({horizon * 10}ms lead time) ---")
        df_f = make_forecast_label(df, horizon_steps=horizon)
        sanity_check_no_leakage(df_f, feature_col="f_G")