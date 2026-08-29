

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Physics_featureM import add_physics_features
from forecast_label import make_forecast_label
# Hardcoded paths -- edit these if your file locations change.
INPUT_PATH = "../DATA/CLEANED/dl_dataframe_clean.csv"
OUT_DIR = ".."
SEED = 42
HORIZON_STEPS = 20  # 200ms lead time at 10ms/step -- see forecast_label.py for the sanity check across horizons
 
 
def shot_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per shot_id: whether it EVER entered the phase, plus median
    machine params. Must be computed from the ORIGINAL (pre-forecast-filter)
    dataframe -- once make_forecast_label() runs, every row has
    disruptive == 0 by construction (in-phase rows get dropped), so this
    can't be recomputed after that step.
    """
    summary = df.groupby("shot_id").agg(
        disruptive=("disruptive", "max"),
        toroidal_B_field=("toroidal_B_field", "median"),
    ).reset_index()
    summary["disruptive"] = summary["disruptive"].astype(int)
    return summary
 
 
def report_split(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    def counts(part):
        shots = part["shot_id"].nunique()
        pos_shots = part.groupby("shot_id")["forecast_label"].max().sum()
        pos_rows = part["forecast_label"].sum()
        return shots, int(pos_shots), int(pos_rows)
 
    t_shots, t_pos_shots, t_pos_rows = counts(train_df)
    e_shots, e_pos_shots, e_pos_rows = counts(test_df)
    print(f"[{name}] train: {t_shots} shots ({t_pos_shots} with >=1 forecast-positive row, "
          f"{t_pos_rows} positive rows total) | "
          f"test: {e_shots} shots ({e_pos_shots} with >=1 forecast-positive row, {e_pos_rows} positive rows total)")
 
    if t_pos_shots < 5 or e_pos_shots < 5:
        print(f"  WARNING: fewer than 5 shots with a positive forecast window on one side -- "
              f"metrics from this split will be noisy/unreliable.")
 
 
def make_experiment1_stratified(df_content: pd.DataFrame, summary: pd.DataFrame,
                                 seed: int = 42, test_fraction: float = 0.2):
    train_ids, test_ids = train_test_split(
        summary["shot_id"], test_size=test_fraction, random_state=seed,
        stratify=summary["disruptive"],
    )
    train_df = df_content[df_content["shot_id"].isin(train_ids)].copy()
    test_df = df_content[df_content["shot_id"].isin(test_ids)].copy()
    return train_df, test_df
 
 
def make_domain_split(df_content: pd.DataFrame, summary: pd.DataFrame,
                       low_q: float = 0.33, high_q: float = 0.66):
    """Train on low toroidal_B_field shots, test on high toroidal_B_field shots.
    Middle third is dropped on purpose to create a clear gap between train and test.
    """
    low_val = np.nanquantile(summary["toroidal_B_field"], low_q)
    high_val = np.nanquantile(summary["toroidal_B_field"], high_q)
 
    train_ids = summary.loc[summary["toroidal_B_field"] <= low_val, "shot_id"]
    test_ids = summary.loc[summary["toroidal_B_field"] >= high_val, "shot_id"]
 
    train_df = df_content[df_content["shot_id"].isin(train_ids)].copy()
    test_df = df_content[df_content["shot_id"].isin(test_ids)].copy()
    return train_df, test_df
 
 
def main():
    raw_df = pd.read_csv(INPUT_PATH)
    raw_df = add_physics_features(raw_df)  # adds n_G, f_G -- computed BEFORE forecast filtering
 
    # shot-level summary from the ORIGINAL data (has real disruptive values,
    # needed for stratification and domain-split quantiles)
    summary = shot_level_summary(raw_df)
 
    # NOW convert to the forecast task: drop in-phase rows, add forecast_label
    df_forecast = make_forecast_label(raw_df, horizon_steps=HORIZON_STEPS, label_col="disruptive")
    print(f"Forecast label built at horizon={HORIZON_STEPS} steps ({HORIZON_STEPS*10}ms). "
          f"Rows remaining after dropping in-phase timesteps: {len(df_forecast)} / {len(raw_df)}")
 
    RAW_COLS = ["shot_id", "time", "forecast_label",
                "density", "elongation", "minor_radius",
                "plasma_current", "toroidal_B_field", "triangularity"]
 
    # --- Experiment 1: baseline ---
    train1, test1 = make_experiment1_stratified(df_forecast, summary, seed=SEED)
    report_split("Experiment 1 (stratified baseline)", train1, test1)
    save(OUT_DIR, "experiment_1_baseline", train1[RAW_COLS], test1[RAW_COLS])
 
    # --- Experiment 2: raw features, domain drift ---
    train2, test2 = make_domain_split(df_forecast, summary)
    report_split("Experiment 2 (raw features, domain drift)", train2, test2)
    save(OUT_DIR, "experiment_2_raw_features", train2[RAW_COLS], test2[RAW_COLS])
 
    # --- Experiment 3: physics features, same domain drift ---
    train3, test3 = make_domain_split(df_forecast, summary)
    report_split("Experiment 3 (physics features, domain drift)", train3, test3)

    keep_cols = ["shot_id", "time", "forecast_label",
                 "elongation", "toroidal_B_field", "triangularity", "f_G"]
    save(OUT_DIR, "experiment_3_physics_features", train3[keep_cols], test3[keep_cols])
 
 
def save(out_dir: str, experiment_folder: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    folder = os.path.join(out_dir, experiment_folder)
    os.makedirs(folder, exist_ok=True)
    train_path = os.path.join(folder, "train.csv")
    test_path = os.path.join(folder, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  saved -> {train_path}, {test_path}\n")
 
 
if __name__ == "__main__":
    main()