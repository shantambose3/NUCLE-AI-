"""
clean_data.py

Takes the raw DL_DataFrame.csv (or similar multi-shot dataframe) and produces a
cleaned version with consistent column names, so every other script in this
project can rely on the same schema instead of guessing column names.
"""

import pandas as pd

# Hardcoded paths -- edit these if your file locations change.
INPUT_PATH = "../DATA/ORIGINAL/DL_DataFrame(MULTISHOT).csv"
OUTPUT_PATH = "../DATA/CLEANED/dl_dataframe_clean.csv"

# Map from raw column name -> clean column name.
# Add more entries here if a future dataset uses different raw names.
COLUMN_RENAME_MAP = {
    "discharge_ID": "shot_id",
    "density_limit_phase": "disruptive",
}

# Columns that MUST exist (after renaming) for downstream scripts to work.
REQUIRED_COLUMNS = [
    "shot_id",
    "time",
    "disruptive",
    "density",
    "plasma_current",
    "minor_radius",
]


def load_and_clean(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    print(f"Loaded '{input_path}' -> shape {df.shape}")
    print(f"Raw columns: {list(df.columns)}")

    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Check nothing required is missing after renaming.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"After renaming, still missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}\n"
            f"Fix COLUMN_RENAME_MAP at the top of this file to match your raw data."
        )

    # Basic sanity checks -- fail loudly instead of silently continuing with bad data.
    n_missing_values = df.isna().sum().sum()
    if n_missing_values > 0:
        print(f"WARNING: {n_missing_values} missing values found across the dataset.")
        print(df.isna().sum())

    if not set(df["disruptive"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Expected 'disruptive' column to be binary (0/1), found values: {df['disruptive'].unique()}"
        )

    n_shots = df["shot_id"].nunique()
    n_disruptive_shots = df.groupby("shot_id")["disruptive"].max().sum()
    print(f"Total shots: {n_shots}")
    print(f"Disruptive shots (label==1 at some point): {n_disruptive_shots} "
          f"({100 * n_disruptive_shots / n_shots:.1f}%)")

    return df


def main():
    df = load_and_clean(INPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned data to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()