"""
physics_features.py

Computes the Greenwald density limit and Greenwald fraction -- the standard
domain-knowledge indicator for density-limit disruptions -- from the raw
machine parameters already in the dataset.

    Greenwald limit:    n_G = I_p / (pi * a^2)
    Greenwald fraction: f_G = density / n_G

Where:
    I_p = plasma_current
    a   = minor_radius
    density = line-averaged density

f_G is dimensionless. f_G approaching/exceeding 1.0 is the physical signature
of an impending density-limit disruption -- that's the entire point of using
it as a feature instead of the three raw variables separately.

IMPORTANT: this does NOT silently fall back to a placeholder (e.g. a
z-score or a constant zero) if a required column is missing -- it raises
an error instead. A silent fallback here would quietly break the entire
physics-vs-raw comparison without any visible sign that something went
wrong, which is worse than crashing.

Usage (as a module):
    from physics_features import add_physics_features
    df_with_features = add_physics_features(df)

Usage (standalone, for a quick check):
    python physics_features.py --input ../data/clean/dl_dataframe_clean.csv
"""

import argparse

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["density", "plasma_current", "minor_radius"]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot compute Greenwald fraction -- missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}\n"
            f"Fix the column names (see clean_data.py) before calling add_physics_features()."
        )

    out = df.copy()

    # Greenwald limit (n_G). Add a tiny epsilon to avoid divide-by-zero if
    # minor_radius is ever exactly 0 (shouldn't happen physically, but data
    # is data).
    out["n_G"] = out["plasma_current"] / (np.pi * out["minor_radius"] ** 2 + 1e-12)

    # Greenwald fraction (f_G) -- this is the actual domain-knowledge feature.
    out["f_G"] = out["density"] / (out["n_G"] + 1e-12)

    _sanity_check(out)

    return out


def _sanity_check(df: pd.DataFrame) -> None:
    """Print basic stats so a wrong unit/formula shows up immediately rather
    than during model training three files later."""
    print("Greenwald fraction (f_G) summary:")
    print(df["f_G"].describe())

    frac_above_1 = (df["f_G"] >= 1.0).mean()
    print(f"Fraction of rows with f_G >= 1.0 (at/above Greenwald limit): {frac_above_1:.3%}")

    if "disruptive" in df.columns:
        mean_fg_disruptive = df.loc[df["disruptive"] == 1, "f_G"].mean()
        mean_fg_stable = df.loc[df["disruptive"] == 0, "f_G"].mean()
        print(f"Mean f_G during disruptive timesteps: {mean_fg_disruptive:.3f}")
        print(f"Mean f_G during stable timesteps:     {mean_fg_stable:.3f}")
        if mean_fg_disruptive <= mean_fg_stable:
            print("WARNING: f_G is not higher during disruptive timesteps than stable ones. "
                  "Check units/column mapping -- this indicator should rise near disruption.")


def main():
    parser = argparse.ArgumentParser(description="Compute and sanity-check the Greenwald fraction feature.")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV (from clean_data.py)")
    parser.add_argument("--output", default=None, help="Optional path to save the result with f_G added")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_out = add_physics_features(df)

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()