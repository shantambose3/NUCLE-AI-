# Label Leakage and Domain-Shift Robustness in Physics-Informed Deep Learning for Tokamak Disruption Forecasting

Code accompanying the report *"Label Leakage and Domain-Shift Robustness in Physics-Informed Deep
Learning for Tokamak Disruption Forecasting."* This repository:

1. Diagnoses a same-timestep label-leakage problem in naive density-limit disruption prediction
   (a bare threshold on the Greenwald fraction, $f_G$, nearly solves the task with zero training).
2. Fixes it with a forecasting relabeling scheme (predict disruption onset 100–1000 ms ahead,
   with in-phase timesteps removed).
3. Trains a gated-fusion model (**PlasmaMCAT**) that combines raw sensor channels with the
   physics-derived $f_G$ signal.
4. Evaluates under an explicit domain-shift protocol (train on low toroidal-field shots, test on
   high toroidal-field shots) with 5-fold cross-validation, paired significance testing, and a
   70-run data-scalability sweep.

## Data

The dataset is the [MIT-PSFC Open Density Limit Database](https://github.com/MIT-PSFC/open_density_limit_database),
a multi-shot, per-timestep tokamak dataset (10 ms resolution) covering the density-limit disruption
mechanism. A copy is included under `data/` so the pipeline runs out of the box; see that repository
for provenance, machine coverage, and update history.

- `data/ORIGINAL/DL_DataFrame(MULTISHOT).csv` — raw multi-shot dataframe.
- `data/CLEANED/dl_dataframe_clean.csv` — output of `src/CLEAN_DATA.py` (standardized column names,
  required-field validation).

## Pipeline

Scripts are meant to be run from inside `src/` (they use relative paths, e.g. `../data/...`,
`../experiment_1_baseline/...`). Run in this order:

```bash
pip install -r requirements.txt
cd src

# 1. Clean the raw dataframe -> data/CLEANED/dl_dataframe_clean.csv
python CLEAN_DATA.py

# 2. Sanity-check the leakage problem and confirm the forecast relabeling fixes it
#    -> ../leakage_diagnosis_log.txt, ../leakage_diagnostic_results.csv
#    -> ../leakage_diagnosis.png, ../fig1_leakage_diagnostic.png
python leakage_diagnostic.py
python leakage_diagnostic_figure.py

# 3. Build the three experiment splits (stratified baseline + domain-shift raw/physics arms)
#    -> ../experiment_1_baseline/, ../experiment_2_raw_features/, ../experiment_3_physics_features/
python Split.py

# 4. Train PlasmaMCAT on all three experiments and compare
#    -> loss_curve.png, pr_curve.png, results_summary.csv, best_model.pt per experiment folder
python run_all_experiments.py

# 5. Robust 5-fold cross-validated comparison + paired significance testing (all 4 horizons)
#    -> ../robust_eval_summary.csv, ../robust_eval_raw_results.csv, ../paired_significance_test.csv
#    -> ../robust_eval_auc_pr.png, ../robust_eval_gap.png
python robust_evaluation.py

# 6. Data-scalability sweep (40-770 training shots, 5 seeds, 70 runs total)
#    -> ../scalability_summary.csv, ../scalability_raw_results.csv
#    -> ../scalability_auc_pr.png, ../scalability_f1.png
python Data_scalability.py
```

### Module map

| File | Role |
|---|---|
| `CLEAN_DATA.py` | Standardizes raw column names, validates required fields exist. |
| `Physics_featureM.py` | Computes the Greenwald limit $n_G = I_p / (\pi a^2)$ and fraction $f_G = n_e / n_G$. |
| `forecast_label.py` | Replaces the same-timestep disruption label with the forecasting label at each horizon; removes already-disrupting rows. |
| `leakage_diagnostic.py` / `leakage_diagnostic_figure.py` | Runs and plots the untrained-threshold leakage check, same-timestep vs. forecast label. |
| `Split.py` | Builds the stratified baseline split and the low-field/high-field domain-shift split for the raw- and physics-feature arms. |
| `CORE_MODEL.py` | `PlasmaMCAT`: two-stream model with a `GatedConcatFusion` layer fusing the raw sensor stream and the $f_G$ domain stream. |
| `M_Train.py` | Trains one experiment folder (focal loss, early stopping on val AUC-PR). |
| `run_all_experiments.py` | Runs `M_Train` on all three experiment folders and prints a comparison table. |
| `robust_evaluation.py` | 5-fold stratified CV across all four horizons, both arms, with paired $t$-test / Wilcoxon significance testing against each other and against a naive $f_G$-threshold baseline. |
| `Data_scalability.py` | Sweeps training-set size (40–770 shots × 5 seeds) for both arms at the 200 ms horizon. |

## Results summary

- **Leakage**: a bare threshold on $f_G$ alone hits **AUC-ROC 0.976** on the same-timestep label
  (prevalence 1.36%) — nearly solving the task with zero training. Reframed as a forecasting task,
  this collapses to 0.74–0.86 depending on horizon.
- **Domain-shift, 5-fold CV**: physics-informed features beat raw features at every horizon tested,
  reaching statistical significance at **200 ms** (paired $t$-test $p=0.0004$), **500 ms**
  ($p=0.0056$), and **1000 ms** ($p=0.029$), but not at the shortest horizon, **100 ms**
  ($p=0.57$).
- **Absolute performance**: both trained arms still trail a naive, untrained $f_G$-threshold in
  absolute AUC-PR terms at every horizon under this domain shift — headroom remains.
- **Scalability**: the physics-informed arm's advantage over raw features widens (not saturates)
  as training data grows from 40 to 770 shots, suggesting the effect is not a small-sample artifact.

Full numbers are in `results/`, plots in `figures/`. The write-up with discussion and limitations
is the accompanying report.

## Repository layout

```
.
├── data/                     # Open Density Limit Database (raw + cleaned)
├── src/                      # pipeline scripts (see Module map above)
├── results/                  # summary CSVs and logs from the runs reported in the paper
├── figures/                  # plots from the runs reported in the paper
├── requirements.txt
└── README.md
```

`experiment_*/train.csv`, `experiment_*/test.csv`, and `experiment_*/best_model.pt` are not
committed (see `.gitignore`) since they're large and fully regenerable by running the pipeline
above — `Split.py` recreates the splits deterministically (`SEED = 42`) and `run_all_experiments.py`
retrains the checkpoints.

## Requirements

```
numpy
pandas
scipy
scikit-learn
torch
matplotlib
```

Install with `pip install -r requirements.txt`. CUDA is used automatically if available, otherwise
falls back to CPU.

## Citation

If you use this code or the accompanying analysis, please cite the report and the underlying
dataset:

```
MIT Plasma Science and Fusion Center. Open Density Limit Database. GitHub, 2025-2026.
https://github.com/MIT-PSFC/open_density_limit_database
```

## License

See `LICENSE`.
