"""
train.py

Trains PlasmaMCAT (from model.py) on one experiment folder and saves:
  - loss_curve.png       (train vs test loss per epoch)
  - pr_curve.png         (precision-recall curve of the best checkpoint)
  - results_summary.csv  (small table: shot counts, best epoch, best metrics)
  - best_model.pt        (checkpoint with highest test AUC-PR)

Uses CUDA automatically if available (checks torch.cuda.is_available()) 
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

from CORE_MODEL import PlasmaMCAT

# ----------------------------------------------------------------------
# Config -- change EXPERIMENT_DIR to point at whichever experiment you're
# running. Everything else adapts automatically based on what columns are
# present (raw features vs. f_G).
# ----------------------------------------------------------------------
EXPERIMENT_DIR = "../experiment_1_baseline"   # or experiment_2_raw_features / experiment_3_physics_features
MAX_LEN = 142          # longest shot in the whole dataset -- pad everything to this
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
SEED = 42

NON_FEATURE_COLS = {"shot_id", "time", "forecast_label"}


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class ShotDataset(Dataset):
    """Groups a per-timestep CSV by shot_id and pads each shot to MAX_LEN.

    Automatically splits columns into:
      - domain columns: ['f_G'] if present (Experiment 3), else none
      - sensor columns: everything else that isn't shot_id/time/forecast_label

    If there's no f_G column (Experiment 1/2), domain_physics is filled with
    zeros -- the gate in GatedConcatFusion should learn to ignore it, so
    the model architecture stays identical across all 3 experiments and only
    the *input* changes. That's what makes it a fair comparison.

    label is now a PER-TIMESTEP sequence (forecast_label at each row), not a
    single shot-level scalar -- this is a forecasting task, not detection.
    """

    def __init__(self, csv_path: str = None, df: pd.DataFrame = None, max_len: int = MAX_LEN):
        if df is None:
            df = pd.read_csv(csv_path)

        self.has_physics = "f_G" in df.columns
        self.domain_cols = ["f_G"] if self.has_physics else []
        self.sensor_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c not in self.domain_cols]

        self.max_len = max_len
        self.shots = []  # list of dicts: sensor [T,C], domain [T,D], length, label_seq [T]

        for shot_id, g in df.groupby("shot_id"):
            g = g.sort_values("time")
            length = len(g)
            if length > max_len:
                g = g.iloc[:max_len]
                length = max_len

            sensor = g[self.sensor_cols].to_numpy(dtype=np.float32) if self.sensor_cols else np.zeros((length, 1), dtype=np.float32)
            domain = g[self.domain_cols].to_numpy(dtype=np.float32) if self.domain_cols else np.zeros((length, 1), dtype=np.float32)
            label_seq = g["forecast_label"].to_numpy(dtype=np.float32)

            pad_len = max_len - length
            if pad_len > 0:
                sensor = np.pad(sensor, ((0, pad_len), (0, 0)))
                domain = np.pad(domain, ((0, pad_len), (0, 0)))
                label_seq = np.pad(label_seq, (0, pad_len))  # padded label values are masked out later, never used

            self.shots.append({
                "sensor": sensor,
                "domain": domain,
                "length": length,
                "label_seq": label_seq,
                "shot_id": shot_id,
            })

        self.sensor_channels = self.shots[0]["sensor"].shape[1]
        self.domain_channels = self.shots[0]["domain"].shape[1]
        self.n_shots = len(self.shots)
        self.n_positive_rows = int(sum(s["label_seq"].sum() for s in self.shots))
        self.n_positive_shots = int(sum(1 for s in self.shots if s["label_seq"].sum() > 0))

    def __len__(self):
        return len(self.shots)

    def __getitem__(self, idx):
        s = self.shots[idx]
        return (
            torch.from_numpy(s["sensor"]),
            torch.from_numpy(s["domain"]),
            torch.tensor(s["length"], dtype=torch.long),
            torch.from_numpy(s["label_seq"]),
        )


# ----------------------------------------------------------------------
# Loss -- Focal loss for the ~3% positive rate.
# ----------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * focal_weight * bce
        return loss.mean()


# ----------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------
def flatten_valid_timesteps(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """values: [Batch, Time] -- returns a 1D tensor of only the valid
    (non-padded) entries, using `lengths` to know how many timesteps per
    shot are real."""
    batch_size, max_len = values.shape
    device = values.device
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = positions < lengths.unsqueeze(1)
    return values[valid_mask]


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_logits, all_labels = [], []
    total_loss = 0.0
    total_valid = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for sensor, domain, lengths, label_seq in loader:
            sensor, domain, lengths, label_seq = (
                sensor.to(device), domain.to(device), lengths.to(device), label_seq.to(device)
            )
            logits_seq = model(sensor, domain, lengths, per_timestep=True).squeeze(-1)  # [Batch, Time]

            flat_logits = flatten_valid_timesteps(logits_seq, lengths)
            flat_labels = flatten_valid_timesteps(label_seq, lengths)

            loss = criterion(flat_logits, flat_labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * flat_labels.numel()
            total_valid += flat_labels.numel()
            all_logits.append(flat_logits.detach().cpu())
            all_labels.append(flat_labels.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / total_valid

    return avg_loss, all_logits, all_labels


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Imbalance-appropriate metrics. Deliberately does NOT report accuracy --
    with ~3% positives, a model predicting all-zero gets ~97% accuracy while
    being useless. AUC-PR and F1 at a tuned threshold are what matter here.
    """
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    auc_pr = average_precision_score(labels, probs)

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_f1 = float(np.nanmax(f1_scores))

    return {"auc_pr": float(auc_pr), "best_f1": best_f1}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available -- using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available -- falling back to CPU.")
    return device


def train_one_experiment(experiment_dir: str, epochs: int = EPOCHS, seed: int = SEED,
                          make_plots: bool = True) -> dict:
    """Runs the full train/eval loop for one experiment folder.

    Returns a dict of final results (also written to results_summary.csv and,
    if make_plots=True, loss_curve.png / pr_curve.png inside experiment_dir).
    """
    torch.manual_seed(seed)
    device = get_device()

    train_path = os.path.join(experiment_dir, "train.csv")
    test_path = os.path.join(experiment_dir, "test.csv")

    train_ds = ShotDataset(csv_path=train_path)
    test_ds = ShotDataset(csv_path=test_path)

    print(f"[{experiment_dir}] sensor_cols={train_ds.sensor_cols}, "
          f"domain_cols={train_ds.domain_cols or '(none -- zero-filled)'}")
    print(f"[{experiment_dir}] train: {train_ds.n_shots} shots ({train_ds.n_positive_shots} with >=1 "
          f"forecast-positive row, {train_ds.n_positive_rows} positive rows) | "
          f"test: {test_ds.n_shots} shots ({test_ds.n_positive_shots} with >=1 "
          f"forecast-positive row, {test_ds.n_positive_rows} positive rows)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PlasmaMCAT(
        sensor_channels=train_ds.sensor_channels,
        domain_channels=train_ds.domain_channels,
    ).to(device)

    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "test_loss": [], "auc_pr": [], "best_f1": []}
    best_test_auc_pr = -1.0
    best_logits, best_labels = None, None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_logits, test_labels = run_epoch(model, test_loader, criterion, None, device)
        metrics = compute_metrics(test_logits, test_labels)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["auc_pr"].append(metrics["auc_pr"])
        history["best_f1"].append(metrics["best_f1"])

        print(f"[{experiment_dir}] Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | AUC-PR={metrics['auc_pr']:.4f} | F1={metrics['best_f1']:.4f}")

        if metrics["auc_pr"] > best_test_auc_pr:
            best_test_auc_pr = metrics["auc_pr"]
            best_epoch = epoch
            best_logits, best_labels = test_logits, test_labels
            torch.save(model.state_dict(), os.path.join(experiment_dir, "best_model.pt"))

    results = {
        "experiment": os.path.basename(experiment_dir.rstrip("/")),
        "n_train_shots": train_ds.n_shots,
        "n_train_positive_shots": train_ds.n_positive_shots,
        "n_train_positive_rows": train_ds.n_positive_rows,
        "n_test_shots": test_ds.n_shots,
        "n_test_positive_shots": test_ds.n_positive_shots,
        "n_test_positive_rows": test_ds.n_positive_rows,
        "sensor_cols": ";".join(train_ds.sensor_cols),
        "domain_cols": ";".join(train_ds.domain_cols) or "none",
        "best_epoch": best_epoch,
        "best_auc_pr": best_test_auc_pr,
        "best_f1": max(history["best_f1"]),
    }

    pd.DataFrame([results]).to_csv(os.path.join(experiment_dir, "results_summary.csv"), index=False)
    print(f"[{experiment_dir}] saved results_summary.csv")

    if make_plots:
        _plot_loss_curve(experiment_dir, history)
        _plot_pr_curve(experiment_dir, best_logits, best_labels, best_epoch)

    return results


def _plot_loss_curve(experiment_dir: str, history: dict) -> None:
    """Graph 1: train vs test loss per epoch -- shows if/when the model overfits."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train loss", marker="o", markersize=3)
    ax.plot(epochs, history["test_loss"], label="test loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal loss")
    ax.set_title(f"Loss curve: {os.path.basename(experiment_dir.rstrip('/'))}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def _plot_pr_curve(experiment_dir: str, logits: np.ndarray, labels: np.ndarray, best_epoch: int) -> None:
    """Graph 2: precision-recall curve of the best checkpoint -- the metric
    that actually matters given ~3% positive rate (unlike an ROC curve, which
    looks deceptively good on imbalanced data)."""
    probs = 1 / (1 + np.exp(-logits))
    precisions, recalls, _ = precision_recall_curve(labels, probs)
    auc_pr = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, color="tab:orange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve (epoch {best_epoch}, AUC-PR={auc_pr:.3f}): "
                 f"{os.path.basename(experiment_dir.rstrip('/'))}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)


def main():
    train_one_experiment(EXPERIMENT_DIR)


if __name__ == "__main__":
    main()