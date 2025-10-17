import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# For preprocessing (allowed for pre/post-processing)
from sklearn.preprocessing import QuantileTransformer
#python spbu_dl_2025/homeworks/homework1/train_year_prediction_adv.py --train_x spbu_dl_2025/homeworks/homework1/train_x.csv --train_y spbu_dl_2025/homeworks/homework1/train_y.csv --test_x spbu_dl_2025/homeworks/homework1/test_x.csv --epochs 150 --epochs_huber 30 --batch_size 512 --lr 0.002 --weight_decay 1e-5 --dropout 0.1 --submission spbu_dl_2025/homeworks/homework1/submission.csv --clip_to_train_range

class YearDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        self.features = torch.from_numpy(features).float()
        self.targets = None if targets is None else torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.targets is None:
            return x
        y = self.targets[idx]
        return x, y


class LayerNormMLP(nn.Module):
    def __init__(self, input_dim: int, dropout_p: float = 0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x).squeeze(1)


@dataclass
class PreprocessArtifacts:
    quant: QuantileTransformer
    feature_mask: np.ndarray  # boolean mask of kept columns
    y_min: int
    y_max: int


def load_raw(train_x_path: str, train_y_path: str, test_x_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    tx = pd.read_csv(train_x_path)
    ty = pd.read_csv(train_y_path)
    te = pd.read_csv(test_x_path)

    # Normalize id columns
    tx = tx.rename(columns={tx.columns[0]: "id"})
    if "id" not in ty.columns:
        ty = ty.rename(columns={ty.columns[0]: "id"})

    df = tx.merge(ty, on="id", how="inner").sort_values("id").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("id", "year")]

    # Test: last column is id
    if "id" in te.columns and te.columns[-1] == "id":
        test_ids = te["id"].to_numpy()
        teX = te.drop(columns=["id"])  # keep feature order
    else:
        te = te.rename(columns={te.columns[0]: "id"})
        test_ids = te["id"].to_numpy()
        teX = te.drop(columns=["id"])  # features

    teX = teX[feature_cols]

    return df[feature_cols], df["year"], pd.DataFrame({"id": test_ids, **{c: teX[c] for c in teX.columns}})


def robust_preprocess(X: np.ndarray, X_test: np.ndarray, clip_p: float = 0.002) -> Tuple[np.ndarray, np.ndarray, PreprocessArtifacts]:
    # Remove constant columns
    std = X.std(axis=0)
    mask = std > 0
    X = X[:, mask]
    X_test = X_test[:, mask]

    # Clip extremes by quantiles
    lo = np.quantile(X, clip_p, axis=0)
    hi = np.quantile(X, 1 - clip_p, axis=0)
    X = np.clip(X, lo, hi)
    X_test = np.clip(X_test, lo, hi)

    # Quantile normalize to N(0,1)
    quant = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, X.shape[0]))
    Xq = quant.fit_transform(X)
    Xq_test = quant.transform(X_test)

    artifacts = PreprocessArtifacts(quant=quant, feature_mask=mask, y_min=1900, y_max=2020)
    return Xq.astype(np.float32), Xq_test.astype(np.float32), artifacts


def stratified_kfold_indices(y: np.ndarray, k: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    # Bin years for stratification
    y_bins = pd.qcut(y, q=min(10, len(np.unique(y))), duplicates='drop').codes
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    # Simple manual stratified split
    folds = [[] for _ in range(k)]
    for b in np.unique(y_bins):
        b_idx = idx[y_bins[idx] == b]
        for i, j in enumerate(np.array_split(b_idx, k)):
            folds[i].extend(j.tolist())
    folds = [np.array(sorted(f)) for f in folds]
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def train_one(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], device: torch.device,
              epochs_huber: int, epochs_mse: int, lr: float, weight_decay: float, use_swa: bool) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs_huber + epochs_mse,
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.15)

    criterion_huber = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()

    swa_model = AveragedModel(model) if use_swa else None
    swa_start_epoch = max(0, epochs_huber + epochs_mse - 20)
    swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.5) if use_swa else None

    best_val = float('inf')
    mse_metric = nn.MSELoss(reduction="sum")

    total_epochs = epochs_huber + epochs_mse
    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = (criterion_huber if epoch < epochs_huber else criterion_mse)(preds, yb)
            loss.backward()
            optimizer.step()
            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
            else:
                scheduler.step()

        if val_loader is not None:
            model.eval()
            s = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    s += mse_metric(pred, yb).item()
            val_mse = s / len(val_loader.dataset)
            best_val = min(best_val, val_mse)

    # Finalize SWA
    if use_swa and swa_model is not None:
        update_bn(train_loader, swa_model, device=device)
        return best_val
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_x", default="train_x.csv")
    parser.add_argument("--train_y", default="train_y.csv")
    parser.add_argument("--test_x", default="test_x.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--epochs_huber", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--clip_to_train_range", action="store_true")
    args = parser.parse_args()

    # Load
    X_df, y_s, test_df = load_raw(args.train_x, args.train_y, args.test_x)
    X_np = X_df.to_numpy(dtype=np.float32)
    y_np = y_s.to_numpy(dtype=np.float32)
    test_ids = test_df["id"].to_numpy()
    X_test_np = test_df.drop(columns=["id"]).to_numpy(dtype=np.float32)

    # Preprocess
    X_proc, X_test_proc, artifacts = robust_preprocess(X_np, X_test_np)
    if args.clip_to_train_range:
        artifacts.y_min = int(y_np.min())
        artifacts.y_max = int(y_np.max())

    # KFold CV for epoch split (Huber -> MSE) using fixed lr/wd/dropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = stratified_kfold_indices(y_np, k=args.kfolds, seed=42)
    val_scores = []
    for i, (tr_idx, vl_idx) in enumerate(folds):
        X_tr, y_tr = X_proc[tr_idx], y_np[tr_idx]
        X_vl, y_vl = X_proc[vl_idx], y_np[vl_idx]
        train_loader = DataLoader(YearDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(YearDataset(X_vl, y_vl), batch_size=args.batch_size, shuffle=False)
        model = LayerNormMLP(input_dim=X_proc.shape[1], dropout_p=args.dropout).to(device)
        val_mse = train_one(model, train_loader, val_loader, device,
                            epochs_huber=args.epochs_huber,
                            epochs_mse=max(1, args.epochs - args.epochs_huber),
                            lr=args.lr, weight_decay=args.weight_decay, use_swa=True)
        print(f"Fold {i+1}/{args.kfolds} MSE: {val_mse:.3f}")
        val_scores.append(val_mse)
    print(f"CV MSE: mean={np.mean(val_scores):.3f} std={np.std(val_scores):.3f}")

    # Train on full data
    full_loader = DataLoader(YearDataset(X_proc, y_np), batch_size=args.batch_size, shuffle=True)
    model = LayerNormMLP(input_dim=X_proc.shape[1], dropout_p=args.dropout).to(device)
    _ = train_one(model, full_loader, None, device,
                  epochs_huber=args.epochs_huber,
                  epochs_mse=max(1, args.epochs - args.epochs_huber),
                  lr=args.lr, weight_decay=args.weight_decay, use_swa=True)

    # Predict
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in DataLoader(YearDataset(X_test_proc, None), batch_size=args.batch_size, shuffle=False):
            xb = xb.to(device)
            yb = model(xb).cpu().numpy()
            preds.append(yb)
    y_pred = np.concatenate(preds, axis=0)
    # Round and clip
    y_pred = np.rint(y_pred).astype(int)
    y_pred = np.clip(y_pred, artifacts.y_min, artifacts.y_max)

    sub = pd.DataFrame({"id": test_ids, "year": y_pred})
    sub.to_csv(args.submission, index=False)
    print(f"Saved submission to {args.submission} with {len(sub)} rows.")


if __name__ == "__main__":
    main()


