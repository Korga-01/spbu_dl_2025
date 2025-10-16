import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
#python train_year_prediction.py --epochs 120 --batch_size 512 --lr 0.002 --weight_decay 1e-5 --dropout 0.1 --submission submission.csv

class YearDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray | None = None):
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


class MLP(nn.Module):
    def __init__(self, input_dim: int, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def load_data(train_x_path: str, train_y_path: str, test_x_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_x_df = pd.read_csv(train_x_path)
    train_y_df = pd.read_csv(train_y_path)
    test_x_df = pd.read_csv(test_x_path)

    # train_x: first column appears to be id, followed by numeric features 0..89
    # train_y: two columns: id, year
    # test_x: features 0..89 and trailing id column named 'id'

    # Align and sort by id to be safe
    train_x_df = train_x_df.rename(columns={train_x_df.columns[0]: "id"})
    if "id" not in train_y_df.columns:
        # First column in train_y is the id but unnamed (e.g., header is empty before comma)
        train_y_df = train_y_df.rename(columns={train_y_df.columns[0]: "id"})
    train_df = train_x_df.merge(train_y_df, on="id", how="inner")
    train_df = train_df.sort_values("id").reset_index(drop=True)

    # Separate features and target
    feature_cols = [c for c in train_df.columns if c not in ("id", "year")]
    X_train = train_df[feature_cols]
    y_train = train_df["year"]

    # Ensure test columns align to feature order
    if "id" in test_x_df.columns and test_x_df.columns[-1] == "id":
        test_ids = test_x_df["id"].to_numpy()
        X_test = test_x_df.drop(columns=["id"])  # features are the rest
    else:
        # Fallback: assume first column is id
        test_x_df = test_x_df.rename(columns={test_x_df.columns[0]: "id"})
        test_ids = test_x_df["id"].to_numpy()
        X_test = test_x_df.drop(columns=["id"])  # features

    # Reindex test columns to match training feature order
    X_test = X_test[feature_cols]

    return X_train, y_train, pd.DataFrame({"id": test_ids, **{c: X_test[c] for c in X_test.columns}})


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> None:
    # Train with Huber (SmoothL1) for robustness to outliers
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # OneCycleLR for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.15,
        anneal_strategy="cos",
    )

    best_val = float("inf")
    patience = 5
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            xb, yb = batch
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss_accum = 0.0
            mse = nn.MSELoss(reduction="sum")
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    # monitor MSE on validation
                    val_loss_accum += mse(preds, yb).item()
            val_loss = val_loss_accum / len(val_loader.dataset)
            if val_loss < best_val:
                best_val = val_loss
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        else:
            val_loss = float("nan")

        # Optional: print progress
        print(f"Epoch {epoch:03d} | train MSE: {train_loss:.3f} | val MSE: {val_loss:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_x", default="train_x.csv")
    parser.add_argument("--train_y", default="train_y.csv")
    parser.add_argument("--test_x", default="test_x.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--clip_min", type=int, default=1900)
    parser.add_argument("--clip_max", type=int, default=2020)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--kfolds", type=int, default=5)
    args = parser.parse_args()

    X_train_df, y_train_s, test_df = load_data(args.train_x, args.train_y, args.test_x)

    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    y_train_np = y_train_s.to_numpy(dtype=np.float32)

    standardizer = Standardizer.fit(X_train_np)
    X_train_np = standardizer.transform(X_train_np)

    test_ids = test_df["id"].to_numpy()
    X_test_np = test_df.drop(columns=["id"]).to_numpy(dtype=np.float32)
    X_test_np = standardizer.transform(X_test_np)

    # Optional tuning with KFold
    def kfold_eval(lr: float, weight_decay: float, dropout_p: float, epochs: int) -> float:
        k = max(2, int(args.kfolds))
        n = X_train_np.shape[0]
        indices = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        fold_sizes = [(n + i) // k for i in range(k)]
        start = 0
        mses = []
        device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for fold_size in fold_sizes:
            val_idx = indices[start:start + fold_size]
            tr_idx = np.concatenate([indices[:start], indices[start + fold_size:]])
            start += fold_size

            X_tr, y_tr = X_train_np[tr_idx], y_train_np[tr_idx]
            X_vl, y_vl = X_train_np[val_idx], y_train_np[val_idx]

            st = Standardizer.fit(X_tr)
            X_tr_s = st.transform(X_tr)
            X_vl_s = st.transform(X_vl)

            train_loader = DataLoader(YearDataset(X_tr_s, y_tr), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(YearDataset(X_vl_s, y_vl), batch_size=args.batch_size, shuffle=False)

            model_local = MLP(input_dim=X_train_np.shape[1], dropout_p=dropout_p).to(device_local)
            train(model_local, train_loader, val_loader, device_local, epochs=epochs, lr=lr, weight_decay=weight_decay)

            model_local.eval()
            mse = nn.MSELoss(reduction="sum")
            s = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device_local)
                    yb = yb.to(device_local)
                    pred = model_local(xb)
                    s += mse(pred, yb).item()
            mses.append(s / len(val_loader.dataset))
        return float(np.mean(mses))

    if args.tune:
        grid_lrs = [5e-4, 1e-3, 2e-3]
        grid_wds = [1e-5, 3e-5, 1e-4]
        grid_drops = [0.1, 0.2, 0.3]
        best_params = None
        best_score = float("inf")
        for lr_ in grid_lrs:
            for wd_ in grid_wds:
                for dp_ in grid_drops:
                    score = kfold_eval(lr_, wd_, dp_, max(20, args.epochs // 2))
                    print(f"TUNE lr={lr_} wd={wd_} drop={dp_} -> MSE={score:.4f}")
                    if score < best_score:
                        best_score = score
                        best_params = (lr_, wd_, dp_)
        if best_params is not None:
            args.lr, args.weight_decay, args.dropout = best_params
            print(f"BEST: lr={args.lr} wd={args.weight_decay} drop={args.dropout} mse={best_score:.4f}")

    # Train/val split
    n_samples = X_train_np.shape[0]
    idx = np.arange(n_samples)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    val_size = int(n_samples * args.val_split)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr, y_tr = X_train_np[tr_idx], y_train_np[tr_idx]
    X_val, y_val = X_train_np[val_idx], y_train_np[val_idx]

    train_ds = YearDataset(X_tr, y_tr)
    val_ds = YearDataset(X_val, y_val)
    test_ds = YearDataset(X_test_np, None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X_train_np.shape[1], dropout_p=args.dropout).to(device)

    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

    # Fit on full data briefly to consolidate (optional fine-tuning)
    full_loader = DataLoader(YearDataset(X_train_np, y_train_np), batch_size=args.batch_size, shuffle=True)
    train(model, full_loader, None, device, epochs=max(5, args.epochs // 5), lr=args.lr * 0.5, weight_decay=args.weight_decay)

    # Predict on test
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in DataLoader(test_ds, batch_size=args.batch_size, shuffle=False):
            xb = xb.to(device)
            yb = model(xb).cpu().numpy()
            preds.append(yb)
    y_pred = np.concatenate(preds, axis=0)

    # Round to integer years and clip to a plausible range
    y_pred = np.rint(y_pred).astype(int)
    y_pred = np.clip(y_pred, args.clip_min, args.clip_max)

    # Write submission: columns exactly 'index,year' as requested
    sub_df = pd.DataFrame({"id": test_ids, "year": y_pred})
    sub_df.to_csv(args.submission, index=False)
    print(f"Saved submission to {args.submission} with {len(sub_df)} rows.")


if __name__ == "__main__":
    main()


