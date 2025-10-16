import math
from typing import List, Tuple, Optional

import torch


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def _binary_cross_entropy(y_true: torch.Tensor, y_prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    y_prob = torch.clamp(y_prob, eps, 1.0 - eps)
    return -(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))


def train_neuron(
    features: List[List[float]],
    labels: List[int],
    initial_weights: List[float],
    initial_bias: float,
    learning_rate: float,
    epochs: int,
    method: str = "gd",
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[float], float, List[float]]:
    """
    Train a single sigmoid neuron using manual gradients and NLL (BCE) loss.

    - method: 'gd' | 'sgd' | 'minibatch'
    - For 'gd', batch_size is ignored (full batch).
    - For 'sgd', batch_size is forced to 1.
    - For 'minibatch', provide batch_size > 1.

    Returns (updated_weights, updated_bias, nll_values_per_epoch_rounded_4dp)
    """
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    num_samples, num_features = X.shape

    w = torch.tensor(initial_weights, dtype=torch.float32).view(num_features, 1)
    b = torch.tensor([initial_bias], dtype=torch.float32)

    method = method.lower()
    if method not in {"gd", "sgd", "minibatch"}:
        raise ValueError("method must be one of: 'gd', 'sgd', 'minibatch'")

    if method == "sgd":
        batch_size = 1
    elif method == "gd":
        batch_size = num_samples
    else:
        if batch_size is None or batch_size <= 0:
            raise ValueError("For 'minibatch', provide batch_size > 0")

    nll_values: List[float] = []

    indices = torch.arange(num_samples)
    for _ in range(epochs):
        if shuffle and batch_size < num_samples:
            perm = torch.randperm(num_samples)
            indices = indices[perm]

        epoch_loss_accum = 0.0
        total_seen = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            Xb = X[batch_idx]  # (B, F)
            yb = y[batch_idx]  # (B, 1)

            # Forward
            z = Xb.matmul(w) + b  # (B, 1)
            y_hat = _sigmoid(z)   # (B, 1)

            # Loss (mean over batch)
            loss_vec = _binary_cross_entropy(yb, y_hat)  # (B, 1)
            batch_loss = loss_vec.mean()

            # Gradients (manual, no autograd): for BCE with sigmoid, dL/dz = y_hat - y
            dz = (y_hat - yb)  # (B, 1)
            grad_w = Xb.t().matmul(dz) / Xb.shape[0]  # (F, 1)
            grad_b = dz.mean(dim=0)  # (1,)

            # Update
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

            # accumulate for reporting epoch NLL (weighted by batch size to compute mean over all samples)
            epoch_loss_accum += float(batch_loss.item()) * Xb.shape[0]
            total_seen += Xb.shape[0]

        # epoch mean NLL
        epoch_mean_loss = epoch_loss_accum / max(1, total_seen)
        nll_values.append(round(epoch_mean_loss, 4))

    return w.view(-1).tolist(), float(b.item()), nll_values


if __name__ == "__main__":
    # Minimal demo on a toy linearly separable dataset
    toy_X = [
        [1.0, 2.0], [2.0, 1.0], [-1.0, -2.0], [-2.0, -1.5], [1.5, 1.0], [-1.2, -2.2]
    ]
    toy_y = [1, 1, 0, 0, 1, 0]
    w0 = [0.1, -0.2]
    b0 = 0.0

    w_gd, b_gd, hist_gd = train_neuron(
        toy_X, toy_y, w0, b0, learning_rate=0.2, epochs=50, method="gd", seed=42
    )

    w_sgd, b_sgd, hist_sgd = train_neuron(
        toy_X, toy_y, w0, b0, learning_rate=0.1, epochs=50, method="sgd", seed=42
    )

    w_mb, b_mb, hist_mb = train_neuron(
        toy_X, toy_y, w0, b0, learning_rate=0.15, epochs=50, method="minibatch", batch_size=3, seed=42
    )
    # CLI table summarizing variants
    header = (
        f"{'Method':<12} | {'LR':>6} | {'Epochs':>6} | {'Final NLL':>10} | {'Weights':<20} | {'Bias':>8}"
    )
    sep = "-" * len(header)
    def fmt_row(name: str, lr: float, epochs: int, hist: list[float], w: list[float], b: float) -> str:
        final_loss = hist[-1] if len(hist) > 0 else float('nan')
        w_str = "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
        return f"{name:<12} | {lr:>6.3f} | {epochs:>6d} | {final_loss:>10.4f} | {w_str:<20} | {b:>8.4f}"

    print(sep)
    print(header)
    print(sep)
    print(fmt_row("GD", 0.2, 50, hist_gd, w_gd, b_gd))
    print(fmt_row("SGD", 0.1, 50, hist_sgd, w_sgd, b_sgd))
    print(fmt_row("Mini-batch", 0.15, 50, hist_mb, w_mb, b_mb))
    print(sep)


