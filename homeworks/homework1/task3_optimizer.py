from __future__ import annotations

from typing import Iterable, List, Tuple

import torch


class Adam:
    """Minimal Adam optimizer for lists of tensors (parameters).

    Works with 1D/shape-any tensors. Updates are done in-place.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params: List[torch.Tensor] = [p for p in params]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m: List[torch.Tensor] = [torch.zeros_like(p) for p in self.params]
        self.v: List[torch.Tensor] = [torch.zeros_like(p) for p in self.params]

    def step(self, grads: Iterable[torch.Tensor]) -> None:
        grads_list = [g for g in grads]
        if len(grads_list) != len(self.params):
            raise ValueError("grads must match params length")

        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads_list)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            p -= update


def train_neuron_with_adam(
    features: List[List[float]],
    labels: List[int],
    initial_weights: List[float],
    initial_bias: float,
    learning_rate: float,
    epochs: int,
    seed: int | None = None,
) -> Tuple[List[float], float, List[float]]:
    """Train the neuron from Task 1 but use custom Adam optimizer.

    Returns (updated_weights, updated_bias, nll_values_per_epoch_rounded_4dp)
    """
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    num_samples, num_features = X.shape
    w = torch.tensor(initial_weights, dtype=torch.float32).view(num_features, 1)
    b = torch.tensor([initial_bias], dtype=torch.float32)

    optimizer = Adam([w, b], lr=learning_rate)

    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def bce(y_true: torch.Tensor, y_prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        y_prob = torch.clamp(y_prob, eps, 1.0 - eps)
        return -(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))

    history: List[float] = []
    indices = torch.arange(num_samples)
    for _ in range(epochs):
        # Full-batch Adam to show the difference vs vanilla GD
        z = X.matmul(w) + b
        y_hat = sigmoid(z)
        loss_vec = bce(y, y_hat)
        loss = loss_vec.mean()

        dz = (y_hat - y)
        grad_w = X.t().matmul(dz) / X.shape[0]
        grad_b = dz.mean(dim=0)

        optimizer.step([grad_w, grad_b])

        history.append(round(float(loss.item()), 4))

    return w.view(-1).tolist(), float(b.item()), history


if __name__ == "__main__":
    toy_X = [
        [1.0, 2.0], [2.0, 1.0], [-1.0, -2.0], [-2.0, -1.5], [1.5, 1.0], [-1.2, -2.2]
    ]
    toy_y = [1, 1, 0, 0, 1, 0]
    w0 = [0.1, -0.2]
    b0 = 0.0

    w_adam, b_adam, hist_adam = train_neuron_with_adam(
        toy_X, toy_y, w0, b0, learning_rate=0.1, epochs=50, seed=42
    )
    # For comparison, run vanilla GD (reuse logic here for a single full batch)
    X = torch.tensor(toy_X, dtype=torch.float32)
    y = torch.tensor(toy_y, dtype=torch.float32).view(-1, 1)
    w_gd = torch.tensor(w0, dtype=torch.float32).view(-1, 1)
    b_gd = torch.tensor([b0], dtype=torch.float32)
    lr_gd = 0.1
    hist_gd: list[float] = []

    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def bce(y_true: torch.Tensor, y_prob: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        y_prob = torch.clamp(y_prob, eps, 1.0 - eps)
        return -(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))

    for _ in range(50):
        z = X.matmul(w_gd) + b_gd
        y_hat = sigmoid(z)
        loss = bce(y, y_hat).mean()
        dz = y_hat - y
        grad_w = X.t().matmul(dz) / X.shape[0]
        grad_b = dz.mean(dim=0)
        w_gd -= lr_gd * grad_w
        b_gd -= lr_gd * grad_b
        hist_gd.append(round(float(loss.item()), 4))

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
    print(fmt_row("Adam", 0.1, 50, hist_adam, w_adam, b_adam))
    print(fmt_row("GD", 0.1, 50, hist_gd, w_gd.view(-1).tolist(), float(b_gd.item())))
    print(sep)


