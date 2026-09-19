"""Training loop with node-level train/val/test masks and metrics."""

from __future__ import annotations

import numpy as np
import torch

from .models import GNNRegressor, BaselineMLP, DegreeBaseline


def split_masks(
    n: int, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random node-level train/val/test boolean masks."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    train[perm[:n_train]] = True
    val[perm[n_train : n_train + n_val]] = True
    test[perm[n_train + n_val :]] = True
    return train, val, test


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach().float()
    y_pred = y_pred.detach().float()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))


def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true.detach().float() - y_true.detach().float().mean()
    yp = y_pred.detach().float() - y_pred.detach().float().mean()
    denom = (yt.norm() * yp.norm()).clamp_min(1e-12)
    return float((yt * yp).sum() / denom)


def evaluate(
    pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Metrics on masked nodes: R^2 and Pearson correlation."""
    return {
        "r2": r2_score(y[mask], pred[mask]),
        "pearson": pearson_corr(y[mask], pred[mask]),
    }


def train_model(
    model: torch.nn.Module,
    graph,
    y: np.ndarray,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Train a node-regression model with MSE loss on the train mask,
    early stopping on validation loss. Returns history of val losses.

    The model is selected by its class: GNNRegressor consumes the full graph;
    BaselineMLP consumes node features; DegreeBaseline consumes degree features.
    """
    torch.manual_seed(seed)
    x, edge_index, edge_weight = graph.to_torch()
    y_t = torch.as_tensor(y, dtype=torch.float32)

    if isinstance(model, GNNRegressor):
        forward = lambda: model(x, edge_index, edge_weight)
    elif isinstance(model, DegreeBaseline):
        feats = torch.as_tensor(graph.degree_features(), dtype=torch.float32)
        forward = lambda: model(feats)
    elif isinstance(model, BaselineMLP):
        forward = lambda: model(x)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience, wait = 30, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = forward()
        loss = loss_fn(pred[train_mask], y_t[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = forward()
            val_loss = float(loss_fn(pred[val_mask], y_t[val_mask]))
        history["train_loss"].append(float(loss.detach()))
        history["val_loss"].append(val_loss)
        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch}: train={float(loss):.4f} val={val_loss:.4f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        model._cached_pred = forward()  # noqa: SLF001 - convenience for callers
    return history
