"""Models: pure-PyTorch message-passing GNN and non-graph baselines.

Implemented with adjacency-based message passing (scatter aggregation), so
no torch_geometric dependency is required. Models regress a node-level
tuning property (e.g., orientation selectivity index) from graph structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


class MessagePassingLayer(nn.Module):
    """Weighted message passing: h_i <- MLP(h_i || sum_j w_ji * W h_j).

    Messages flow along edges (source -> target), weighted by edge weight,
    normalized by the square root of in-degree for stability.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.msg = nn.Linear(in_dim, out_dim)
        self.upd = nn.Linear(in_dim + out_dim, out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        w = edge_weight / edge_weight.mean().clamp_min(1e-6)
        m = self.msg(x[src]) * w.unsqueeze(-1)
        agg = scatter_add(m, dst, n)
        out = self.upd(torch.cat([x, agg], dim=-1))
        return torch.relu(out)


class GNNRegressor(nn.Module):
    """Graph neural network regressing a tuning property per neuron."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            [MessagePassingLayer(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return self.head(h).squeeze(-1)


class BaselineMLP(nn.Module):
    """Features-only MLP baseline (node features, no wiring)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DegreeBaseline(nn.Module):
    """Hand-crafted connectivity features (degree/E-I balance) MLP baseline.
    Input is the deterministic degree feature vector from
    ConnectomeGraph.degree_features().
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MeanBaseline:
    """Null model: always predicts the train-set mean. Non-torch."""

    def fit(self, y_train: torch.Tensor) -> "MeanBaseline":
        self.mean_ = float(y_train.mean())
        return self

    def predict(self, n: int) -> torch.Tensor:
        return torch.full((n,), self.mean_)
