"""Synthetic connectome generator with planted tuning-from-wiring signal.

The planted generative model makes tuning a *function of wiring* so that
models that exploit connectivity CAN succeed, while a degree-preserving
rewired null destroys the fine connectivity and should fail. This is the
self-contained, data-free testbed for the benchmark (MICrONS not required).
"""

from __future__ import annotations

import numpy as np

from .graphs import ConnectomeGraph


def make_synthetic_connectome(
    n_neurons: int = 400,
    n_cell_types: int = 3,
    avg_degree: float = 12.0,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[ConnectomeGraph, np.ndarray]:
    """Build a synthetic connectome where tuning is a planted function of wiring.

    Each neuron has a latent "tuning angle" theta_i. Edges follow a
    like-to-like wiring rule (matching the Ding et al. 2025 finding in
    MICrONS): neurons with similar latent angles connect preferentially.
    The observed tuning label y is a noisy function of the *connectivity
    profile* (per-cell-type incoming synaptic weight), so it is recoverable
    from the graph but destroyed by topology-null rewiring.

    Returns:
    graph: ConnectomeGraph with one-hot cell-type node features.
    y: (N,) target labels (continuous tuning property in [-1, 1]).
    """
    rng = np.random.default_rng(seed)

    cell_types = rng.integers(0, n_cell_types, size=n_neurons)
    theta = rng.uniform(-np.pi, np.pi, size=n_neurons)

    # Like-to-like attachment: p_ij ~ (1 + cos(theta_i - theta_j)) * type compat
    compat = 0.5 + 0.5 * (cell_types[:, None] == cell_types[None, :]).astype(float)
    sim = 1.0 + np.cos(theta[:, None] - theta[None, :])
    p = sim * compat
    np.fill_diagonal(p, 0.0)
    p *= avg_degree / p.sum(axis=1, keepdims=True)

    draw = rng.random((n_neurons, n_neurons)) < p
    src, dst = np.nonzero(draw)
    # Synaptic weights (counts), inhibitory synapses get larger weights
    w = rng.poisson(3, size=len(src)).astype(np.float64) + 1.0
    w[cell_types[src] > 0] *= 2.0

    graph = ConnectomeGraph(
        node_features=np.eye(n_cell_types, dtype=np.float32)[cell_types],
        edge_index=np.stack([src, dst]).astype(np.int64),
        edge_weight=w,
        cell_types=cell_types,
        node_ids=np.arange(n_neurons),
    )

    # Planted label: a fixed function of the *fine* incoming connectivity
    # profile -- the total synaptic weight received from each partner cell
    # type (normalized). A single message-passing step can express this, so a
    # GNN can learn it, while a degree-preserving rewiring (which randomizes
    # partner identity and hence per-type incoming weight) destroys it.
    type_gain = np.linspace(1.0, -0.6, n_cell_types)  # e.g. [1, 0.2, -0.6]
    in_signal = np.zeros(n_neurons)
    in_total = np.zeros(n_neurons)
    np.add.at(in_signal, dst, w * type_gain[cell_types[src]])
    np.add.at(in_total, dst, w)
    eps = 1e-9
    y = in_signal / (in_total + eps)
    y = y + noise * rng.standard_normal(n_neurons)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return graph, y


def degree_preserving_rewire(
    graph: ConnectomeGraph, n_swaps: int | None = None, seed: int = 0
) -> ConnectomeGraph:
    """Degree-preserving topology null: random edge swaps that keep the exact in/out degree sequence but destroy fine connectivity.

    Uses the double-edge swap: for edges (a->b), (c->d), rewire to
    (a->d), (c->b), skipping self-loops and duplicate edges.
    """
    rng = np.random.default_rng(seed)
    src = graph.edge_index[0].copy()
    dst = graph.edge_index[1].copy()
    n_edges = len(src)
    if n_swaps is None:
        n_swaps = 10 * n_edges
    existing = set(zip(src.tolist(), dst.tolist()))
    for _ in range(n_swaps):
        i, j = rng.integers(0, n_edges, size=2)
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        if a == d or c == b:
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        existing.add((a, d))
        existing.add((c, b))
        dst[i], dst[j] = d, b
    return ConnectomeGraph(
        node_features=graph.node_features,
        edge_index=np.stack([src, dst]).astype(np.int64),
        edge_weight=graph.edge_weight.copy(),
        cell_types=graph.cell_types,
        node_ids=graph.node_ids,
    )
