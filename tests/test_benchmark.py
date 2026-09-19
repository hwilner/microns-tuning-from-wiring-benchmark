"""Synthetic benchmark tests (data-free, fast, CPU).

Core scientific guarantees:
1. The planted tuning-from-wiring signal is recoverable: the GNN beats the
   mean null model on held-out nodes.
2. The GNN recovers the *wiring* structure, not spurious correlations:
   on a degree-preserving rewired null graph it performs no better than null.
3. Training loop smoke test runs end-to-end.
4. The degree baseline feature extractor is deterministic.
"""

import numpy as np
import torch

from wiring_tuning import (
    ConnectomeGraph,
    GNNRegressor,
    make_synthetic_connectome,
    degree_preserving_rewire,
    sample_subgraph,
    split_masks,
    train_model,
    evaluate,
)
from wiring_tuning.models import MeanBaseline

N = 200
EPOCHS = 120


def _fit_gnn(seed=0, n=N, epochs=EPOCHS, rewired=False):
    graph, y = make_synthetic_connectome(n_neurons=n, seed=seed)
    if rewired:
        graph = degree_preserving_rewire(graph, seed=seed)
    train, val, test = split_masks(graph.num_nodes, seed=seed)
    model = GNNRegressor(graph.node_features.shape[1], hidden_dim=32)
    train_model(model, graph, y, train, val, epochs=epochs, seed=seed)
    pred = model._cached_pred
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return graph, y_t, pred, (train, val, test)


def test_gnn_beats_mean_null():
    """GNN recovers planted wiring signal better than the mean null."""
    graph, y_t, pred, (train_mask, _, test) = _fit_gnn()
    mean_pred = MeanBaseline().fit(y_t[train_mask]).predict(graph.num_nodes)
    gnn = evaluate(pred, y_t, test)
    null = evaluate(mean_pred, y_t, test)
    assert gnn["r2"] > null["r2"] + 0.3, (gnn, null)
    assert gnn["pearson"] > 0.5, gnn


def test_gnn_recovers_structure_not_null():
    """On a degree-preserving rewired graph, GNN performance collapses to null."""
    _, y_t, pred_good, (_, _, test) = _fit_gnn(seed=1)
    _, _, pred_rewired, (_, _, test2) = _fit_gnn(seed=1, rewired=True)
    good = evaluate(pred_good, y_t, test)
    bad = evaluate(pred_rewired, y_t, test2)
    assert good["r2"] > 0.4, good
    assert bad["r2"] < 0.2, bad
    assert good["r2"] > bad["r2"] + 0.3, (good, bad)


def test_training_loop_smoke():
    graph, y = make_synthetic_connectome(n_neurons=100, seed=2)
    train, val, _ = split_masks(graph.num_nodes, seed=2)
    model = GNNRegressor(graph.node_features.shape[1], hidden_dim=16)
    hist = train_model(model, graph, y, train, val, epochs=20, seed=2)
    assert len(hist["train_loss"]) >= 1
    assert np.isfinite(hist["val_loss"][-1])
    assert model._cached_pred.shape == (graph.num_nodes,)


def test_degree_features_deterministic():
    graph, _ = make_synthetic_connectome(n_neurons=80, seed=3)
    f1 = graph.degree_features()
    f2 = graph.degree_features()
    assert f1.shape == (80, 6)
    np.testing.assert_array_equal(f1, f2)


def test_rewire_preserves_degree_sequence():
    graph, _ = make_synthetic_connectome(n_neurons=100, seed=4)
    rewired = degree_preserving_rewire(graph, seed=4)
    n = graph.num_nodes
    # double-edge swaps preserve the exact per-node in/out degree sequences
    in0 = np.bincount(graph.edge_index[1], minlength=n)
    out0 = np.bincount(graph.edge_index[0], minlength=n)
    in1 = np.bincount(rewired.edge_index[1], minlength=n)
    out1 = np.bincount(rewired.edge_index[0], minlength=n)
    np.testing.assert_array_equal(in0, in1)
    np.testing.assert_array_equal(out0, out1)


def test_subgraph_sampling():
    graph, _ = make_synthetic_connectome(n_neurons=60, seed=5)
    sub = sample_subgraph(graph, np.arange(20))
    assert sub.num_nodes == 20
    assert sub.node_features.shape[0] == 20
    assert sub.edge_index.max() < 20 if sub.num_edges else True
