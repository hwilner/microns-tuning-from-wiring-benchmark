"""Benchmark runner: train models across seeds and produce a results table."""

from __future__ import annotations

import pandas as pd
import torch

from .graphs import ConnectomeGraph
from .models import GNNRegressor, BaselineMLP, DegreeBaseline, MeanBaseline
from .simulate import make_synthetic_connectome, degree_preserving_rewire
from .train import split_masks, train_model, evaluate

MODELS = ("gnn", "degree_mlp", "features_mlp", "mean_null")


def _fit_predict(
    model_name: str, graph: ConnectomeGraph, y, masks, epochs: int, seed: int
) -> torch.Tensor:
    train_mask, val_mask, _ = masks
    if model_name == "mean_null":
        m = MeanBaseline().fit(torch.as_tensor(y, dtype=torch.float32)[train_mask])
        return m.predict(graph.num_nodes)
    if model_name == "gnn":
        model = GNNRegressor(graph.node_features.shape[1])
    elif model_name == "degree_mlp":
        model = DegreeBaseline()
    elif model_name == "features_mlp":
        model = BaselineMLP(graph.node_features.shape[1])
    else:
        raise ValueError(f"unknown model: {model_name}")
    train_model(model, graph, y, train_mask, val_mask, epochs=epochs, seed=seed)
    return model._cached_pred  # noqa: SLF001


def run_benchmark(
    seeds: tuple[int, ...] = (0, 1, 2),
    n_neurons: int = 300,
    epochs: int = 150,
    include_rewired_null: bool = True,
) -> pd.DataFrame:
    """Run all models across seeds on the planted synthetic connectome
    (and optionally its degree-preserving rewired null).

    Returns a long-format results table:
    [graph, model, seed, r2, pearson] evaluated on the test mask.
    """
    rows = []
    for seed in seeds:
        graph, y = make_synthetic_connectome(n_neurons=n_neurons, seed=seed)
        masks = split_masks(graph.num_nodes, seed=seed)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        variants = [("planted", graph)]
        if include_rewired_null:
            variants.append(("rewired_null", degree_preserving_rewire(graph, seed=seed)))
        for gname, g in variants:
            for model_name in MODELS:
                pred = _fit_predict(model_name, g, y, masks, epochs, seed)
                m = evaluate(pred, y_t, masks[2])
                rows.append(
                    {"graph": gname, "model": model_name, "seed": seed, **m}
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std over seeds per (graph, model)."""
    return (
        results.groupby(["graph", "model"])[["r2", "pearson"]]
        .agg(["mean", "std"])
        .round(3)
    )


def main() -> None:
    results = run_benchmark()
    summary = summarize(results)
    print("Benchmark results (test set, mean +/- std over seeds):")
    print(summary.to_string())
    out = "benchmark_results.csv"
    results.to_csv(out, index=False)
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    main()
