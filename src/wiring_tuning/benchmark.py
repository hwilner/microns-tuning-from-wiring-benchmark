"""Benchmark runner: train models across seeds and produce a results table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .graphs import ConnectomeGraph
from .models import GNNRegressor, BaselineMLP, DegreeBaseline, MeanBaseline
from .simulate import make_synthetic_connectome, degree_preserving_rewire
from .train import deterministic_kernels, split_masks, train_model, evaluate

MODELS = ("gnn", "degree_mlp", "features_mlp", "mean_null")

#: Metrics carried by the long-format results table and its summary.
SUMMARY_METRICS = ("r2", "pearson")

#: Suffixes of the derived columns in the summary; everything else is a key.
SUMMARY_SUFFIXES = ("_mean", "_std", "_count", "_ci95")


def _fit_predict(
    model_name: str, graph: ConnectomeGraph, y, masks, epochs: int, seed: int
) -> torch.Tensor:
    train_mask, val_mask, _ = masks
    if model_name == "mean_null":
        m = MeanBaseline().fit(torch.as_tensor(y, dtype=torch.float32)[train_mask])
        return m.predict(graph.num_nodes)
    # Seed *before* the model is built. Torch draws parameter initialisation
    # from its global RNG, and `train_model` only seeds once the model already
    # exists, so an unseeded build left every run irreproducible: the same
    # seed produced different predictions on every call.
    torch.manual_seed(seed)
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
    """Run all models across seeds on the planted synthetic connectome (and optionally its degree-preserving rewired null).

    Returns a long-format results table:
    [graph, model, seed, r2, pearson] evaluated on the test mask.
    """
    rows = []
    # Deterministic kernels: the message-passing scatter-add is
    # multi-threaded on CPU, so without this the same seed can differ in the
    # last digits of float32 arithmetic between runs and between machines.
    with deterministic_kernels():
        for seed in seeds:
            graph, y = make_synthetic_connectome(n_neurons=n_neurons, seed=seed)
            masks = split_masks(graph.num_nodes, seed=seed)
            y_t = torch.as_tensor(y, dtype=torch.float32)
            variants = [("planted", graph)]
            if include_rewired_null:
                variants.append(
                    ("rewired_null", degree_preserving_rewire(graph, seed=seed))
                )
            for gname, g in variants:
                for model_name in MODELS:
                    pred = _fit_predict(model_name, g, y, masks, epochs, seed)
                    m = evaluate(pred, y_t, masks[2])
                    rows.append(
                        {"graph": gname, "model": model_name, "seed": seed, **m}
                    )
    return pd.DataFrame(rows)


def summarize(
    results: pd.DataFrame, keys: tuple[str, ...] = ("graph", "model")
) -> pd.DataFrame:
    """Aggregate a long-format results table into the published summary schema.

    For every group the summary carries the mean, standard deviation, seed
    count and 95% confidence interval of each metric. The derived columns are
    flattened to ``<metric>_mean``, ``<metric>_std``, ``<metric>_count`` and
    ``<metric>_ci95`` so that the real-data run and the data-free synthetic run
    write the same schema and can be compared column by column.

    Args:
        results: long-format rows carrying ``r2`` and ``pearson`` columns.
        keys: columns that identify one leaderboard row.

    Returns:
        pd.DataFrame: one row per group, rounded to four decimals.
    """
    agg = results.groupby(list(keys))[["r2", "pearson"]].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    summary = agg.reset_index()
    for metric in SUMMARY_METRICS:
        summary[f"{metric}_ci95"] = (
            1.96 * summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])
        )
    return summary.round(4)


def main() -> None:
    """Main."""
    results = run_benchmark()
    summary = summarize(results)
    print("Benchmark results (test set, mean +/- std over seeds):")
    print(summary.to_string())
    out = "benchmark_results.csv"
    results.to_csv(out, index=False)
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    main()
