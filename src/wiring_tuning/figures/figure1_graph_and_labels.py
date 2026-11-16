"""Figure 1: dataset overview — graph sparsity schematic + label distributions.

Panels:
    a. Edge filtering: raw coregistered pairs vs. retained connected edges
       (log scale), annotated with node count and mean degree.
    b. Neurons per brain area.
    c. Neurons per layer (nodes without a layer assignment shown
       explicitly as "unassigned").
    d-f. Distributions of the three benchmark tuning labels (gosi, osi,
       pref_ori) across all 12,894 coregistered neurons.

Inputs are the committed artifacts ``data/manifest.json`` and
``reports/figure1_label_distributions.csv``; see the companion
``.meta.json`` for the derivation provenance of the latter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import PALETTE, save_figure


def _histogram_panel(ax, dist: pd.DataFrame, xlabel: str) -> None:
    """Draw one pre-binned label-distribution histogram.

    Args:
        ax: matplotlib axes.
        dist: rows of the figure1 table with ``bin_left``, ``bin_right``
            and ``count`` columns for a single label.
        xlabel: x-axis label.
    """
    left = dist["bin_left"].to_numpy(float)
    right = dist["bin_right"].to_numpy(float)
    counts = dist["count"].to_numpy(float)
    ax.bar(
        left,
        counts,
        width=right - left,
        align="edge",
        color=PALETTE["neutral"],
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("neurons")


def make_figure1(root: str | Path) -> Path:
    """Render Figure 1 (dataset overview) from committed artifacts.

    Args:
        root: repository root containing ``data/manifest.json`` and
            ``reports/figure1_label_distributions.csv``.

    Returns:
        Path of the written PNG.

    Raises:
        ValueError: if the label-distribution table does not cover all
            neurons recorded in the manifest.
    """
    root = Path(root)
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    dist = pd.read_csv(root / "reports" / "figure1_label_distributions.csv")
    n_nodes = manifest["n_nodes"]
    for section, frame in dist.groupby("section"):
        if int(frame["count"].sum()) != n_nodes:
            raise ValueError(
                f"section {section!r} covers {int(frame['count'].sum())} "
                f"neurons, expected {n_nodes}"
            )

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))

    ax = axes[0, 0]
    raw, kept = manifest["n_edges_raw"], manifest["n_edges"]
    bars = ax.bar(
        ["raw coregistered\npairs", "connected edges\n(kept)"],
        [raw, kept],
        color=[PALETTE["mean_null"], PALETTE["gnn"]],
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e3, top=1e9)
    ax.set_ylabel("pairs / edges (log scale)")
    ax.bar_label(bars, labels=[f"{raw:,}", f"{kept:,}"], padding=3)
    ax.set_title(
        f"a. Sparse usable graph: {n_nodes:,} neurons,\n"
        f"mean in+out degree ≈ {2 * kept / n_nodes:.2f}",
        fontsize=10,
    )

    ax = axes[0, 1]
    areas = dist[dist["section"] == "area_counts"].sort_values("category")
    ax.bar(
        areas["category"], areas["count"], color=PALETTE["features_mlp"]
    )
    ax.set_ylabel("neurons")
    ax.set_title("b. Brain areas", fontsize=10)

    ax = axes[0, 2]
    layers = dist[dist["section"] == "layer_counts"]
    order = ["L2/3", "L4", "L5", "unassigned"]
    layers = layers.set_index("category").loc[order].reset_index()
    ax.bar(layers["category"], layers["count"], color=PALETTE["degree_mlp"])
    ax.set_ylabel("neurons")
    ax.set_title("c. Layers", fontsize=10)

    _histogram_panel(
        axes[1, 0],
        dist[dist["section"] == "histogram_gosi"],
        "gosi (digital twin)",
    )
    axes[1, 0].set_title("d. gOSI distribution", fontsize=10)
    _histogram_panel(
        axes[1, 1],
        dist[dist["section"] == "histogram_osi"],
        "osi (digital twin)",
    )
    axes[1, 1].set_title("e. OSI distribution", fontsize=10)
    _histogram_panel(
        axes[1, 2],
        dist[dist["section"] == "histogram_pref_ori"],
        "preferred orientation (deg)",
    )
    axes[1, 2].set_title("f. Preferred-orientation distribution", fontsize=10)

    fig.suptitle(
        "Figure 1. MICrONS tuning-from-wiring benchmark v1: dataset overview",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, root / "reports" / "figures" / "figure1_dataset_overview.svg")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: optional argument list; the single optional positional
            argument is the repository root.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parents[3]
    print(f"wrote {make_figure1(root)}")


if __name__ == "__main__":
    main()
