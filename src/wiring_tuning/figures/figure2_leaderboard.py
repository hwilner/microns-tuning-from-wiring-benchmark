"""Figure 2: benchmark leaderboard with CIs + real-vs-rewired deltas.

Panel a: test-set R² (mean ± 95% CI across seeds 0-2) per tuning property
for the four models; the GNN is shown on both the real graph and the
degree-preserving rewired null.
Panel b: paired per-seed deltas ΔR² = GNN(real) − GNN(rewired) with their
mean ± 95% CI — the pre-registered control contrast of ``docs/RESULTS.md``
Table 3.

Inputs: ``reports/leaderboard_summary.csv`` and ``reports/leaderboard.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import PALETTE, save_figure

PROPERTIES = ["gosi", "osi", "pref_ori"]
PROPERTY_TITLES = {
    "gosi": "gosi",
    "osi": "osi",
    "pref_ori": "pref_ori (cos 2θ)",
}
#: (column key, display label, graph filter) in draw order.
SERIES = [
    ("features_mlp", "features_mlp", "microns"),
    ("gnn_real", "gnn (real graph)", "microns"),
    ("gnn_rewired", "gnn (rewired null)", "microns_rewired_null"),
    ("degree_mlp", "degree_mlp", "microns"),
    ("mean_null", "mean_null", "microns"),
]
SERIES_COLOURS = {
    "features_mlp": PALETTE["features_mlp"],
    "gnn_real": PALETTE["gnn"],
    "gnn_rewired": "#E8A17C",
    "degree_mlp": PALETTE["degree_mlp"],
    "mean_null": PALETTE["mean_null"],
}


def _summary_lookup(root: Path) -> dict[tuple[str, str, str], pd.Series]:
    """Index the committed leaderboard summary by (property, graph, model).

    Args:
        root: repository root.

    Returns:
        Mapping from ``(property, graph, model)`` to its summary row.
    """
    summary = pd.read_csv(root / "reports" / "leaderboard_summary.csv")
    return {
        (r["property"], r["graph"], r["model"]): r
        for _, r in summary.iterrows()
    }


def _paired_deltas(root: Path) -> dict[str, pd.Series]:
    """Compute per-seed paired deltas GNN(real) − GNN(rewired).

    Args:
        root: repository root.

    Returns:
        Mapping from property to the per-seed delta series.

    Raises:
        ValueError: if real and rewired seeds do not align for a property.
    """
    long = pd.read_csv(root / "reports" / "leaderboard.csv")
    gnn = long[long["model"] == "gnn"]
    out = {}
    for prop in PROPERTIES:
        real = gnn[(gnn["property"] == prop) & (gnn["graph"] == "microns")]
        null = gnn[
            (gnn["property"] == prop) & (gnn["graph"] == "microns_rewired_null")
        ]
        real = real.set_index("seed").sort_index()
        null = null.set_index("seed").sort_index()
        if not real.index.equals(null.index):
            raise ValueError(f"seed mismatch for {prop}")
        out[prop] = real["r2"] - null["r2"]
    return out


def make_figure2(root: str | Path) -> Path:
    """Render Figure 2 (leaderboard + null-control deltas).

    Args:
        root: repository root containing the committed leaderboard CSVs.

    Returns:
        Path of the written PNG.
    """
    root = Path(root)
    lookup = _summary_lookup(root)
    deltas = _paired_deltas(root)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    x = np.arange(len(PROPERTIES))
    width = 0.16
    for i, (key, label, graph) in enumerate(SERIES):
        means, cis = [], []
        for prop in PROPERTIES:
            model = key.split("_real")[0].split("_rewired")[0] if key.startswith("gnn") else key
            row = lookup[(prop, graph, model)]
            means.append(row["r2_mean"])
            cis.append(row["r2_ci95"])
        ax.bar(
            x + (i - 2) * width,
            means,
            width,
            yerr=cis,
            capsize=2.5,
            label=label,
            color=SERIES_COLOURS[key],
            error_kw={"linewidth": 0.8},
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([PROPERTY_TITLES[p] for p in PROPERTIES])
    ax.set_ylabel("test-set R² (mean ± 95% CI, seeds 0-2)")
    ax.set_title("a. Leaderboard vs. baselines and topology null", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    means = [float(deltas[p].mean()) for p in PROPERTIES]
    cis = [float(1.96 * deltas[p].std(ddof=1) / len(deltas[p]) ** 0.5) for p in PROPERTIES]
    ax2.bar(
        x,
        means,
        0.5,
        yerr=cis,
        capsize=3,
        color=PALETTE["accent"],
        error_kw={"linewidth": 0.8},
    )
    for prop_i, prop in enumerate(PROPERTIES):
        ax2.plot(
            np.full(len(deltas[prop]), prop_i),
            deltas[prop].to_numpy(),
            "o",
            color="black",
            markersize=3,
        )
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([PROPERTY_TITLES[p] for p in PROPERTIES])
    ax2.set_ylabel("ΔR² (real − rewired)")
    ax2.set_title("b. Wiring-beyond-degree signal\n(paired per-seed deltas)", fontsize=10)

    fig.suptitle(
        "Figure 2. Tuning-from-wiring benchmark v1: no measurable wiring "
        "signal beyond degree",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, root / "reports" / "figures" / "figure2_leaderboard.svg")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: optional argument list; the single optional positional
            argument is the repository root.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parents[3]
    print(f"wrote {make_figure2(root)}")


if __name__ == "__main__":
    main()
