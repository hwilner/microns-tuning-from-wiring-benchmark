"""Paper Table 2: full benchmark leaderboard.

Built from the committed leaderboard artifacts written by the evaluation
harness: the seed-aggregated summary (``reports/leaderboard_summary.csv``)
and the per-seed long format (``reports/leaderboard.csv``). Display rows
follow ``docs/RESULTS.md`` Table 2: real graph only, mean ± 95% CI across
seeds 0-2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .formatting import format_mean_ci

#: Display order and labels for the four benchmarked models.
MODEL_ORDER = ["features_mlp", "gnn", "degree_mlp", "mean_null"]

#: Display labels for the three tuning properties.
PROPERTY_LABELS = {
    "gosi": "gosi",
    "osi": "osi",
    "pref_ori": "pref_ori (cos 2θ encoding)",
}


def build_leaderboard(root: str | Path) -> pd.DataFrame:
    """Build the summary leaderboard (Paper Table 2, main panel).

    Args:
        root: repository root containing ``reports/leaderboard_summary.csv``.

    Returns:
        Table with one row per (property, model) on the real MICrONS graph,
        metrics formatted as mean ± 95% CI.
    """
    summary = pd.read_csv(Path(root) / "reports" / "leaderboard_summary.csv")
    real = summary[summary["graph"] == "microns"]
    rows = []
    for prop in ["gosi", "osi", "pref_ori"]:
        for model in MODEL_ORDER:
            hit = real[(real["property"] == prop) & (real["model"] == model)]
            if len(hit) != 1:
                raise ValueError(
                    f"expected exactly one summary row for ({prop}, microns, "
                    f"{model}), found {len(hit)}"
                )
            r = hit.iloc[0]
            rows.append(
                {
                    "Property": PROPERTY_LABELS[prop],
                    "Model": model,
                    "R² (mean ± 95% CI)": format_mean_ci(r["r2_mean"], r["r2_ci95"]),
                    "Pearson (mean ± 95% CI)": format_mean_ci(
                        r["pearson_mean"], r["pearson_ci95"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_per_seed_leaderboard(root: str | Path) -> pd.DataFrame:
    """Build the per-seed long-format leaderboard (Paper Table 2, supplement).

    Args:
        root: repository root containing ``reports/leaderboard.csv``.

    Returns:
        The per-seed table sorted by property, graph, model and seed, with
        metrics rounded to 6 decimals for stable serialisation.
    """
    long = pd.read_csv(Path(root) / "reports" / "leaderboard.csv")
    long = long.sort_values(
        ["property", "graph", "model", "seed"], kind="stable"
    ).reset_index(drop=True)
    long["r2"] = long["r2"].round(6)
    long["pearson"] = long["pearson"].round(6)
    return long.rename(
        columns={
            "property": "Property",
            "graph": "Graph",
            "model": "Model",
            "seed": "Seed",
            "r2": "R²",
            "pearson": "Pearson",
        }
    )
