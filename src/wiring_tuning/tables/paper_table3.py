"""Paper Table 3: topology-null control with paired per-seed deltas.

Compares the GNN on the real MICrONS graph against the GNN on the
degree-preserving rewired null. Per-seed deltas are paired by seed
(``ΔR²(seed) = R²_real(seed) − R²_rewired(seed)``), then summarised as
mean ± 95% CI across seeds 0-2, matching ``docs/RESULTS.md`` Table 3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .formatting import format_mean_ci
from .paper_table2 import PROPERTY_LABELS


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    """Compute mean and descriptive 95% CI half-width across seeds.

    Uses the same convention as the committed harness summary
    (``reports/leaderboard_summary.csv``): ``1.96 * std / sqrt(n)`` with
    ddof=1; the CI is 0 for a single seed. This keeps the paper table
    arithmetically consistent with the versioned artifacts.

    Args:
        values: per-seed metric values.

    Returns:
        ``(mean, ci95_half_width)``.
    """
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0
    half = float(1.96 * values.std(ddof=1) / n**0.5)
    return mean, half


def build_null_comparison(root: str | Path) -> pd.DataFrame:
    """Build the real-vs-rewired GNN comparison (Paper Table 3).

    Args:
        root: repository root containing ``reports/leaderboard.csv``.

    Returns:
        One row per tuning property: GNN R² on the real graph, on the
        rewired null, the paired per-seed deltas, and their mean ± 95% CI.

    Raises:
        ValueError: if any (property, seed) pair is missing a real or
            rewired GNN row.
    """
    long = pd.read_csv(Path(root) / "reports" / "leaderboard.csv")
    gnn = long[long["model"] == "gnn"]
    rows = []
    for prop in ["gosi", "osi", "pref_ori"]:
        real = gnn[(gnn["property"] == prop) & (gnn["graph"] == "microns")]
        null = gnn[
            (gnn["property"] == prop) & (gnn["graph"] == "microns_rewired_null")
        ]
        real = real.set_index("seed").sort_index()
        null = null.set_index("seed").sort_index()
        if not real.index.equals(null.index):
            raise ValueError(
                f"seed mismatch for {prop}: real {list(real.index)} vs "
                f"rewired {list(null.index)}"
            )
        deltas = real["r2"] - null["r2"]
        real_mean, real_ci = _mean_ci(real["r2"])
        null_mean, null_ci = _mean_ci(null["r2"])
        delta_mean, delta_ci = _mean_ci(deltas)
        rows.append(
            {
                "Property": PROPERTY_LABELS[prop],
                "GNN R² real (mean ± 95% CI)": format_mean_ci(real_mean, real_ci),
                "GNN R² rewired (mean ± 95% CI)": format_mean_ci(null_mean, null_ci),
                "Per-seed ΔR²": ", ".join(
                    f"seed {s}: {d:+.4f}" for s, d in deltas.items()
                ),
                "ΔR² (mean ± 95% CI)": format_mean_ci(delta_mean, delta_ci, 4),
            }
        )
    return pd.DataFrame(rows)
