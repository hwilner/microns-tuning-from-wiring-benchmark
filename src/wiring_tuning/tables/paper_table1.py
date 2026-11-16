"""Paper Table 1: dataset assembly and QC summary.

The numbers come from the committed dataset manifest (``data/manifest.json``,
written when the dataset was assembled for issue #5) and cross-checked
against the benchmark run metadata (``reports/leaderboard_meta.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_dataset_summary(root: str | Path) -> pd.DataFrame:
    """Build Paper Table 1 (dataset assembly + QC) from committed manifests.

    Args:
        root: repository root containing ``data/manifest.json`` and
            ``reports/leaderboard_meta.json``.

    Returns:
        Two-column ``Quantity``/``Value`` table.

    Raises:
        ValueError: if the dataset manifest and the run metadata disagree on
            the assembled node or edge counts.
    """
    root = Path(root)
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    meta = json.loads((root / "reports" / "leaderboard_meta.json").read_text())
    run = meta["dataset_manifest"]

    for key in ("n_nodes", "n_edges"):
        if manifest[key] != run[key]:
            raise ValueError(
                f"manifest and run metadata disagree on {key}: "
                f"{manifest[key]} != {run[key]}"
            )

    qc = manifest["qc"]
    n_nodes = manifest["n_nodes"]
    n_edges = manifest["n_edges"]
    mean_degree = 2 * n_edges / n_nodes
    rows = [
        ("Neurons (nodes, EM\u21942P coregistered)", f"{n_nodes:,}"),
        (
            f"Directed edges (n_synapses >= {manifest['min_synapses']})",
            f"{n_edges:,} (of {manifest['n_edges_raw']:,} raw pairs)",
        ),
        ("Mean in+out degree", f"\u2248 {mean_degree:.2f}"),
        ("Brain areas", ", ".join(sorted(qc["brain_areas"]))),
        ("Layers", ", ".join(qc["layers"])),
        ("Median digital-twin performance (cc_norm_cvt)", f"{qc['cc_norm_cvt_median']:.3f}"),
        (
            "Tuning labels (in-silico *_cvt_monet_full)",
            ", ".join(meta["properties"]),
        ),
        ("Fraction of nodes with gosi label", f"{qc['frac_nodes_with_gosi']:.1f}"),
        (
            "Splits",
            f"60/20/20 by nucleus ID, seed {manifest['split_seed']}",
        ),
        (
            "Release SHA-256 (node_data_v1.pkl)",
            manifest["source"]["files"]["node_data_v1.pkl"][:16] + "\u2026",
        ),
    ]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])
