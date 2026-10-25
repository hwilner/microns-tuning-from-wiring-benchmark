"""Evaluation harness: one-command reproduction of the real-data leaderboard.

Usage:
python -m wiring_tuning.harness                # full run (downloads data)
python -m wiring_tuning.harness --check        # verify against reports/, read-only
python -m wiring_tuning.harness --synthetic    # data-free synthetic run

``--check`` never writes to the output directory: it reads the committed
``leaderboard_summary.csv`` as the baseline, runs the benchmark into a scratch
directory and compares the two. It exits non-zero when they disagree.

Outputs (default ``reports/``):
leaderboard.csv           long-format per-(property, graph, model, seed)
leaderboard_summary.csv   mean/std over seeds
leaderboard_meta.json     dataset manifest + run config
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .benchmark import (
    MODELS,
    SUMMARY_SUFFIXES,
    _fit_predict,
    run_benchmark,
    summarize,
)
from .graphs import ConnectomeGraph
from .microns import assemble_dataset, load_processed_dataset
from .simulate import degree_preserving_rewire
from .train import evaluate, split_masks

DEFAULT_PROPERTIES = ("gosi", "osi", "pref_ori")


def _pref_ori_targets(theta_deg: np.ndarray) -> np.ndarray:
    """Encode preferred orientation as a single circular signal in [-1, 1]: cos(2*theta) -- orientation is 180-degree periodic."""
    return np.cos(2.0 * np.deg2rad(theta_deg)).astype(np.float32)


def run_real_benchmark(
    graph: ConnectomeGraph,
    labels: dict[str, np.ndarray],
    properties: tuple[str, ...] = DEFAULT_PROPERTIES,
    seeds: tuple[int, ...] = (0, 1, 2),
    epochs: int = 150,
    include_rewired_null: bool = True,
) -> pd.DataFrame:
    """Train all models on the real MICrONS graph for each tuning property.

    Nodes with undefined labels (NaN) are excluded per property; masks are
    computed on the valid subset. Returns the long-format results table.
    """
    rows = []
    n = graph.num_nodes
    for prop in properties:
        if prop not in labels:
            continue
        y = labels[prop]
        if prop == "pref_ori":
            y = _pref_ori_targets(y)
        valid = np.isfinite(y)
        idx = np.where(valid)[0]
        y = y[valid]
        sub = ConnectomeGraph(
            node_features=graph.node_features[idx],
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            cell_types=graph.cell_types[idx],
            node_ids=graph.node_ids[idx],
        )
        # remap edges into the valid subset
        remap = np.full(n, -1, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        eok = (remap[sub.edge_index[0]] >= 0) & (remap[sub.edge_index[1]] >= 0)
        sub = ConnectomeGraph(
            node_features=sub.node_features,
            edge_index=remap[sub.edge_index[:, eok]],
            edge_weight=sub.edge_weight[eok],
            cell_types=sub.cell_types,
            node_ids=sub.node_ids,
        )
        for seed in seeds:
            masks = split_masks(sub.num_nodes, seed=seed)
            y_t = torch.as_tensor(y, dtype=torch.float32)
            variants = [("microns", sub)]
            if include_rewired_null:
                variants.append(
                    ("microns_rewired_null", degree_preserving_rewire(sub, seed=seed))
                )
            for gname, g in variants:
                for model_name in MODELS:
                    pred = _fit_predict(model_name, g, y, masks, epochs, seed)
                    m = evaluate(pred, y_t, masks[2])
                    rows.append(
                        {
                            "property": prop,
                            "graph": gname,
                            "model": model_name,
                            "seed": seed,
                            **m,
                        }
                    )
    return pd.DataFrame(rows)


def _compare_summaries(
    fresh: pd.DataFrame, baseline: pd.DataFrame, atol: float = 5e-3
) -> tuple[bool, str]:
    """Compare a freshly computed summary against a committed baseline.

    The two frames are matched on their key columns (every column that is not
    a derived metric) and the shared metric columns are compared elementwise.
    A baseline produced in a different mode -- for example the real-data table
    compared against a synthetic run -- has no comparable rows and is reported
    as such rather than silently passing.

    Args:
        fresh: summary produced by the current run.
        baseline: summary read from the committed artifact.
        atol: absolute tolerance for the comparison.

    Returns:
        tuple[bool, str]: whether the two summaries agree, and a
        human-readable description of what was compared.
    """
    key_cols = [c for c in fresh.columns if not c.endswith(SUMMARY_SUFFIXES)]
    metric_cols = [c for c in fresh.columns if c.endswith(SUMMARY_SUFFIXES)]
    baseline_key_cols = [
        c for c in baseline.columns if not c.endswith(SUMMARY_SUFFIXES)
    ]
    if set(key_cols) != set(baseline_key_cols):
        return False, (
            f"the committed baseline is keyed by {baseline_key_cols} but this "
            f"run is keyed by {key_cols}, so it was produced in a different "
            "mode and cannot be compared"
        )
    merged = fresh.merge(baseline, on=key_cols, how="inner", suffixes=("_new", "_ref"))
    if merged.empty:
        return False, (
            "the committed baseline shares no row with the fresh run "
            f"(keys {key_cols})"
        )
    for col in metric_cols:
        if col not in baseline.columns:
            return False, f"the committed baseline has no {col} column"
        new = merged[f"{col}_new"].to_numpy(dtype=float)
        ref = merged[f"{col}_ref"].to_numpy(dtype=float)
        if not np.allclose(new, ref, atol=atol, equal_nan=True):
            worst = float(np.nanmax(np.abs(new - ref)))
            return False, (
                f"{col} differs from the committed baseline on {len(merged)} "
                f"compared rows (largest difference {worst:.6g})"
            )
    return True, f"{len(merged)} rows compared"


def main() -> None:
    """Main."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", help="dataset directory")
    ap.add_argument("--out-dir", default="reports", help="output directory")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--min-synapses", type=int, default=1)
    ap.add_argument(
        "--properties", default=",".join(DEFAULT_PROPERTIES),
        help="comma-separated tuning properties",
    )
    ap.add_argument("--check", action="store_true",
                    help="compare freshly computed summary to committed one")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the data-free synthetic benchmark instead")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    properties = tuple(p.strip() for p in args.properties.split(","))

    # A --check run verifies the committed artifacts and must leave them
    # untouched: the baseline is read before anything is written and the fresh
    # run is materialised in a scratch directory. Writing the fresh run over
    # the baseline first (as this used to do) made the check compare the run
    # against itself, so it could never fail, and the committed leaderboard
    # was destroyed by the very command meant to verify it.
    baseline = None
    scratch_dir = None
    if args.check:
        baseline_path = out_dir / "leaderboard_summary.csv"
        if not baseline_path.exists():
            print(f"reproduction check: no committed baseline at {baseline_path}")
            raise SystemExit(1)
        baseline = pd.read_csv(baseline_path)
        scratch_dir = Path(tempfile.mkdtemp(prefix="wiring-tuning-check-"))
        write_dir = scratch_dir
    else:
        write_dir = out_dir
        write_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.synthetic:
            results = run_benchmark(seeds=seeds, epochs=args.epochs)
            summary = summarize(results)
            meta = {"mode": "synthetic", "seeds": seeds}
        else:
            processed = Path(args.data_dir) / "processed"
            if not (processed / "graph.npz").exists():
                print("assembling dataset from MICrONS release ...")
                manifest = assemble_dataset(
                    args.data_dir, min_synapses=args.min_synapses
                )
                print(json.dumps(manifest["qc"], indent=2))
            graph, labels, _ = load_processed_dataset(processed)
            t0 = time.time()
            results = run_real_benchmark(
                graph, labels, properties=properties, seeds=seeds, epochs=args.epochs
            )
            meta = {
                "mode": "real",
                "seeds": seeds,
                "epochs": args.epochs,
                "properties": properties,
                "n_nodes": graph.num_nodes,
                "n_edges": graph.num_edges,
                "runtime_s": round(time.time() - t0, 1),
            }
            with open(processed / "manifest.json") as fh:
                meta["dataset_manifest"] = json.load(fh)
            summary = summarize(results, keys=("property", "graph", "model"))
        results.to_csv(write_dir / "leaderboard.csv", index=False)
        summary.to_csv(write_dir / "leaderboard_summary.csv", index=False)
        with open(write_dir / "leaderboard_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(summary.to_string(index=False))
        print(
            f"\nwrote {write_dir}/leaderboard.csv, "
            f"{write_dir}/leaderboard_summary.csv, "
            f"{write_dir}/leaderboard_meta.json"
        )

        if args.check:
            same, detail = _compare_summaries(summary, baseline)
            print(
                "reproduction check:", "PASS" if same else "MISMATCH", f"({detail})"
            )
            if not same:
                raise SystemExit(1)
    finally:
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
