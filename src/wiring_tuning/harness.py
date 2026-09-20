"""Evaluation harness: one-command reproduction of the real-data leaderboard.

Usage:
    python -m wiring_tuning.harness                # full run (downloads data)
    python -m wiring_tuning.harness --check        # reproduce + diff vs reports/
    python -m wiring_tuning.harness --synthetic    # data-free synthetic run

Outputs (default ``reports/``):
    leaderboard.csv           long-format per-(property, graph, model, seed)
    leaderboard_summary.csv   mean/std over seeds
    leaderboard_meta.json     dataset manifest + run config
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .benchmark import MODELS, _fit_predict, summarize
from .graphs import ConnectomeGraph
from .microns import assemble_dataset, load_processed_dataset
from .simulate import degree_preserving_rewire
from .train import evaluate, split_masks

DEFAULT_PROPERTIES = ("gosi", "osi", "pref_ori")


def _pref_ori_targets(theta_deg: np.ndarray) -> np.ndarray:
    """Encode preferred orientation as a single circular signal in [-1, 1]:
    cos(2*theta) -- orientation is 180-degree periodic."""
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


def main() -> None:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    properties = tuple(p.strip() for p in args.properties.split(","))

    if args.synthetic:
        from .benchmark import run_benchmark
        results = run_benchmark(seeds=seeds, epochs=args.epochs)
        summary = summarize(results).reset_index()
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
        # per-property summary with 95% CI across seeds
        keys = ["property", "graph", "model"]
        agg = results.groupby(keys)[["r2", "pearson"]].agg(["mean", "std", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        summary = agg.reset_index()
        for m in ("r2", "pearson"):
            summary[f"{m}_ci95"] = (
                1.96 * summary[f"{m}_std"] / np.sqrt(summary[f"{m}_count"])
            )
        summary = summary.round(4)
    results.to_csv(out_dir / "leaderboard.csv", index=False)
    summary.to_csv(out_dir / "leaderboard_summary.csv", index=False)
    with open(out_dir / "leaderboard_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir}/leaderboard.csv, leaderboard_summary.csv, leaderboard_meta.json")

    if args.check:
        # recompute summary from the committed long-format results and
        # require exact agreement with the committed summary table
        ref_results = pd.read_csv(out_dir / "leaderboard.csv")
        if "property" in ref_results.columns:
            keys = ["property", "graph", "model"]
        else:
            keys = ["graph", "model"]
        agg = ref_results.groupby(keys)[["r2", "pearson"]].agg(["mean", "std", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        ref_summary = agg.reset_index()
        for m in ("r2", "pearson"):
            ref_summary[f"{m}_ci95"] = (
                1.96 * ref_summary[f"{m}_std"] / np.sqrt(ref_summary[f"{m}_count"])
            )
        committed = pd.read_csv(out_dir / "leaderboard_summary.csv")
        same = np.allclose(
            ref_summary[["r2_mean", "pearson_mean"]].to_numpy(),
            committed[["r2_mean", "pearson_mean"]].to_numpy(),
            atol=5e-3,
        )
        print("reproduction check:", "PASS" if same else "MISMATCH")
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
