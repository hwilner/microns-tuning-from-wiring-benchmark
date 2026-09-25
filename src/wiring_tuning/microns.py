"""MICrONS data access: functional connectomics (structure + tuning labels).

Two access paths are supported, in order of preference:

1. **Anonymous S3 release (no token needed).** The functional-connectomics
   data release accompanying Ding et al. 2025 (Nature 640:459-469,
   doi:10.1038/s41586-025-08840-3) is hosted publicly on bossdb-open-data
   (AWS Open Data). It provides, keyed by EM ``nucleus_id``:

   - ``node_data_v1.pkl``: per-neuron table with nucleus coordinates,
     brain area, layer, digital-twin performance (cc_norm_cvt), and
     orientation/direction tuning (OSI, gOSI, pref_ori; both in-silico
     ``*_cvt_monet_full`` and in-vivo ``*_iv`` variants).
   - ``edge_data_v1.pkl``: per-pair table with ``n_synapses`` between
     coregistered neurons (plus control pairs; filter with
     ``population == "Connected"`` or ``n_synapses > 0``).

2. **CAVE tables (requires a free token).** The underlying EM tables
   (``synapses_pni_2``, ``nucleus_detection_v0``, ``coregistration_*``) on
   the ``minnie65_public`` datastack can be queried with ``caveclient``.
   CAVE requires a (free) auth token even for public data: create one at
   https://global.daf-apis.com/auth/api/v1/create_token after logging in at
   https://minnie.microns-daf.com and expose it via the ``CAVE_TOKEN``
   environment variable. Anonymous access was attempted and fails: every
   materialization API endpoint redirects to the sticky_auth authorize
   page (HTTP 302 -> global.daf-apis.com/sticky_auth/api/v1/authorize).

Downloads use parallel HTTP range requests because the release files are
295 MB / 201 MB and single-stream throughput can be heavily throttled.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .graphs import ConnectomeGraph

S3_BASE = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
    "functional_data/functional_connectomics/node_and_edge_properties/v1"
)
NODE_FILE = "node_data_v1.pkl"
EDGE_FILE = "edge_data_v1.pkl"
README_FILE = "readme_v1.md"

CAVE_DATASTACK = "minnie65_public"
DANDI_DANDISET = "000402"  # MICrONS Two Photon Functional Imaging (raw NWB)

# Expected SHA-256 of the release files used to build the committed dataset.
EXPECTED_SHA256 = {
    "node_data_v1.pkl": "5a310585ccefcfb2f14d15b2cc2e607e730b0ec0a86587e606af362463abf6ef",
    "edge_data_v1.pkl": "efcd387233960026464f285597fd5bc1c3a00432e4d62570bc578149a9707bbb",
}

TUNING_COLUMNS = {
    "gosi": "gosi_cvt_monet_full",
    "osi": "osi_cvt_monet_full",
    "pref_ori": "pref_ori_cvt_monet_full",
    "gosi_invivo": "gosi_iv",
    "osi_invivo": "osi_iv",
    "pref_ori_invivo": "pref_ori_iv",
}


def _fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read: {len(data)} of {end - start + 1}")
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def download_file(
    url: str,
    dest: str | Path,
    n_conn: int = 32,
    chunk_mb: int = 4,
    verbose: bool = True,
) -> Path:
    """Parallel range-request download (works on heavily throttled links)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    chunk = chunk_mb * 1024 * 1024
    ranges = [(i, min(i + chunk - 1, size - 1)) for i in range(0, size, chunk)]
    buf: list[bytes | None] = [None] * len(ranges)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(n_conn) as ex:
        futs = {
            ex.submit(_fetch_range, url, a, b): i for i, (a, b) in enumerate(ranges)
        }
        for f in concurrent.futures.as_completed(futs):
            buf[futs[f]] = f.result()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        for data in buf:
            fh.write(data)  # type: ignore[arg-type]
    tmp.rename(dest)
    if verbose:
        dt = time.time() - t0
        print(f"downloaded {dest.name}: {size / 1e6:.1f} MB in {dt:.0f}s")
    return dest


def sha256_file(path: str | Path, chunk_mb: int = 8) -> str:
    """Sha256 file.

    Args:
        path (str | Path): path.
        chunk_mb (int): chunk mb.

    Returns:
        str: the file.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk_mb * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_functional_connectomics(raw_dir: str | Path) -> tuple[Path, Path]:
    """Download (if needed) the node/edge release files. Returns paths."""
    raw_dir = Path(raw_dir)
    node = download_file(f"{S3_BASE}/{NODE_FILE}", raw_dir / NODE_FILE)
    edge = download_file(f"{S3_BASE}/{EDGE_FILE}", raw_dir / EDGE_FILE)
    download_file(f"{S3_BASE}/{README_FILE}", raw_dir / README_FILE)
    return node, edge


def load_functional_connectomics(
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the release node and edge tables (downloading if missing)."""
    node, edge = fetch_functional_connectomics(raw_dir)
    nodes = pd.read_pickle(node)
    edges = pd.read_pickle(edge)
    return nodes, edges


def try_cave_client(datastack: str = CAVE_DATASTACK):
    """Attempt CAVE access: env token first, then anonymous (expected to fail).

    Returns a CAVEclient or raises RuntimeError explaining exactly what failed
    and how to fix it.
    """
    try:
        from caveclient import CAVEclient
    except ImportError as e:
        raise RuntimeError(
            "caveclient is not installed; `pip install caveclient` or "
            "`pip install wiring-tuning[data]`."
        ) from e
    token = os.environ.get("CAVE_TOKEN")
    try:
        return CAVEclient(datastack, auth_token=token)
    except Exception as e:
        raise RuntimeError(
            f"CAVE access failed for datastack '{datastack}' "
            f"(token {'provided via CAVE_TOKEN' if token else 'NOT set'}): {e}. "
            "CAVE requires a free auth token even for public data: log in at "
            "https://minnie.microns-daf.com, then create a token at "
            "https://global.daf-apis.com/auth/api/v1/create_token and set it "
            "as environment variable CAVE_TOKEN (or repo secret CAVE_TOKEN)."
        ) from e


def build_connectome(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    min_synapses: int = 1,
    cc_norm_min: float = 0.0,
) -> tuple[ConnectomeGraph, dict[str, np.ndarray], pd.DataFrame]:
    """Assemble a ConnectomeGraph + tuning labels from the release tables.

    Nodes are coregistered neurons with valid nucleus coordinates and digital
    twin performance ``cc_norm_cvt >= cc_norm_min`` (functional-signal QC).
    Edges are coregistered neuron pairs with ``n_synapses >= min_synapses``.

    Returns:
        graph: ConnectomeGraph with node features
            [x, y, z (standardized), one-hot brain_area, one-hot layer];
            ``cell_types`` all 0 (2P coregistered units are excitatory-cell
            dominated; EM cell-type labels require CAVE access).
        labels: dict of tuning property name -> (N,) float array
            (NaN where the property is undefined for that neuron).
        table: the filtered per-neuron DataFrame (for QC/manifest).
    """
    keep = (
        nodes["nucleus_x"].notna()
        & nodes["nucleus_y"].notna()
        & nodes["nucleus_z"].notna()
    )
    if cc_norm_min > 0:
        keep &= nodes["cc_norm_cvt"].fillna(0) >= cc_norm_min
    table = nodes.loc[keep].copy()
    table = table.reset_index(drop=True)
    nid = table["nucleus_id"].to_numpy(dtype=np.int64)
    idx_of = {int(v): i for i, v in enumerate(nid)}

    conn = edges[edges["n_synapses"] >= min_synapses]
    src = conn["pre_nucleus_id"].map(idx_of)
    dst = conn["post_nucleus_id"].map(idx_of)
    ok = src.notna() & dst.notna()
    src = src[ok].astype(np.int64).to_numpy()
    dst = dst[ok].astype(np.int64).to_numpy()
    w = conn.loc[ok, "n_synapses"].to_numpy(dtype=np.float64)
    # merge duplicate pairs (release can contain multiple population rows)
    pair = src.astype(np.uint64) << 32 | dst.astype(np.uint64)
    agg = pd.Series(w).groupby(pair).sum()
    uk = agg.index.to_numpy()
    src = (uk >> 32).astype(np.int64)
    dst = (uk & 0xFFFFFFFF).astype(np.int64)
    w = agg.to_numpy()

    xyz = table[["nucleus_x", "nucleus_y", "nucleus_z"]].to_numpy(np.float32)
    xyz = (xyz - xyz.mean(0)) / (xyz.std(0) + 1e-9)
    areas = sorted(table["brain_area"].dropna().unique())
    layers = sorted(table["layer"].dropna().unique())
    area_oh = np.zeros((len(table), len(areas)), np.float32)
    layer_oh = np.zeros((len(table), len(layers)), np.float32)
    for j, a in enumerate(areas):
        area_oh[table["brain_area"].to_numpy() == a, j] = 1.0
    for j, l in enumerate(layers):
        layer_oh[table["layer"].to_numpy() == l, j] = 1.0
    feats = np.concatenate([xyz, area_oh, layer_oh], axis=1).astype(np.float32)

    graph = ConnectomeGraph(
        node_features=feats,
        edge_index=np.stack([src, dst]).astype(np.int64),
        edge_weight=w.astype(np.float64),
        cell_types=np.zeros(len(table), dtype=np.int64),
        node_ids=nid,
    )
    labels = {
        name: table[col].to_numpy(np.float32)
        for name, col in TUNING_COLUMNS.items()
        if col in table.columns
    }
    return graph, labels, table


def assemble_dataset(
    data_dir: str | Path,
    out_dir: str | Path | None = None,
    min_synapses: int = 1,
    cc_norm_min: float = 0.0,
    seed: int = 0,
) -> dict:
    """Build the versioned dataset: graph arrays, labels, fixed splits, manifest.

    Writes to ``out_dir`` (default ``<data_dir>/processed``):
      - graph.npz (edge_index, edge_weight, node_features, node_ids)
      - labels.npz (tuning properties, NaN = undefined)
      - nodes.csv (per-neuron metadata incl. QC columns)
      - splits.json (train/val/test nucleus IDs, fixed by seed)
      - manifest.json (source URLs, SHA-256 checksums, row counts, QC stats)
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    out_dir = Path(out_dir) if out_dir else data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_path, edge_path = fetch_functional_connectomics(raw_dir)
    checksums = {
        node_path.name: sha256_file(node_path),
        edge_path.name: sha256_file(edge_path),
    }
    nodes, edges = load_functional_connectomics(raw_dir)
    graph, labels, table = build_connectome(
        nodes, edges, min_synapses=min_synapses, cc_norm_min=cc_norm_min
    )

    # Fixed splits by cell ID (splits published as IDs only)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(graph.num_nodes)
    n_train = int(0.6 * graph.num_nodes)
    n_val = int(0.2 * graph.num_nodes)
    ids = graph.node_ids
    splits = {
        "train": sorted(int(i) for i in ids[perm[:n_train]]),
        "val": sorted(int(i) for i in ids[perm[n_train : n_train + n_val]]),
        "test": sorted(int(i) for i in ids[perm[n_train + n_val :]]),
        "seed": seed,
    }

    np.savez_compressed(
        out_dir / "graph.npz",
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        node_features=graph.node_features,
        cell_types=graph.cell_types,
        node_ids=graph.node_ids,
    )
    np.savez_compressed(out_dir / "labels.npz", **labels)
    table.to_csv(out_dir / "nodes.csv", index=False)
    with open(out_dir / "splits.json", "w") as fh:
        json.dump(splits, fh)

    manifest = {
        "source": {
            "name": "MICrONS functional connectomics release (Ding et al. 2025)",
            "doi": "10.1038/s41586-025-08840-3",
            "base_url": S3_BASE,
            "files": checksums,
        },
        "n_nodes_raw": int(len(nodes)),
        "n_edges_raw": int(len(edges)),
        "n_nodes": int(graph.num_nodes),
        "n_edges": int(graph.num_edges),
        "min_synapses": min_synapses,
        "cc_norm_min": cc_norm_min,
        "labels": list(labels.keys()),
        "qc": {
            "cc_norm_cvt_median": float(table["cc_norm_cvt"].median()),
            "frac_nodes_with_gosi": float(
                np.isfinite(labels["gosi"]).mean()
            )
            if "gosi" in labels
            else None,
            "brain_areas": sorted(table["brain_area"].dropna().unique().tolist()),
            "layers": sorted(table["layer"].dropna().unique().tolist()),
        },
        "split_seed": seed,
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_processed_dataset(
    out_dir: str | Path,
) -> tuple[ConnectomeGraph, dict[str, np.ndarray], dict]:
    """Load a previously assembled dataset (graph + labels + splits)."""
    out_dir = Path(out_dir)
    g = np.load(out_dir / "graph.npz")
    graph = ConnectomeGraph(
        node_features=g["node_features"],
        edge_index=g["edge_index"],
        edge_weight=g["edge_weight"],
        cell_types=g["cell_types"],
        node_ids=g["node_ids"],
    )
    labels = {k: np.asarray(v) for k, v in np.load(out_dir / "labels.npz").items()}
    with open(out_dir / "splits.json") as fh:
        splits = json.load(fh)
    return graph, labels, splits
