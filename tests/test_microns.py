"""Tests for the MICrONS loader/harness that do NOT download real data.

Uses a tiny fake release (in-memory pickles written to tmp_path) exercising
fetch-free paths: graph assembly, label extraction, manifest writing, and
processed-dataset round trip.
"""

import json

import numpy as np
import pandas as pd
import pytest

from wiring_tuning import microns


def _fake_release(tmp_path, n=40):
    rng = np.random.default_rng(0)
    nodes = pd.DataFrame(
        {
            "nucleus_id": 1000 + np.arange(n),
            "nucleus_x": rng.normal(size=n),
            "nucleus_y": rng.normal(size=n),
            "nucleus_z": rng.normal(size=n),
            "brain_area": rng.choice(["V1", "RL", "AL", "LM"], size=n),
            "layer": rng.choice(["L2/3", "L4", "L5"], size=n),
            "cc_norm_cvt": rng.uniform(0.3, 1.0, size=n),
            "gosi_cvt_monet_full": rng.uniform(size=n),
            "osi_cvt_monet_full": rng.uniform(size=n),
            "pref_ori_cvt_monet_full": rng.uniform(0, 180, size=n),
            "gosi_iv": rng.uniform(size=n),
        }
    )
    # a few invalid rows
    nodes.loc[0, "nucleus_x"] = np.nan
    edges = pd.DataFrame(
        {
            "pre_nucleus_id": rng.integers(1000, 1000 + n, size=120),
            "post_nucleus_id": rng.integers(1000, 1000 + n, size=120),
            "n_synapses": rng.integers(1, 10, size=120),
            "population": "Connected",
        }
    )
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    nodes.to_pickle(raw / microns.NODE_FILE)
    edges.to_pickle(raw / microns.EDGE_FILE)
    return nodes, edges


def test_build_connectome(tmp_path):
    nodes, edges = _fake_release(tmp_path)
    graph, labels, table = microns.build_connectome(nodes, edges, min_synapses=1)
    # node with NaN coordinate is dropped
    assert graph.num_nodes == len(nodes) - 1
    assert graph.node_features.shape[0] == graph.num_nodes
    # features: 3 xyz + 4 areas + 3 layers
    assert graph.node_features.shape[1] == 10
    assert graph.num_edges > 0
    assert set(labels) >= {"gosi", "osi", "pref_ori"}
    assert len(labels["gosi"]) == graph.num_nodes
    assert graph.edge_index.max() < graph.num_nodes


def test_assemble_and_reload(tmp_path):
    _fake_release(tmp_path)
    manifest = microns.assemble_dataset(tmp_path, seed=7)
    assert manifest["n_nodes"] > 0
    assert set(manifest["labels"]) >= {"gosi"}
    out = tmp_path / "processed"
    for f in ("graph.npz", "labels.npz", "nodes.csv", "splits.json", "manifest.json"):
        assert (out / f).exists(), f
    graph, labels, splits = microns.load_processed_dataset(out)
    assert graph.num_nodes == manifest["n_nodes"]
    n = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert n == graph.num_nodes
    assert set(splits["train"]).isdisjoint(splits["test"])
    # checksums recorded
    m = json.loads((out / "manifest.json").read_text())
    assert len(m["source"]["files"]["node_data_v1.pkl"]) == 64


def test_pref_ori_circular_encoding():
    from wiring_tuning.harness import _pref_ori_targets

    t = _pref_ori_targets(np.array([0.0, 90.0, 180.0]))
    # orientation is 180-periodic: 0 and 180 map to the same value
    assert t[0] == pytest.approx(t[2])
    assert t[0] == pytest.approx(-t[1])


def test_cave_error_message_actionable(monkeypatch):
    monkeypatch.delenv("CAVE_TOKEN", raising=False)
    try:
        import caveclient

        # avoid slow network: simulate the auth failure CAVE returns
        class _Denied(Exception):
            pass

        monkeypatch.setattr(
            caveclient, "CAVEclient",
            lambda *a, **k: (_ for _ in ()).throw(
                _Denied("401 Unauthorized: valid auth token required")
            ),
        )
        with pytest.raises(RuntimeError, match="CAVE_TOKEN"):
            microns.try_cave_client()
    except ImportError:
        with pytest.raises(RuntimeError, match="caveclient"):
            microns.try_cave_client()
