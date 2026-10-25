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


@pytest.fixture
def synthetic_release(monkeypatch):
    """Accept the in-memory fake release as if it were the published one.

    The loader verifies the release files against `EXPECTED_SHA256`. The fake
    release used by these tests is not the public release, so it declares no
    published digest and is therefore digested but not verified.
    """
    monkeypatch.setattr(microns, "EXPECTED_SHA256", {})


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
    """Test build connectome.

    Args:
        tmp_path: tmp path.
    """
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


def test_assemble_and_reload(tmp_path, synthetic_release):
    """Test assemble and reload.

    Args:
        tmp_path: tmp path.
        synthetic_release: fake release digests fixture.
    """
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
    """Test pref ori circular encoding."""
    from wiring_tuning.harness import _pref_ori_targets

    t = _pref_ori_targets(np.array([0.0, 90.0, 180.0]))
    # orientation is 180-periodic: 0 and 180 map to the same value
    assert t[0] == pytest.approx(t[2])
    assert t[0] == pytest.approx(-t[1])


def test_cave_error_message_actionable(monkeypatch):
    """Test cave error message actionable.

    Args:
        monkeypatch: monkeypatch.
    """
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


def test_expected_sha256_covers_the_release_files():
    """The published digests must cover every release file that is fetched."""
    assert set(microns.EXPECTED_SHA256) == {microns.NODE_FILE, microns.EDGE_FILE}
    for digest in microns.EXPECTED_SHA256.values():
        assert len(digest) == 64
        int(digest, 16)  # must be hexadecimal


def test_verify_release_checksums_accepts_a_matching_file(tmp_path, monkeypatch):
    """A file whose digest matches the published one is accepted.

    Args:
        tmp_path: tmp path.
        monkeypatch: monkeypatch.
    """
    path = tmp_path / microns.NODE_FILE
    path.write_bytes(b"release payload")
    digest = microns.sha256_file(path)
    monkeypatch.setitem(microns.EXPECTED_SHA256, microns.NODE_FILE, digest)

    digests = microns.verify_release_checksums({microns.NODE_FILE: path})

    assert digests[microns.NODE_FILE] == digest


def test_verify_release_checksums_rejects_a_damaged_file(tmp_path):
    """A file whose digest differs from the published one is rejected.

    Args:
        tmp_path: tmp path.
    """
    path = tmp_path / microns.NODE_FILE
    path.write_bytes(b"truncated release")

    with pytest.raises(ValueError, match="SHA-256"):
        microns.verify_release_checksums({microns.NODE_FILE: path})


def test_loader_rejects_a_damaged_release(tmp_path):
    """A damaged release must fail in the loader, not reach dataset assembly.

    Args:
        tmp_path: tmp path.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / microns.NODE_FILE).write_bytes(b"corrupt")
    (raw / microns.EDGE_FILE).write_bytes(b"corrupt")
    (raw / microns.README_FILE).write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="SHA-256"):
        microns.fetch_functional_connectomics(raw)
