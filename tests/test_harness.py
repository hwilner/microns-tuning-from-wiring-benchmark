"""Tests for the leaderboard reproduction check of the evaluation harness.

The check is the repository's only guard on the published leaderboard, so it
has to be able to fail. These tests pin the three properties that make it
meaningful: it compares the fresh run against the *committed* baseline, it
leaves that baseline untouched, and it reports a mismatch instead of passing.
"""

import sys

import pandas as pd
import pytest

from wiring_tuning import harness


def _summary(rows):
    """Build a summary table in the published schema.

    Args:
        rows: sequence of (property, graph, model, r2_mean, pearson_mean).

    Returns:
        pd.DataFrame: a summary with the columns the harness writes.
    """
    frame = pd.DataFrame(
        rows, columns=["property", "graph", "model", "r2_mean", "pearson_mean"]
    )
    frame["r2_std"] = 0.0
    frame["r2_count"] = 3
    frame["pearson_std"] = 0.0
    frame["pearson_count"] = 3
    frame["r2_ci95"] = 0.0
    frame["pearson_ci95"] = 0.0
    return frame


def _run_harness(monkeypatch, argv):
    """Run the harness main entry point with the given arguments.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        argv: harness command-line arguments.
    """
    monkeypatch.setattr(sys, "argv", ["wiring_tuning.harness", *argv])
    harness.main()


def test_check_accepts_an_identical_baseline():
    """A baseline that agrees with the fresh run is accepted."""
    summary = _summary([("gosi", "microns", "gnn", 0.04, 0.21)])

    same, detail = harness._compare_summaries(summary, summary.copy())

    assert same, detail


def test_check_detects_a_changed_leaderboard_value():
    """A baseline that disagrees with the fresh run is reported as a mismatch."""
    summary = _summary([("gosi", "microns", "gnn", 0.04, 0.21)])
    baseline = summary.copy()
    baseline.loc[0, "r2_mean"] = 0.5

    same, detail = harness._compare_summaries(summary, baseline)

    assert not same
    assert "r2_mean" in detail


def test_check_rejects_a_baseline_from_a_different_mode():
    """A real-data baseline cannot be compared against a synthetic run."""
    baseline = _summary([("gosi", "microns", "gnn", 0.04, 0.21)])
    fresh = baseline.drop(columns=["property"])

    same, detail = harness._compare_summaries(fresh, baseline)

    assert not same
    assert "different mode" in detail


def test_check_run_leaves_the_committed_artifacts_untouched(tmp_path, monkeypatch):
    """A check run must not overwrite the leaderboard that it verifies."""
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    argv = ["--synthetic", "--out-dir", str(out_dir), "--seeds", "0", "--epochs", "1"]
    _run_harness(monkeypatch, argv)
    committed_summary = (out_dir / "leaderboard_summary.csv").read_bytes()
    committed_long = (out_dir / "leaderboard.csv").read_bytes()

    _run_harness(monkeypatch, ["--check", *argv])

    assert (out_dir / "leaderboard_summary.csv").read_bytes() == committed_summary
    assert (out_dir / "leaderboard.csv").read_bytes() == committed_long


def test_check_run_fails_on_a_corrupted_baseline(tmp_path, monkeypatch):
    """The check must be able to fail: a changed leaderboard has to be caught."""
    out_dir = tmp_path / "reports"
    out_dir.mkdir()
    argv = ["--synthetic", "--out-dir", str(out_dir), "--seeds", "0", "--epochs", "1"]
    _run_harness(monkeypatch, argv)
    summary = pd.read_csv(out_dir / "leaderboard_summary.csv")
    summary.loc[0, "r2_mean"] = 999.0
    summary.to_csv(out_dir / "leaderboard_summary.csv", index=False)

    with pytest.raises(SystemExit) as excinfo:
        _run_harness(monkeypatch, ["--check", *argv])

    assert excinfo.value.code == 1


def test_check_run_without_a_committed_baseline_fails(tmp_path, monkeypatch):
    """Without a committed baseline there is nothing to verify against."""
    out_dir = tmp_path / "reports"
    out_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        _run_harness(
            monkeypatch,
            ["--check", "--synthetic", "--out-dir", str(out_dir), "--seeds", "0"],
        )

    assert excinfo.value.code == 1
