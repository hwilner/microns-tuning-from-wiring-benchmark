"""Tests for the paper-table generators (data-free, fast, CPU).

Core guarantees:
1. Tables are pure functions of the committed artifacts: regenerating them
   yields byte-identical files.
2. Headline numbers in the paper tables match the committed leaderboard
   summary exactly (the tables cannot silently drift from the artifacts).
3. The null-control deltas reproduce the pre-registered Table 3 values.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from wiring_tuning.tables.make_all import make_all
from wiring_tuning.tables.paper_table2 import (
    build_leaderboard,
    build_per_seed_leaderboard,
)
from wiring_tuning.tables.paper_table3 import build_null_comparison

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED = REPO_ROOT / "reports" / "tables"


@pytest.fixture()
def staged_root(tmp_path):
    """A scratch repo root containing only the committed table inputs.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the staged root.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    shutil.copy(REPO_ROOT / "data" / "manifest.json", tmp_path / "data")
    shutil.copy(REPO_ROOT / "reports" / "leaderboard_meta.json", tmp_path / "reports")
    shutil.copy(REPO_ROOT / "reports" / "leaderboard.csv", tmp_path / "reports")
    shutil.copy(
        REPO_ROOT / "reports" / "leaderboard_summary.csv", tmp_path / "reports"
    )
    return tmp_path


def test_regeneration_is_byte_identical_to_committed(staged_root):
    """Re-running make_all reproduces the committed tables byte-for-byte."""
    written = make_all(staged_root)
    assert len(written) == 12  # 4 tables x (csv, md, tex)
    for path in written:
        committed = COMMITTED / path.name
        assert committed.exists(), f"{committed.name} not committed"
        assert path.read_bytes() == committed.read_bytes(), (
            f"regenerated {committed.name} differs from the committed table"
        )


def test_headline_numbers_match_committed_summary(staged_root):
    """Paper Table 2 agrees with reports/leaderboard_summary.csv."""
    table = build_leaderboard(staged_root)
    summary = pd.read_csv(REPO_ROOT / "reports" / "leaderboard_summary.csv")
    hit = summary[
        (summary["property"] == "gosi")
        & (summary["graph"] == "microns")
        & (summary["model"] == "features_mlp")
    ].iloc[0]
    row = table[(table["Property"] == "gosi") & (table["Model"] == "features_mlp")]
    assert row.iloc[0]["R² (mean ± 95% CI)"] == (
        f"{hit['r2_mean']:.3f} ± {hit['r2_ci95']:.3f}"
    )
    assert len(table) == 12  # 3 properties x 4 models


def test_per_seed_table_covers_full_leaderboard(staged_root):
    """The per-seed supplement lists every (property, graph, model, seed)."""
    table = build_per_seed_leaderboard(staged_root)
    long = pd.read_csv(REPO_ROOT / "reports" / "leaderboard.csv")
    assert len(table) == len(long) == 72  # 24 rows x 3 seeds


def test_null_deltas_reproduce_preregistered_values(staged_root):
    """Paired-delta means reproduce docs/RESULTS.md Table 3 at 3 decimals."""
    table = build_null_comparison(staged_root).set_index("Property")
    expected = {
        "gosi": 0.001,
        "osi": -0.003,
        "pref_ori (cos 2θ encoding)": 0.004,
    }
    for prop, delta in expected.items():
        mean = float(table.loc[prop, "ΔR² (mean ± 95% CI)"].split(" ± ")[0])
        assert round(mean, 3) == pytest.approx(delta, abs=5e-4)
        assert abs(mean) < 0.01, "wiring-beyond-degree signal must stay ~0"


def test_tables_do_not_invent_rows(staged_root):
    """Every leaderboard row traces to a committed summary row."""
    table = build_leaderboard(staged_root)
    summary = pd.read_csv(REPO_ROOT / "reports" / "leaderboard_summary.csv")
    keys = {(r["property"], r["model"]) for _, r in summary.iterrows()}
    for _, row in table.iterrows():
        prop = row["Property"].split(" ")[0]
        assert (prop, row["Model"]) in keys
