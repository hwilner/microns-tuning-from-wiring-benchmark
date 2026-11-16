"""Tests for the paper-figure generators (data-free, fast, CPU).

Core guarantees:
1. Figures regenerate deterministically: two renders from the same
   committed artifacts are byte-identical.
2. Figure 1's label-distribution input covers every neuron in the dataset
   manifest, and the generator refuses a truncated distribution table.
3. Figure 2's null-control deltas are computed from paired seeds only;
   a seed mismatch between real and rewired runs is an error, not a plot.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from wiring_tuning.figures.figure1_graph_and_labels import make_figure1
from wiring_tuning.figures.figure2_leaderboard import make_figure2
from wiring_tuning.figures.make_all import make_all as make_all_figures

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def staged_root(tmp_path):
    """A scratch repo root containing only the committed figure inputs.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the staged root.
    """
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    shutil.copy(REPO_ROOT / "data" / "manifest.json", tmp_path / "data")
    for name in (
        "figure1_label_distributions.csv",
        "leaderboard.csv",
        "leaderboard_summary.csv",
    ):
        shutil.copy(REPO_ROOT / "reports" / name, tmp_path / "reports")
    return tmp_path


def test_figures_are_byte_deterministic(staged_root):
    """Rendering twice from unchanged artifacts yields identical bytes."""
    first = make_figure1(staged_root).read_bytes()
    second = make_figure1(staged_root).read_bytes()
    assert first == second
    first2 = make_figure2(staged_root).read_bytes()
    second2 = make_figure2(staged_root).read_bytes()
    assert first2 == second2


def test_figures_deterministic_across_processes(staged_root):
    """A fresh process with a different hash seed renders identical bytes.

    Matplotlib salts generated SVG element ids per process; the figure
    package renumbers them so regeneration is stable everywhere.
    """
    expected = {p.name: p.read_bytes() for p in make_all_figures(staged_root)}
    env = dict(os.environ, PYTHONHASHSEED="7", PYTHONPATH=str(REPO_ROOT / "src"))
    subprocess.run(
        [sys.executable, "-m", "wiring_tuning.figures.make_all", str(staged_root)],
        check=True,
        env=env,
        capture_output=True,
    )
    for name, blob in expected.items():
        assert (staged_root / "reports" / "figures" / name).read_bytes() == blob


def test_figure1_rejects_truncated_distributions(staged_root):
    """A distribution table that drops neurons is a hard error."""
    dist_path = staged_root / "reports" / "figure1_label_distributions.csv"
    dist = pd.read_csv(dist_path)
    dist.loc[dist["section"] == "histogram_gosi", "count"] -= 1
    dist.to_csv(dist_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="histogram_gosi"):
        make_figure1(staged_root)


def test_figure2_rejects_seed_mismatch(staged_root):
    """Unpaired real/rewired seeds must fail loudly, not plot."""
    long_path = staged_root / "reports" / "leaderboard.csv"
    long = pd.read_csv(long_path)
    drop = long[
        (long["property"] == "osi")
        & (long["graph"] == "microns_rewired_null")
        & (long["model"] == "gnn")
        & (long["seed"] == 2)
    ].index
    long.drop(drop).to_csv(long_path, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="seed mismatch"):
        make_figure2(staged_root)
