"""Regenerate every paper table with a single command.

Usage:
    python -m wiring_tuning.tables.make_all [REPO_ROOT]

Writes CSV + Markdown + LaTeX side by side under ``reports/tables/``.
Deterministic: regenerating from unchanged committed artifacts yields
byte-identical outputs (enforced by ``tests/test_tables.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .formatting import write_table
from .paper_table1 import build_dataset_summary
from .paper_table2 import build_leaderboard, build_per_seed_leaderboard
from .paper_table3 import build_null_comparison


def make_all(root: str | Path) -> list[Path]:
    """Regenerate all paper tables under ``root/reports/tables``.

    Args:
        root: repository root containing the committed artifacts.

    Returns:
        Sorted list of all written file paths.
    """
    root = Path(root)
    out = root / "reports" / "tables"
    written: list[Path] = []
    written += write_table(
        build_dataset_summary(root),
        out / "paper_table1_dataset_summary",
        caption=(
            "Dataset assembly and QC for the coregistered MICrONS "
            "functional-connectomics graph (benchmark v1)."
        ),
        label="tab:dataset",
    )
    written += write_table(
        build_leaderboard(root),
        out / "paper_table2_leaderboard",
        caption=(
            "Benchmark leaderboard on the real MICrONS graph; test-set "
            "mean ± 95% CI across seeds 0-2."
        ),
        label="tab:leaderboard",
    )
    written += write_table(
        build_per_seed_leaderboard(root),
        out / "paper_table2_per_seed",
        caption=(
            "Per-seed leaderboard (long format) for both the real graph "
            "and the degree-preserving rewired null."
        ),
        label="tab:perseed",
    )
    written += write_table(
        build_null_comparison(root),
        out / "paper_table3_null_comparison",
        caption=(
            "Topology-null control: GNN on the real graph vs. the "
            "degree-preserving rewired null, with paired per-seed deltas."
        ),
        label="tab:null",
    )
    return sorted(written)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: optional argument list; the single optional positional
            argument is the repository root (defaults to the parent of
            this package's ``src`` directory).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parents[3]
    for path in make_all(root):
        print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
