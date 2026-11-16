"""Regenerate every results figure with a single command.

Usage:
    python -m wiring_tuning.figures.make_all [REPO_ROOT]

Deterministic: regenerating from unchanged committed artifacts yields
byte-identical SVGs (enforced by ``tests/test_figures.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .figure1_graph_and_labels import make_figure1
from .figure2_leaderboard import make_figure2


def make_all(root: str | Path) -> list[Path]:
    """Regenerate all paper figures under ``root/reports/figures``.

    Args:
        root: repository root containing the committed artifacts.

    Returns:
        Sorted list of all written file paths.
    """
    root = Path(root)
    return sorted([make_figure1(root), make_figure2(root)])


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
