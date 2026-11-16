"""Figure scripts for the Paper 1 results package.

Each figure has its own scripted entry point and regenerates
deterministically from the committed, versioned artifacts:

- Figure 1 reads ``data/manifest.json`` and
  ``reports/figure1_label_distributions.csv`` (itself derived from the
  SHA-256-verified MICrONS release; provenance in the companion
  ``.meta.json``).
- Figure 2 reads ``reports/leaderboard_summary.csv`` and
  ``reports/leaderboard.csv``.

Run ``python -m wiring_tuning.figures.make_all`` to regenerate everything
under ``reports/figures/``.
"""

from .figure1_graph_and_labels import make_figure1
from .figure2_leaderboard import make_figure2

__all__ = ["make_figure1", "make_figure2"]
