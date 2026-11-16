"""Paper-table generation for the MICrONS tuning-from-wiring benchmark.

Paper Tables 1-3 are regenerated deterministically from the committed,
versioned artifacts only:

- ``data/manifest.json`` and ``reports/leaderboard_meta.json`` (dataset),
- ``reports/leaderboard_summary.csv`` (seed-aggregated leaderboard),
- ``reports/leaderboard.csv`` (per-seed long format).

No new measurements are produced here; this package re-formats and
re-derives summary statistics from numbers that were measured by the
benchmark harness (issue #3) and verified by the reproduction check
(issue #11). See ``docs/RESULTS.md`` Table 4 for the pre-registration.
"""

from .paper_table1 import build_dataset_summary
from .paper_table2 import build_leaderboard, build_per_seed_leaderboard
from .paper_table3 import build_null_comparison

__all__ = [
    "build_dataset_summary",
    "build_leaderboard",
    "build_per_seed_leaderboard",
    "build_null_comparison",
]
