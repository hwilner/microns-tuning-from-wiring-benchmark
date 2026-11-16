# Manuscript outline — Paper 1: Tuning-from-Wiring Benchmark

**Series position:** Paper 1 of 4 in the MICrONS function-from-wiring
series. **Papers 2–4 reuse this benchmark** — the dataset assembly, fixed
splits, evaluation harness, and leaderboard protocol released here are the
shared infrastructure every later paper is scored against (see issue #4 and
`docs/ANALYSIS_PLAN.md`).

**Working title:** *A benchmark for predicting neuronal visual tuning from
synaptic wiring: an honest negative result on the coregistered MICrONS
graph.*

**Central question (pre-registered):** can a neuron's orientation tuning be
predicted from its synaptic wiring alone? Pre-registered success criterion:
the GNN beats both the features-only baseline and its degree-preserving
rewired null by more than the 95% CI. **Result: criterion not met — honest
negative result at mean in+out degree ≈ 1.26.**

## Section → asset map

| Manuscript section | Content | Results-package asset | Source artifact |
|---|---|---|---|
| 1. Introduction | Function-from-wiring question; why a benchmark with explicit nulls | — | `docs/INTRODUCTION.md`, `docs/EXTENDED_INTRODUCTION.md` |
| 2.1 Dataset | Dataset assembly + QC | Paper Table 1; Figure 1 (all panels) | `reports/tables/paper_table1_dataset_summary.*`; `reports/figures/figure1_dataset_overview.svg` |
| 2.2 Models + nulls | GNN, features/degree baselines, mean null; degree-preserving rewiring (incl. weighted-in-degree limitation) | — | `docs/METHODS.md` |
| 2.3 Protocol | Seeds 0–2, fixed splits, metrics, success criterion, determinism | — | `docs/ANALYSIS_PLAN.md`, `docs/METHODS.md` |
| 3.1 Leaderboard | No model beats features-only; all R² ≲ 0.06 | Paper Table 2 (+ per-seed supplement); Figure 2a | `reports/tables/paper_table2_*`; `reports/figures/figure2_leaderboard.svg` |
| 3.2 Topology-null control | ΔR² ≈ 0 real vs. rewired; wiring beyond degree adds no measurable signal | Paper Table 3; Figure 2b | `reports/tables/paper_table3_null_comparison.*` |
| 3.3 Positive control | Synthetic planted signal is recovered (GNN R² 0.652 vs −0.058 rewired) | cite validation record | `docs/RESULTS.md` Table 5 |
| 4. Discussion | Sparsity explanation (≈1.26 mean degree); motivates denser proofread graph via CAVE; boundary: prediction only, no mechanism | — | `docs/RESULTS.md`, `docs/DATA_ACCESS.md` |
| 5. Reproducibility | One-command harness, `--check`, checksums, deterministic kernels; validation history | — | `README.md`, `docs/RESULTS.md` Table 5 |

## Regeneration contract

- All paper tables: `python -m wiring_tuning.tables.make_all`
  → `reports/tables/` (CSV + Markdown + LaTeX).
- All paper figures: `python -m wiring_tuning.figures.make_all`
  → `reports/figures/` (SVG, deterministic).
- Both pipelines are deterministic from committed artifacts; byte-parity is
  enforced by `tests/test_tables.py` and `tests/test_figures.py`.

## Explicit non-claims (carried into the manuscript)

- No claim that wiring is uninformative in general — only that at this edge
  density (8,128 directed edges over 12,894 coregistered neurons) the
  benchmark detects no signal beyond degree and features.
- No mechanistic claim (per the pre-registered boundary).
- The rewired null controls unweighted degree and weighted out-degree, not
  weighted in-degree (known limitation, stated in `docs/METHODS.md`).

## Review gate

Scientific-owner review happens at the umbrella level (issue #1) before any
submission step. This outline is a skeleton only; no full manuscript text
exists yet.
