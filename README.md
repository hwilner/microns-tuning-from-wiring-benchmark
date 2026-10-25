# MICrONS Tuning-from-Wiring Benchmark 

This independent research repository plans and tracks an open benchmark: predicting neuronal visual tuning from electron-microscopy synaptic connectivity in the MICrONS mm³ dataset. It provides graph-learning utilities for transparent review and extension.

## Introduction

Can a neuron's visual tuning be predicted from its synaptic wiring alone?
This repository is an open, reproducible benchmark for that question on the
MICrONS functional-connectomics release (Ding et al. 2025,
doi:10.1038/s41586-025-08840-3): 12,894 neurons coregistered between
electron microscopy and 2-photon calcium imaging, with orientation/direction
tuning labels and the synaptic graph between them. See
[docs/INTRODUCTION.md](docs/INTRODUCTION.md) for a gentle entry point.

## Extended introduction

A zero-background walkthrough of the science (connectomics, tuning curves,
graph neural networks, null models) and of every design decision in this
benchmark lives in
[docs/EXTENDED_INTRODUCTION.md](docs/EXTENDED_INTRODUCTION.md). No prior
background is assumed.

## Materials and methods

Design decisions and statistics for contributors:
[docs/METHODS.md](docs/METHODS.md); the pre-registered metrics and budget:
[docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md); data access (anonymous S3
mirror and CAVE token path): [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md);
dataset spec: [data/README.md](data/README.md).

| Component | Where |
|---|---|
| Connectome graph container + subgraph sampling | `src/wiring_tuning/graphs.py` |
| MICrONS loader (S3 download, checksums, QC, fixed splits) | `src/wiring_tuning/microns.py` |
| Synthetic connectome with planted signal + rewiring null | `src/wiring_tuning/simulate.py` |
| GNN regressor + baselines (features MLP, degree MLP, mean null) | `src/wiring_tuning/models.py` |
| Training loop (node masks, early stopping, R² / Pearson) | `src/wiring_tuning/train.py` |
| Synthetic benchmark runner | `src/wiring_tuning/benchmark.py` |
| Real-data evaluation harness | `src/wiring_tuning/harness.py` |
| Data-free tests (fast, CPU-only) | `tests/` |

## Results

Full captioned tables and validation history:
[docs/RESULTS.md](docs/RESULTS.md). Headline numbers (benchmark v1, seeds
0–2, test set; 95% CIs):

| Finding | Value | Source |
|---|---|---|
| Best tuning prediction (features_mlp on gosi/osi) | R² = 0.060, Pearson = 0.247 | `reports/leaderboard_summary.csv`, RESULTS.md Table 2 |
| GNN on the real graph | R² = 0.033–0.047 — does **not** beat the features-only baseline | RESULTS.md Table 2 |
| GNN real vs. degree-preserving rewired null | ΔR² ≤ 0.002 — wiring beyond degree adds no measurable signal | RESULTS.md Table 3 |
| Synthetic positive control | GNN R² = 0.652 planted vs. −0.058 rewired — the pipeline detects wiring signal when present | RESULTS.md Table 5 |

**Key finding (honest negative result):** at the current edge sparsity of
the coregistered graph (8,128 directed edges over 12,894 neurons; mean
in+out degree ≈ 1.26), synaptic wiring adds no measurable signal for
predicting orientation tuning beyond position/area/layer features. This is
established against a degree-preserving rewiring null, not merely a failure
to find signal.

## Research plan and issue tracking

| Planned work | Status | Tracking |
|---|---|---|
| Assemble MICrONS EM graph + co-registered functional labels | **Done**: 12,894 neurons, 8,128 edges, versioned manifest + fixed splits (`data/`) | issue #5 (closed) |
| Release evaluation harness and model artifacts | **Done**: one-command harness reproduces `reports/leaderboard_summary.csv` | issue #4 (closed) |
| Train and evaluate the tuning-prediction benchmark | **Done (v1)**: leaderboard with 95% CIs in `reports/` | issue #3 (closed) |
| Baseline features and topology nulls | **Done**: deterministic baselines + degree-preserving rewiring null | issue #2 (closed) |
| Figure scripts for the Paper 1 results package | Open — Figures 1–2 pre-registered in `docs/RESULTS.md` | issue #6 |
| Results tables for the Paper 1 results package | Open — paper tables pre-registered in `docs/RESULTS.md` | issue #7 |
| Draft manuscript outline for Paper 1 | Open | issue #8 (Depends on #6, #7) |
| Assemble results package and manuscript outline | Open (umbrella) | issue #1 (Depends on #6, #7, #8) |
| Reproducibility re-run of Paper 1 results | Open | issue #9 (Depends on #1) |
| Make the benchmark reproducible and verify the downloaded release | **Done**: seeded model initialisation, SHA-256 verification of the release | issue #10 (closed) |
| Make `harness --check` verify the committed leaderboard without destroying it | **Done**: read-only, non-vacuous reproduction check | issue #11 (closed) |
| Align the documentation with the code and add CI | **Done**: null-model contract documented, CI runs the suite and the check | issue #12 (closed) |

## Use and validation

```bash
pip install -e ".[dev]"
python -m pytest -q                          # data-free tests
python -m wiring_tuning.benchmark            # synthetic benchmark
python -m wiring_tuning.harness              # real MICrONS benchmark (downloads ~500 MB once)
python -m wiring_tuning.harness --check      # verify the committed leaderboard reproduces (read-only)
```

Latest validation (2026-10-25): 24/24 data-free tests pass (Python 3.11, CPU,
torch 2.14.1); the committed leaderboard was regenerated from the public
MICrONS release and reproduces from a fresh run; details in
`docs/RESULTS.md` Table 5.

## Keywords

MICrONS, connectomics, graph neural networks, visual cortex, calcium imaging, structure-function, computational neuroscience, reproducible research.

## Documentation

- [Introduction for new readers](docs/INTRODUCTION.md)
- [Extended introduction (zero background assumed)](docs/EXTENDED_INTRODUCTION.md)
- [Methods and design decisions](docs/METHODS.md)
- [Results (captioned tables + validation record)](docs/RESULTS.md)
- [MICrONS data access](docs/DATA_ACCESS.md)
- [Analysis plan](docs/ANALYSIS_PLAN.md)
- [Data spec](data/README.md)
- [Contributing](CONTRIBUTING.md)
