# MICrONS Tuning-from-Wiring Benchmark (Paper 1)

This independent research repository plans and tracks an open benchmark: predicting neuronal visual tuning from electron-microscopy synaptic connectivity in the MICrONS mm³ dataset. It provides graph-learning utilities for transparent review and extension.

## Series position

This is **Paper 1** of the MICrONS function-from-wiring series (4 papers). It is the foundation of the series; Papers 2–4 build on its benchmark, trained models, and data splits.

## Research plan

| Planned work | Expected outcome |
|---|---|
| Assemble MICrONS EM graph + co-registered 2-photon functional labels | **Done**: 12,894 coregistered neurons, 8,128 connected edges, versioned manifest + fixed splits (`data/`). |
| Baseline features (degree, motifs, E/I balance) vs. topology-null controls | **Done**: deterministic baselines + degree-preserving rewiring null. |
| GNN tuning prediction (orientation, direction, RF) | **Done (v1)**: leaderboard with CIs in `reports/`; wiring adds no measurable signal at current edge sparsity. |
| Fixed public splits and evaluation harness | **Done**: one-command harness (`python -m wiring_tuning.harness`) reproduces `reports/leaderboard_summary.csv`. |

**Current status:** real-data benchmark v1 complete on the MICrONS functional
connectomics release (Ding et al. 2025). Key finding: tuning properties are
only weakly predictable from structure alone (best R² ≈ 0.06, Pearson ≈ 0.25
from position/area/layer features); the GNN does not beat the features-only
baseline and is unchanged on degree-preserving rewired nulls — the
coregistered synaptic graph is extremely sparse (≈1.3 edges/neuron).

## What is included

| Path | Contents |
|---|---|
| `src/wiring_tuning/graphs.py` | Connectome graph representation (neurons as nodes with cell-type features, synapses as weighted directed edges) and subgraph sampling. |
| `src/wiring_tuning/microns.py` | Real MICrONS loader: anonymous S3 release download (parallel ranges), optional CAVE path (`CAVE_TOKEN`), dataset assembly with checksums, QC, and fixed splits. |
| `src/wiring_tuning/simulate.py` | Synthetic connectome generator with a planted tuning-from-wiring signal, plus a degree-preserving rewiring null. |
| `src/wiring_tuning/models.py` | Pure-PyTorch message-passing GNN regressor and baselines (features-only MLP, degree/E-I-balance MLP, mean null). |
| `src/wiring_tuning/train.py` | Training loop with node-level train/val/test masks and R² / Pearson metrics. |
| `src/wiring_tuning/benchmark.py` | Synthetic benchmark runner (`python -m wiring_tuning.benchmark`). |
| `src/wiring_tuning/harness.py` | Real-data evaluation harness: assembles the dataset, trains all models across seeds, writes the leaderboard (`python -m wiring_tuning.harness`, `--check` to verify reproduction, `--synthetic` for data-free). |
| `data/` | Dataset spec, manifest with SHA-256 checksums, split spot-check, 50-row sample. Raw data are downloaded, not committed. |
| `reports/` | Leaderboard: per-(property, model, seed) results and summary with 95% CIs. |
| `tests/` | Synthetic graph tests + loader tests on a fake in-memory release; fast, CPU-only, data-free. |
| `docs/` | Analysis plan (pre-registered metrics/budget), data-access notes, contribution guidance. |

## Use and validation

```bash
pip install -e ".[dev]"
python -m pytest -q                          # data-free tests
python -m wiring_tuning.benchmark            # synthetic benchmark
python -m wiring_tuning.harness              # real MICrONS benchmark (downloads ~500 MB once)
python -m wiring_tuning.harness --check      # verify committed leaderboard reproduces
```

## Keywords

MICrONS, connectomics, graph neural networks, visual cortex, calcium imaging, structure-function, computational neuroscience, reproducible research.

## Documentation

- [Introduction for new readers](docs/INTRODUCTION.md)
- [MICrONS data access](docs/DATA_ACCESS.md)
- [Analysis plan](docs/ANALYSIS_PLAN.md)
- [Data spec](data/README.md)
- [Contributing](CONTRIBUTING.md)
