# MICrONS Tuning-from-Wiring Benchmark (Paper 1)

This independent research repository plans and tracks an open benchmark: predicting neuronal visual tuning from electron-microscopy synaptic connectivity in the MICrONS mm³ dataset. It provides graph-learning utilities for transparent review and extension.

## Series position

This is **Paper 1** of the MICrONS function-from-wiring series (4 papers). It is the foundation of the series; Papers 2–4 build on its benchmark, trained models, and data splits.

## Research plan

| Planned work | Expected outcome |
|---|---|
| Assemble MICrONS EM graph + co-registered 2-photon functional labels (~75–80k neurons) | Versioned graph + tuning-label dataset. |
| Baseline features (degree, motifs, E/I balance) vs. topology-null controls | Reference performance without learned graph features. |
| GNN/graph-transformer tuning prediction (orientation, direction, RF) | Benchmark leaderboard with uncertainty. |
| Fixed public splits and evaluation harness | Reusable benchmark adopted by Papers 2–4. |

**Current status:** benchmark software and synthetic-data tests implemented; MICrONS data are openly available (microns-explorer.org); real-connectome runs pending dataset assembly (see backlog issues).

## What is included

| Path | Contents |
|---|---|
| `src/wiring_tuning/graphs.py` | Connectome graph representation (neurons as nodes with cell-type features, synapses as weighted directed edges) and subgraph sampling. |
| `src/wiring_tuning/simulate.py` | Synthetic connectome generator with a planted tuning-from-wiring signal (like-to-like wiring), plus a degree-preserving rewiring null. |
| `src/wiring_tuning/models.py` | Pure-PyTorch message-passing GNN regressor and baselines (features-only MLP, degree/E-I-balance MLP, mean null). |
| `src/wiring_tuning/train.py` | Training loop with node-level train/val/test masks and R² / Pearson metrics. |
| `src/wiring_tuning/benchmark.py` | Benchmark runner producing a results table across models and seeds (`python -m wiring_tuning.benchmark`). |
| `tests/` | Synthetic graph tests: GNN beats the null on planted structure, collapses on rewired nulls; all fast, CPU-only, data-free. |
| `docs/` | Research status, methods scope, data-access notes, and contribution guidance. |

## Use and validation

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m wiring_tuning.benchmark
```

## Keywords

MICrONS, connectomics, graph neural networks, visual cortex, calcium imaging, structure-function, computational neuroscience, reproducible research.

## Documentation

- [Introduction for new readers](docs/INTRODUCTION.md)
- [MICrONS data access](docs/DATA_ACCESS.md)
- [Contributing](CONTRIBUTING.md)
