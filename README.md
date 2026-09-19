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

**Current status:** planning stage; MICrONS data are openly available (microns-explorer.org); no experiments have been run.

## What is included

| Path | Contents |
|---|---|
| `src/` | Graph construction, baseline features, and GNN utilities. |
| `tests/` | Synthetic graph tests. |
| `docs/` | Research status, methods scope, and contribution guidance. |

## Use and validation

```bash
python -m pytest -q
```

## Keywords

MICrONS, connectomics, graph neural networks, visual cortex, calcium imaging, structure-function, computational neuroscience, reproducible research.

## Documentation

- [Introduction for new readers](docs/INTRODUCTION.md)
