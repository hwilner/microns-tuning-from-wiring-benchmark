# Introduction — Paper 1: Predicting Neuronal Tuning from Wiring (MICrONS Benchmark)

**Series note:** This is **Paper 1 of 4** in the MICrONS function-from-wiring series. It is the first paper and does not build on any former paper. Papers 2 (interpretability), 3 (cross-area/cross-species generalization), and 4 (scalability methods) all build on this paper's benchmark, splits, and trained models.

## Background

The MICrONS mm³ dataset (mouse visual cortex; >200k cells, ~0.5B synapses) uniquely combines electron-microscopy connectivity with co-registered 2-photon calcium imaging of ~75–80k neurons during visual stimulation. The 2025 flagship publications were largely descriptive ("like wires to like"). A *predictive* benchmark — can wiring alone predict tuning? — is open. FlyWire's connectome-constrained model work (Lappalainen et al., Nature 2024) shows the power of this question in fly; the mammalian equivalent is untested.

## Research questions

1. How well does local synaptic connectivity predict orientation/direction tuning and receptive-field properties?
2. How much do learned graph features add over hand-crafted connectivity statistics and topology-null controls?
3. Which tuning properties are predictable from wiring at all?

## Data

| Resource | Scale | Access |
|---|---|---|
| MICrONS mm³ EM + functional | >200k cells, ~0.5B synapses, ~75k functionally co-registered | Open (microns-explorer.org) |
| Allen Visual Coding (2P/Neuropixels) | ~60k neurons / ~100k units | Open (AllenSDK/DANDI) |

## Methods

PyTorch Geometric GNNs/graph transformers; baseline connectivity statistics; degree-preserving topology nulls; fixed public train/val/test splits.

## Expected contributions

- The first open function-from-wiring benchmark on MICrONS.
- Versioned splits, harness, and models reused by Papers 2–4.

## Scope and boundary

Planning, software, and synthetic tests live here. Prediction is not mechanism; biological interpretation is deferred to Paper 2.
