# Data access — MICrONS public resources

This repository's code and tests run entirely on **synthetic data**
(`src/wiring_tuning/simulate.py`); no MICrONS download is required to develop
or to run CI. This note documents where the real data live and what the
loaders will expect, for the dataset-assembly task (see the backlog).

## Public resources

| Resource | What it provides | Entry point |
|---|---|---|
| MICrONS Explorer | Interactive access to the mm³ EM volume, proofread cells, synapses, and co-registered functional data | https://www.microns-explorer.org |
| CAVE (Connectome Annotation Versioning Engine) | Versioned programmatic access to the segmentation, cell tables, and synapse tables (`minnie65_public` datastack) via the `caveclient` Python package | https://caveconnectome.github.io/ |
| cloud-volume / chunkedgraph | Spatial imagery and segmentation cutouts at arbitrary resolution | https://github.com/seung-lab/cloud-volume |
| DANDI | Two-photon calcium imaging recordings (raw functional traces) | https://dandiarchive.org |

Recommended datastack: `minnie65_public` (proofread, versioned materialization).
Pin a **materialization version** and record it in the dataset manifest so all
splits and results are reproducible.

## What the loaders expect

The graph loader (`src/wiring_tuning/graphs.ConnectomeGraph`) consumes:

- **Nodes (neurons):** stable cell ID, cell type (excitatory / inhibitory
  subclass), cortical layer or depth. Node features are one-hot cell type
  plus optional morphology summaries.
- **Edges (synapses):** directed (pre → post) cell pairs; edge weight is the
  **synapse count** on that connection (CACO/synapse table aggregated to the
  cell level).
- **Labels (tuning):** per-cell functional properties from the co-registered
  2-photon sessions — orientation selectivity index (OSI), direction
  selectivity index (DSI), preferred orientation/direction, receptive-field
  parameters — keyed by the same cell ID.

Splits are **fixed by cell ID** and published in-repo as ID lists only (no
imaging or EM data are committed).

## Versioning policy

- Every dataset release records: CAVE datastack, materialization version,
  table versions, and a hash of the exported edge/node/label files.
- Proofreading state changes over time; never mix materialization versions
  within a split.

## Synthetic fallback

If MICrONS access is unavailable, `simulate.make_synthetic_connectome`
produces a graph with the same schema (cell types, directed weighted edges,
planted tuning labels) that exercises all loaders, models, and tests.
