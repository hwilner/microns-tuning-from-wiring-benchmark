# Data access — MICrONS public resources

This repository ships **real MICrONS data loaders** (`src/wiring_tuning/microns.py`)
plus a data-free synthetic fallback (`src/wiring_tuning/simulate.py`) used by CI.

## What works without any credentials (used by the committed dataset)

The functional-connectomics release of Ding et al. 2025 (Nature 640:459–469,
doi:10.1038/s41586-025-08840-3) is mirrored anonymously on AWS Open Data:

```
s3://bossdb-open-data/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/
```

- `node_data_v1.pkl` (295 MB): 12,894 EM↔2P coregistered neurons with nucleus
  coordinates, brain area, layer, digital-twin performance, and orientation
  tuning (OSI/gOSI/pref_ori, in-silico and in-vivo variants).
- `edge_data_v1.pkl` (202 MB): 1,693,276 neuron pairs; 8,128 connected edges
  (`n_synapses >= 1`).

`wiring_tuning.microns.assemble_dataset()` downloads these (parallel range
requests for slow links), verifies SHA-256, builds the graph + labels + fixed
splits, and writes `data/processed/` with a manifest. See `data/README.md`.

## CAVE (optional; requires a free token)

The underlying EM tables (`synapses_pni_2`, nucleus/cell-type tables,
`coregistration_*`) live on the `minnie65_public` CAVE datastack.
**CAVE requires a free auth token even for public data**: anonymous calls to
the materialization API redirect to
`global.daf-apis.com/sticky_auth/api/v1/authorize` (verified 2026-09-20).

To enable: log in at https://minnie.microns-daf.com, create a token at
https://global.daf-apis.com/auth/api/v1/create_token, then

```bash
export CAVE_TOKEN=...   # or add repo secret CAVE_TOKEN for CI
```

`wiring_tuning.microns.try_cave_client()` attempts token auth first, then
anonymous access, and raises an actionable error if both fail.

## DANDI (raw functional traces; not needed for the benchmark)

Two-photon calcium imaging recordings: dandiset
[000402](https://dandiarchive.org/dandiset/000402) — 19 NWB files of ~70 GB
each. The benchmark uses the *derived* tuning labels above; the raw NWB files
are only needed to recompute tuning from scratch.

## Public resources overview

| Resource | What it provides | Entry point |
|---|---|---|
| MICrONS Explorer | Interactive access to the mm³ EM volume and coregistered functional data | https://www.microns-explorer.org |
| bossdb-open-data (anonymous) | Functional-connectomics node/edge tables, digital-twin properties | see URLs above |
| CAVE (`minnie65_public`) | Versioned segmentation, cell tables, synapse tables (token required) | https://caveconnectome.github.io/ |
| DANDI 000402 | Raw two-photon functional imaging (NWB) | https://dandiarchive.org |

## Versioning policy

- The dataset manifest (`data/manifest.json`) records source URLs and
  SHA-256 checksums of the release files; splits are fixed by nucleus ID
  (`data/splits.csv`, seed 0).
- CAVE-based extensions must pin a materialization version; never mix
  versions within a split.

## Synthetic fallback

If MICrONS access is unavailable, `simulate.make_synthetic_connectome`
produces a graph with the same schema that exercises all loaders, models,
and tests (`python -m wiring_tuning.harness --synthetic`).
