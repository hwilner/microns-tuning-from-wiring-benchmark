# Data spec — MICrONS tuning-from-wiring dataset

**Nothing large is committed to git.** This directory contains the dataset
*specification*, checksums, a small sample, and a split spot-check.
The full dataset (including the fixed splits) is rebuilt deterministically
by the loader:

```bash
python - <<'EOF'
from wiring_tuning.microns import assemble_dataset
assemble_dataset("data", min_synapses=1, seed=0)   # downloads ~500 MB once
EOF
```

Because the source files are SHA-256-pinned and the split permutation uses a
fixed seed (0), `data/processed/splits.json` is bit-identical on every
rebuild — this pins the splits more strongly than committing an ID list.

## Source (anonymous, no token required)

MICrONS functional connectomics release accompanying Ding et al. 2025
(Nature 640:459–469, doi:10.1038/s41586-025-08840-3), mirrored publicly at:

```
https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/
```

| File | Size | SHA-256 |
|---|---|---|
| `node_data_v1.pkl` | 294,635,032 B | `5a310585ccefcfb2f14d15b2cc2e607e730b0ec0a86587e606af362463abf6ef` |
| `edge_data_v1.pkl` | 201,511,086 B | `efcd387233960026464f285597fd5bc1c3a00432e4d62570bc578149a9707bbb` |

- `node_data_v1.pkl`: 12,894 coregistered neurons (EM `nucleus_id` ↔
  2-photon `animal_id/session/scan_idx/unit_id`), nucleus coordinates (nm),
  brain area, layer, digital-twin performance (`cc_*_cvt`), and tuning
  labels (`gosi/osi/pref_ori`, in-silico `*_cvt_monet_full` and in-vivo
  `*_iv` variants).
- `edge_data_v1.pkl`: 1,693,276 neuron-pair rows; the connected graph is
  the 8,128 pairs with `population == "Connected"` / `n_synapses >= 1`.

Raw functional traces (not needed here): DANDI dandiset
[000402](https://dandiarchive.org/dandiset/000402) ("MICrONS Two Photon
Functional Imaging", Bae et al.; 19 scans, ~70 GB NWB each).

## CAVE path (optional, token required)

`wiring_tuning.microns.try_cave_client()` queries the `minnie65_public`
datastack via `caveclient`. CAVE requires a **free** auth token even for
public data — anonymous calls redirect to the sticky_auth authorize page
(verified 2026-09-20). To use it: log in at https://minnie.microns-daf.com,
create a token at https://global.daf-apis.com/auth/api/v1/create_token, and
set env var `CAVE_TOKEN` (or repo secret `CAVE_TOKEN` for CI).

## Files in this directory

| Path | Contents |
|---|---|
| `manifest.json` | source URLs, SHA-256 checksums, row counts, QC stats, split seed |
| `splits_preview.csv` | first 100 nucleus IDs of each split (spot-check); the full 60/20/20 split is deterministic via `assemble_dataset(..., seed=0)` on the checksummed source |
| `sample_nodes.csv` | 50-row sample of the assembled node table (metadata + labels) |

Processed artifacts (`data/processed/graph.npz`, `labels.npz`, `nodes.csv`,
`splits.json`) are git-ignored build products of `assemble_dataset`.

## QC

- Nodes: valid 3D nucleus coordinate required (12,894/12,894 pass).
- Functional signal: `cc_norm_cvt` median 0.661 (v1 keeps all nodes;
  threshold configurable in `assemble_dataset`).
- Proofreading status columns exist in the source but are sparsely
  annotated (148/12,894); not used for filtering in v1 — recorded in
  `manifest.json` QC notes.
