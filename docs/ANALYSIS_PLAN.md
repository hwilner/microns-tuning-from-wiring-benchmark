# Analysis plan — tuning-from-wiring benchmark (Paper 1)

Pre-registered metrics and hyperparameter budget for issue #3
(`train-gnn-benchmark`). Written before the real-data runs; the numbers are
reported in `reports/leaderboard_summary.csv`.

## Data

Assembled dataset (issue #5): MICrONS functional connectomics release
(Ding et al. 2025, doi:10.1038/s41586-025-08840-3), anonymous S3 mirror
`s3://bossdb-open-data/iarpa_microns/minnie/functional_data/functional_connectomics/node_and_edge_properties/v1/`.

- Nodes: 12,894 EM↔2P coregistered neurons (V1, RL, AL, LM; layers L2/3, L4, L5).
- Edges: 8,128 directed connected pairs (`population == "Connected"`,
  `n_synapses >= 1`), weight = synapse count.
- QC: all nodes have valid nucleus coordinates; functional-signal quality
  reported via digital-twin `cc_norm_cvt` (median 0.661). No nodes were
  excluded by cc_norm in v1 (threshold 0); thresholds are a parameter of
  `assemble_dataset` for sensitivity analyses.
- Splits: node-level 60/20/20 train/val/test, fixed by nucleus ID
  (`data/splits.csv`, seed 0).

## Prediction targets (per-property)

1. `gosi` — global orientation selectivity index (digital twin, Monet).
2. `osi` — orientation selectivity index (digital twin, Monet).
3. `pref_ori` — preferred orientation, encoded as `cos(2·θ)` (orientation
   is 180°-periodic).

In-vivo variants (`*_iv`) are shipped in `labels.npz` for future work.

## Models and hyperparameter budget

Fixed budget; **no** architecture or hyperparameter search on the test set.

| Model | Inputs | Config |
|---|---|---|
| `gnn` | node features + weighted edges | 2 message-passing layers, hidden 64, lr 1e-2, ≤150 epochs, early stop patience 30 |
| `features_mlp` | node features only (position, area, layer) | hidden 64, 2 layers, same schedule |
| `degree_mlp` | 6 hand-crafted degree/E-I features | hidden 64, 1 layer |
| `mean_null` | — | train-set mean |

Nulls: `mean_null` (target null) and `microns_rewired_null`
(degree-preserving double-edge rewiring of the real graph, 10 swaps/edge).

## Metrics

- Test-set R² and Pearson correlation per (property, graph, model, seed).
- Seeds: 0, 1, 2. Uncertainty: mean ± std and 95% CI across seeds.
- Success criterion for "wiring helps": `gnn` on `microns` beats both
  `features_mlp` and `gnn` on `microns_rewired_null` by more than the CI.

## Results (see reports/)

- All tuning properties are only weakly predictable from structure alone
  (best R² ≈ 0.06, Pearson ≈ 0.25, from the features-only MLP).
- The GNN does **not** beat the features-only MLP, and its performance is
  unchanged on the degree-preserving rewired null (ΔR² ≈ 0.001).
  Interpretation: on the coregistered subset the usable synaptic graph is
  extremely sparse (8,128 edges over 12,894 neurons, ≈1.3 edges/node), so
  message passing has little signal beyond node position/area/layer.
- Explicitly unpredictable at this budget: `pref_ori` (cos 2θ) from wiring
  (R² ≈ 0.03, indistinguishable from null); `gosi`/`osi` weakly predictable
  from cell position and area, not measurably from wiring.

## Boundary

Prediction only; no mechanistic claims (deferred to Paper 2).
