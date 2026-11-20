# Results — MICrONS Tuning-from-Wiring Benchmark (Paper 1)

This document collects the benchmark's results in paper style: captioned
tables and figures with full provenance. Tables 1–3 are **completed** from the
committed leaderboard (`reports/leaderboard_summary.csv`,
`reports/leaderboard_meta.json`). The pre-registered Paper 1 results package
(Table 4) is **produced**: Paper Tables 1–3 and Figures 1–2 regenerate
deterministically from the committed artifacts via
`python -m wiring_tuning.tables.make_all` and
`python -m wiring_tuning.figures.make_all` (issues #6, #7, closed
2026-11-16), and the manuscript outline is committed as
`docs/MANUSCRIPT_OUTLINE.md` (issue #8, closed 2026-11-16). Table 5 records
this repository's validation history (test-suite re-runs).

## Completed results (real MICrONS data, benchmark v1)

**Table 1. Dataset summary: coregistered MICrONS functional-connectomics
graph used for benchmark v1.** Nodes are neurons coregistered between EM and
2-photon functional imaging (Ding et al. 2025, doi:10.1038/s41586-025-08840-3);
edges are directed synaptic connections between coregistered pairs with
`n_synapses >= 1` (weight = synapse count). Fixed 60/20/20 node-level splits
by nucleus ID (`data/`, split seed 0). Source file SHA-256 checksums are
recorded in `reports/leaderboard_meta.json`.

| Quantity | Value |
|---|---|
| Neurons (nodes) | 12,894 |
| Directed edges (after `n_synapses >= 1` filter) | 8,128 (of 1,693,276 raw pairs) |
| Effective graph density | 8,128 directed edges / 12,894 neurons = mean in+out degree ≈ 1.26 |
| Brain areas | V1, RL, AL, LM |
| Layers | L2/3, L4, L5 |
| Median digital-twin performance (cc_norm_cvt) | 0.661 |
| Tuning labels used | gosi, osi, pref_ori (in-silico `*_cvt_monet_full` variants) |
| Fraction of nodes with gosi label | 1.0 |

**Table 2. Benchmark leaderboard on the real MICrONS graph.** Test-set
performance predicting each tuning property from connectivity and/or node
features; mean ± 95% CI across seeds 0–2 (n = 3 seeds; descriptive CIs per
the pre-registered analysis plan — no formal test is meaningful at n = 3,
see `docs/METHODS.md`). `features_mlp` sees only standardized position and
one-hot area/layer (no wiring); `degree_mlp` sees 6 hand-crafted degree/E-I
features; `gnn` is a 2-layer message-passing network over the full graph;
`mean_null` always predicts the train-set mean. Negative R² means worse than
predicting the mean.

| Property | Model | R² (mean ± 95% CI) | Pearson (mean ± 95% CI) |
|---|---|---|---|
| gosi | features_mlp | **0.060 ± 0.015** | **0.247 ± 0.028** |
| gosi | gnn | 0.043 ± 0.007 | 0.211 ± 0.015 |
| gosi | degree_mlp | −0.010 ± 0.002 | −0.021 ± 0.012 |
| gosi | mean_null | −0.000 ± 0.001 | 0.000 ± 0.000 |
| osi | features_mlp | **0.060 ± 0.013** | **0.247 ± 0.025** |
| osi | gnn | 0.046 ± 0.010 | 0.216 ± 0.022 |
| osi | degree_mlp | −0.015 ± 0.014 | −0.022 ± 0.007 |
| osi | mean_null | −0.000 ± 0.000 | 0.000 ± 0.000 |
| pref_ori (cos 2θ encoding) | features_mlp | **0.054 ± 0.006** | **0.240 ± 0.011** |
| pref_ori (cos 2θ encoding) | gnn | 0.035 ± 0.010 | 0.198 ± 0.016 |
| pref_ori (cos 2θ encoding) | degree_mlp | −0.001 ± 0.002 | 0.010 ± 0.031 |
| pref_ori (cos 2θ encoding) | mean_null | −0.001 ± 0.001 | 0.000 ± 0.000 |

**Table 3. Topology-null control: GNN on the real graph vs. a
degree-preserving rewired null.** The rewired null destroys partner identity
by double-edge swaps (10 swaps/edge). It keeps every neuron's *unweighted*
in/out degree count exactly and the *weighted* out-degree exactly, but not
the weighted in-degree — the two target neurons of a swap exchange their
incoming synapse totals — so it is an exact control for the unweighted degree
and the weighted out-degree, not for the weighted in-degree (see the known
limitation in `docs/METHODS.md`; the `degree_mlp` rows of Table 2 differ
between the two variants for exactly that reason). ΔR² = GNN(real) −
GNN(rewired); values near zero with overlapping CIs mean specific wiring
beyond degree contributes no measurable predictive signal at this edge
density. This is the benchmark's key negative result: it separates "wiring is
uninformative here" from "the model is bad" (the same GNN recovers a planted
wiring signal on synthetic graphs, Table 5).

| Property | GNN R² (real) | GNN R² (rewired null) | ΔR² (real − rewired) |
|---|---|---|---|
| gosi | 0.043 ± 0.007 | 0.042 ± 0.008 | 0.001 |
| osi | 0.046 ± 0.010 | 0.048 ± 0.007 | −0.003 |
| pref_ori (cos 2θ) | 0.035 ± 0.010 | 0.031 ± 0.011 | +0.004 |

**Interpretation (pre-registered success criterion not met).** The analysis
plan required the GNN to beat both baselines and its rewired null *by more
than the CI*. On all three properties the GNN (i) does not beat the
features-only baseline and (ii) is unchanged on the rewired null (|ΔR²| ≤
0.004, CIs overlap). We therefore report an honest negative result: at a mean
in+out degree of ≈ 1.26 from coregistered connectivity, synaptic wiring adds
no measurable signal for predicting orientation tuning beyond
position/area/layer. This motivates the denser full-proofread graph via CAVE
(`docs/DATA_ACCESS.md`).

**Regeneration note (2026-10-25).** Tables 2 and 3 were regenerated by
re-running the committed harness on the public MICrONS release with the
deterministic training path: model initialisation is now seeded and the run
uses deterministic torch kernels, so a run is reproducible bit-for-bit (see
`docs/METHODS.md`). The dataset is bit-identical to the v1 one — same
12,894 neurons, same 8,128 edges, same manifest and checksums — and the model
scores moved by at most 0.0018 in R² and 0.0055 in Pearson relative to the
previous, non-reproducible run. The conclusion is unchanged.

## Results package (produced 2026-11-16 from the committed artifacts)

**Table 4. Paper 1 results-package tables and figures.** Every asset
regenerates deterministically from the committed artifacts with a single
command (`python -m wiring_tuning.tables.make_all`,
`python -m wiring_tuning.figures.make_all`); byte-parity is enforced by
`tests/test_tables.py` and `tests/test_figures.py`. Figure 1's label
distributions are pre-binned in `reports/figure1_label_distributions.csv`,
derived from the SHA-256-verified MICrONS release node table (provenance in
the companion `.meta.json`), so figure regeneration never needs the 295 MB
release file.

| Table/figure | Contents | Authorized by | Status |
|---|---|---|---|
| Paper Table 1 | Dataset assembly + QC (extends Table 1 above) | issue #7 | **Done** — `reports/tables/paper_table1_dataset_summary.{csv,md,tex}` |
| Paper Table 2 | Full leaderboard incl. per-seed long format | issue #7 | **Done** — `reports/tables/paper_table2_leaderboard.*` + `paper_table2_per_seed.*` |
| Paper Table 3 | Null-model comparisons incl. paired per-seed deltas | issue #7 | **Done** — `reports/tables/paper_table3_null_comparison.*` |
| Figure 1 | Graph/sparsity schematic + label distributions | issue #6 | **Done** — `reports/figures/figure1_dataset_overview.svg` |
| Figure 2 | Leaderboard bar chart with CIs + real-vs-rewired deltas | issue #6 | **Done** — `reports/figures/figure2_leaderboard.svg` |
| Manuscript outline | Intro/methods/results skeleton citing Tables 1–3 | issue #8 | **Done** — `docs/MANUSCRIPT_OUTLINE.md` |
| Umbrella: results package | Assemble tables+figures+outline into Paper 1 package | issue #1 | all children done — awaiting scientific-owner review |
| Reproducibility re-run | Fresh-environment `harness --check` + test suite | issue #9 | **Partially done 2026-11-20** — clean-checkout byte-parity re-run recorded in Table 5 and `docs/REPRODUCIBILITY_RERUN.md`; real-data leaderboard re-derivation deferred (run-environment bandwidth) |

## Validation record

**Table 5. Validation history for this repository.** Each row is one
independent re-run of the cheap, data-free test suite; heavy compute (the
real-data harness, ~500 MB download + ~4 min CPU training) is marked
deferred with its reason. Blob-SHA parity means the executed files were
verified bit-identical (`git hash-object`) to the committed blobs at the
stated ref before execution.

| Date | Ref / branch | Scope | Outcome | Environment | Notes |
|---|---|---|---|---|---|
| 2026-09-29 | `mavis/verify-and-reproduce` (root `results.md`) | Full suite (10 tests) | **10 pass / 0 fail**, 72.7 s | Python 3.11, CPU-only | Baseline verification; no benchmark metrics reproduced |
| 2026-10-25 | `main` @ 96644c5 | Full suite (10 tests) via minimal pytest shim (pytest not installable offline) | **10 pass / 0 fail**, 7.3 s wall | Python 3.12.12, CPU, numpy 2.2.5, pandas 2.3.2, torch 2.8.0 | All 10 executed files verified bit-identical by git blob-SHA parity: `__init__` 3e95d1b, `graphs` e8d610a, `models` 1c17800, `simulate` 2faf69d, `train` 75f5db2, `benchmark` 4b7cfad, `harness` 6c5d82e, `microns` bca9a9d, `test_benchmark` 1532432, `test_microns` c8f95ed |
| 2026-10-25 | `main` @ 96644c5 | Data-free synthetic benchmark end-to-end (`python -m wiring_tuning.benchmark`, 3 seeds, 300 neurons) | PASS — planted wiring signal recovered: GNN test R² 0.652 ± 0.200 / Pearson 0.820 on planted graph vs. R² −0.058 ± 0.058 on degree-preserving rewired null; features_mlp 0.410 on both (graph-blind); mean_null ≈ 0 | same as above | Positive control for Table 3's negative result: the pipeline detects wiring signal when present and is blind to it after rewiring |
| 2026-10-25 | `chore/daily-2026-10-25-microns-tuning-from-wiring-benchmark` | Real-data harness: full leaderboard re-derivation on the public MICrONS release (seeds 0–2, 150 epochs) | **PASS** — dataset reproduced bit-identically (12,894 neurons, 8,128 edges, identical manifest and SHA-256 digests); leaderboard regenerated | Python 3.11, CPU (2 threads), numpy 2.4.6, pandas 3.1.0, torch 2.14.1 | Release files verified against `EXPECTED_SHA256` before use; training run 375.9 s; the committed `reports/` are this run's output |
| 2026-10-25 | `chore/daily-2026-10-25-microns-tuning-from-wiring-benchmark` | `harness --check` against the regenerated committed leaderboard | **PASS (24 rows compared)**, exit 0; the committed `reports/leaderboard.csv` and `leaderboard_summary.csv` were byte-identical before and after | same as above | The check now compares against the committed baseline instead of against itself, and leaves the baseline untouched |
| 2026-10-25 | `chore/daily-2026-10-25-microns-tuning-from-wiring-benchmark` | Full suite (25 tests), including the new determinism, release-checksum and reproduction-check tests | **25 pass / 0 fail**; 5 consecutive fresh processes all green | Python 3.11, CPU, numpy 2.4.6, pandas 3.1.0, torch 2.14.1 | Run against the unpatched source the same suite fails 15 of 25, so the new checks are provably able to fail |
| 2026-11-16 | `chore/paper-readiness` run (issues #6, #7, #8) | Full suite (34 tests) incl. new table/figure determinism and integrity tests; `tables.make_all` + `figures.make_all` regenerated byte-identical outputs across processes (incl. different `PYTHONHASHSEED`); label distributions derived from the SHA-256-verified release node table | **34 pass / 0 fail**, 16.3 s wall | Python 3.12.12, CPU, numpy 2.2.5, pandas 2.3.2, matplotlib 3.10.3, torch 2.8.0 | Paper 1 results package produced; no benchmark metric changed (tables/figures re-format committed artifacts only). Figures are SVG because the push channel used by automation is text-only; vector output is also preferable for the manuscript. |
| 2026-11-20 | `main` clean tarball (issue #9 re-run) | Full suite (34 tests); `tables.make_all` + `figures.make_all` byte-parity from a clean copy; leaderboard internal consistency; synthetic benchmark | **34 pass / 0 fail**, 16.5 s; **14/14** paper artifacts byte-identical; 24 rows × 8 derived columns of `leaderboard_summary.csv` re-derived exactly from `leaderboard.csv`; synthetic run bit-identical across repeated processes and 1/2/4 threads | Python 3.12.12, CPU (2 cores), numpy 2.2.5, pandas 2.3.2, matplotlib 3.10.3, torch 2.8.0 | Deviation: synthetic positive-control GNN point value re-measured as R² 0.222 ± 0.332 vs. 0.652 ± 0.200 recorded 2026-10-25 — machine-dependent point value, paired planted-vs-rewired contrast unchanged (0.222 vs −0.058); see `docs/REPRODUCIBILITY_RERUN.md`. Real-data `harness --check` attempted but **deferred**: sandbox network throttled to ≲80 KB/s, ~500 MB release download unreachable within the run |

**Leaderboard internal-consistency check (2026-10-25).** Recomputing
mean/std/95% CI per (property, graph, model) from the committed long-format
`reports/leaderboard.csv` reproduces every value in the committed
`reports/leaderboard_summary.csv` exactly (all 24 rows × 8 derived columns,
match at the committed 4-decimal rounding). The committed summary is
arithmetically consistent with its per-seed raw data.

**What the reproduction check does and does not guarantee.** The committed
leaderboard was produced by a run of the committed harness on the public
release, and `harness --check` re-derives it and compares row by row.

The run is bit-for-bit reproducible: the release files are verified against
their published SHA-256 digests, the dataset assembly, the node splits and
the model initialisation are all seeded, and the benchmark runs with
deterministic torch kernels. That last part matters: the GNN's CPU
message-passing scatter-add is multi-threaded, so without deterministic
kernels two fresh runs of the same seed agreed exactly for `degree_mlp`,
`features_mlp` and `mean_null` but the GNN rows differed in the last digits
of float32 accumulation (measured: up to 2e-4 in a per-seed R² and 5e-4 in
the three-seed mean). With deterministic kernels two fresh processes produce
byte-identical leaderboards.

The comparison still uses an absolute tolerance of 5e-3 on every metric
column, as a margin for float32 arithmetic on other machines; it is not
needed for the run itself to be deterministic.
