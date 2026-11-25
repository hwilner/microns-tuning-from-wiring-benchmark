# Reproducibility re-run notes — Paper 1 results package

Issue: #9 (reproducibility re-run of Paper 1 results). Verification only; no
new analysis, no new numbers beyond re-measured ones.

## 2026-11-20 — clean-checkout re-run (automation run)

**Source.** Clean tarball of `main` via codeload.github.com (the sandbox
cannot reach github.com directly); installed with `pip install -e ".[dev]"`.

**Environment.** Python 3.12.12, CPU-only container (2 cores, 3 GB RAM),
numpy 2.2.5, pandas 2.3.2, matplotlib 3.10.3, torch 2.8.0+cu128 — the same
library versions as the 2026-11-16 validation row in `docs/RESULTS.md`
Table 5.

### What reproduced

| Check | Command / method | Outcome |
|---|---|---|
| Data-free test suite | `python -m pytest -q` | **34 passed / 0 failed**, 16.5 s |
| Paper tables | `python -m wiring_tuning.tables.make_all <clean copy>` | **12/12 files byte-identical** to the committed `reports/tables/` (CSV + MD + TeX, Tables 1–3) |
| Paper figures | `python -m wiring_tuning.figures.make_all <clean copy>` | **2/2 SVGs byte-identical** to the committed `reports/figures/` |
| Leaderboard internal consistency | re-derived mean/std/95% CI per (property, graph, model) from committed `reports/leaderboard.csv` | all **24 rows × 8 derived columns** match `leaderboard_summary.csv` exactly at the committed 4-decimal rounding |
| Release checksums | compared `data/manifest.json` digests against `EXPECTED_SHA256` in `src/wiring_tuning/microns.py` | both release files match; `reports/leaderboard_meta.json` carries the same digests |
| Synthetic determinism | `python -m wiring_tuning.benchmark` repeated; also at 1/2/4 torch threads | run-to-run **bit-identical** and thread-count independent on this machine |

### Deviation recorded (does not change any conclusion)

The synthetic positive control's GNN point value did **not** reproduce
cross-machine: this run measured GNN R² = **0.222 ± 0.332** (mean ± std,
seeds 0–2) on the planted graph vs. R² = 0.652 ± 0.200 recorded on
2026-10-25, while every non-GNN number matched exactly (features_mlp 0.410
on both graphs, rewired-null GNN −0.058, mean_null ≈ 0). The paired contrast
that the control exists to demonstrate is unchanged: planted 0.222 vs.
rewired −0.058, and the test-suite assertion (`gnn r2 > mean_null r2 + 0.3`,
`pearson > 0.5` on the strong-signal synthetic) passes.

Diagnosis: the synthetic run is deterministic per machine (bit-identical
across repeated processes and across 1/2/4 threads) but **not bit-stable
across machines** — float32 accumulation-order differences amplify
chaotically over 150 epochs of GNN training, and the three-seed mean of a
high-variance small-graph run (per-seed std 0.33) moves with them. The
real-data leaderboard is protected against this by the `harness --check`
tolerance (5e-3) and by deterministic kernels; the README/RESULTS synthetic
point value (0.652) is a single-machine measurement and should be read as
such. Recommendation for the scientific owner: quote the synthetic control
as the paired planted-vs-rewired contrast (the pre-registered criterion),
not the absolute R².

### Deferred (explicitly)

The full real-data `harness --check` (which re-derives the leaderboard from
the public MICrONS release) was **attempted and deferred**: the sandbox
network was throttled to ≲ 80 KB/s during this run (measured), so the
~500 MB release download (`node_data_v1.pkl` + `edge_data_v1.pkl`) was not
reachable within the run window. The download, SHA-256 verification, and
leaderboard comparison therefore remain open for a runner with normal
bandwidth; per the 2026-10-25 validation row they previously passed
bit-for-bit. This is a resource limitation of the run environment, not a
reproducibility failure.

### Acceptance-criteria status for issue #9

- [x] Clean-checkout re-run reproduces versioned figures and tables
  (byte-identical, 14/14 artifacts).
- [x] Re-run notes (environment, seeds, deviations) committed to `docs/`
  (this file; seeds 0–2 throughout, split seed 0).
- [ ] Full real-data leaderboard re-derivation (`harness --check`) —
  **deferred** on bandwidth, see above.

## 2026-11-25 — second clean-checkout verification (automation run)

**Source.** Clean tarball of `main` at `696dfd2` via codeload.github.com;
installed with `pip install -e . --no-deps` into the sandbox image.

**Environment.** Python 3.12.12, CPU-only, numpy 2.2.5, pandas 2.3.2,
matplotlib 3.10.3, torch 2.8.0+cu128 — identical library versions to the
2026-11-20 run (same sandbox image).

### What reproduced

| Check | Outcome |
|---|---|
| Data-free test suite (`python -m pytest -q`) | **34 passed / 0 failed**, 21.3 s |
| Paper tables + figures (`tables.make_all`, `figures.make_all` on the clean copy) | **14/14 regenerated artifacts byte-identical**; whole `reports/` tree identical to the committed snapshot (19/19 files) |
| Leaderboard internal consistency | re-derived summary from `leaderboard.csv` via `summarize(results, keys=("property","graph","model"))` matches `leaderboard_summary.csv` exactly (24 rows, all derived columns, committed 4-decimal rounding) |

### Synthetic positive control: environment-dependence confirmed

A second sandbox re-run of the synthetic benchmark measured the planted-graph
GNN at R² = **0.2223 ± 0.3319** (seeds 0–2) — **bit-identical to the
2026-11-20 sandbox re-run** (0.222 ± 0.332) in the same library environment,
and again far from the 0.652 ± 0.200 recorded 2026-10-25 on a different
machine. Rewired-null GNN −0.0578 and all non-GNN values also match the
2026-11-20 run exactly. This is the third data point and it pins the
deviation down: the synthetic GNN point value is deterministic *within* an
environment (same versions → same bits) but shifts *across* environments, so
it must be quoted as the paired planted-vs-rewired contrast, per the
2026-11-20 recommendation. No committed artifact is affected (the synthetic
benchmark is a control, not a results-table input).

### Real-data `harness --check`: attempted again, still bandwidth-deferred

The release download was re-attempted with a resumable parallel
range-request downloader (32 connections, 4 MB chunks, SHA-256 verification
against `EXPECTED_SHA256`). Measured throughput: **~17 KB/s single-stream,
~28–70 KB/s aggregate with 8–32 parallel connections** — the throttle is
per-IP, so parallelism does not help. At these rates the 496 MB release
(`node_data_v1.pkl` 295 MB + `edge_data_v1.pkl` 201 MB) needs an estimated
**2–5 hours**, beyond the run window; the download was left running with
resumable on-disk chunks so a later run can continue it, and is otherwise
**explicitly deferred** again. This remains a resource limitation of the run
environment, not a reproducibility failure: per the 2026-10-25 validation
row the real-data check previously passed bit-for-bit.

### Acceptance-criteria status for issue #9 (unchanged)

- [x] Clean-checkout re-run reproduces versioned figures and tables —
  reconfirmed byte-identical on a second independent run.
- [x] Re-run notes committed to `docs/` (this file).
- [ ] Full real-data leaderboard re-derivation (`harness --check`) —
  **deferred** on bandwidth (second measurement recorded above).
