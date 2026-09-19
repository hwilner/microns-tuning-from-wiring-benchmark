# Contributing

Thanks for your interest in the MICrONS tuning-from-wiring benchmark
(Paper 1 of the function-from-wiring series).

## Getting started

```bash
git clone https://github.com/hwilner/microns-tuning-from-wiring-benchmark.git
cd microns-tuning-from-wiring-benchmark
pip install -e ".[dev]"
python -m pytest -q
```

All tests run on CPU in well under a minute and require no MICrONS data
(they use the synthetic planted-signal generator in
`src/wiring_tuning/simulate.py`).

## How to contribute

1. **Pick a task card.** Open issues are the project backlog; each card has a
   size estimate, acceptance criteria, and a boundary statement. Comment on
   the issue before starting work.
2. **Stay within the boundary.** Each card declares what is out of scope
   (e.g., dataset cards exclude model training). PRs that cross boundaries
   will be asked to split.
3. **Branch and PR.** Work on a feature branch, open a pull request using the
   PR template (repo-card format), and link the issue (`Closes #N`).
4. **Tests must pass.** CI runs `python -m pytest -q` on Python 3.10–3.12
   with CPU torch. Add tests for new behavior; keep them fast and data-free.

## Code conventions

- Pure PyTorch message passing (no `torch_geometric` dependency) so the
  benchmark runs anywhere with CPU torch.
- src layout: package code in `src/wiring_tuning/`, tests in `tests/`.
- Deterministic seeds everywhere; synthetic generators take an explicit
  `seed` argument.
- No large data files in the repo. MICrONS-derived artifacts (splits,
  manifests) are published as ID lists only — see `docs/DATA_ACCESS.md`.

## Reporting issues

Use the task template for new backlog cards. For bugs, include the pytest
output and the seed that reproduces the failure.
