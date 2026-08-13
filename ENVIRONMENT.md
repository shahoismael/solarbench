# Environment

Two runtimes. MATLAB trains the neural baselines and runs the transfer matrix;
Python runs the gradient-boosting baseline, every statistical test, the
confound controls, aggregation and all figures.

## MATLAB

| | |
|---|---|
| Version | R2025b |
| Required toolbox | Deep Learning Toolbox |
| Statistics and Machine Learning Toolbox | **not** required |

The Statistics and Machine Learning Toolbox is deliberately not a dependency.
Every statistical test in the paper — Wilcoxon signed-rank, Holm–Bonferroni
correction, matched-pairs rank-biserial effect size — runs in Python with
SciPy, so a MATLAB licence without that toolbox reproduces the full pipeline.

MATLAB scripts, in run order:

```
run_harmonization.m          import and harmonize the four archives
phase2_protocol_design.m     splits, capacity constants, rare-event labels
phase3_step1_persistence.m   naive persistence reference
phase3_step2_mlp.m
phase3_step3_lstm.m
phase3_step4_transformer.m
phase6a_build_transfer_cache.m
phase6b_run_transfer_matrix.m
phase7_tune_hyperparams.m    12 configs per model, dev climate HKUST
phase9_multihorizon.m        15 / 30 / 60 / 180 minutes
phase10_rare_event_sensitivity.m   405 threshold settings per dataset
```

## Python

Python 3.11. See `requirements.txt`.

```
phase8_transfer_significance.py      Wilcoxon, Holm, effect size
phase11_gbm_and_weather.py           gradient boosting + weather ablation
phase12_consolidate_numbers.py       FINAL_NUMBERS.md / .json
phase13_confound_controls.py         controls A, B, C
phase14_within_dataset_controls.py   controls D, E
phase6c_aggregate_transfer.py
phase9b_aggregate_horizons.py
phase10b_aggregate_rare_sensitivity.py
phase16_missing_figures.py
phase17_export_release_protocol.py   writes protocol/ in this repository
```

## Determinism

Three seeds throughout: **42, 7, 123**. Reported values are the mean across
seeds, with the standard deviation where stated.

Test-window sampling uses a separate fixed seed, so every model in every
experiment is scored on identical rows. Two caps apply and both matter when
reading the numbers:

- 2,000 test windows per site in the transfer matrix and the horizon sweep,
  so all sixteen transfer cells stay comparable
- 5,000 test rows per site in the rare-event sensitivity sweep

The rare-event prevalence figures quoted in the paper are measured over that
sampled set on the scored sites, not over every test row in the archive. The
counts in `protocol/rare_events_*.csv` are over **all** rows, which is why
they differ.

Training budget is capped at 100,000 windows for every model in every
condition, including the pooled leave-one-climate-out folds, so no model gains
an advantage from having more data available.

## Hardware

Results are CPU-reproducible. GPU changes wall-clock time, not the reported
metrics, beyond floating-point non-determinism smaller than the seed-to-seed
standard deviation already reported.

## Reproducing without the source data

The four source archives are not redistributed. Once you have obtained them
from their original providers and run `run_harmonization.m`, the files in
`protocol/` pin the evaluation exactly:

- `splits_<dataset>.csv` — which site went in which split, and the date
  boundary of each split
- `capacity_<dataset>.csv` — the per-site normalization constant $c_i$, the
  99.5th percentile of strictly positive output. Use these values; do not
  recompute them.
- `rare_events_<dataset>.csv` — rare-event label counts per site and split
- `protocol_summary.csv` — per-dataset totals

A correct harmonization reproduces 23,312,924 rows across 369 selected sites,
of which 345 are scored.
