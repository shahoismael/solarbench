# SolarBench leaderboard

Protocol version **1.0.0**. Entries produced under a different major version
are not comparable and are listed separately.

The ranking metric is the **forecast skill score against naive persistence**,
averaged over sites:

```
SS = 1 - RMSE_model / RMSE_persistence
```

Positive beats doing nothing. Negative is worse than assuming the next value
equals the current one.

Two columns matter, and a submission must report both. **Same-climate** is
trained and evaluated on the same dataset. **Transferred** is trained on one
dataset and evaluated on the other three. A method that only reports the first
column has not been evaluated on what this benchmark measures.

---

## One step ahead, tuned protocol, three seeds

| # | Method | Same-climate | Transferred | Gap | Params | Submitted |
|---|---|---|---|---|---|---|
| 1 | Gradient boosting | **+0.070** | −0.039 | **0.109** | — | baseline |
| 2 | LSTM | +0.066 | −0.143 | 0.209 | tuned | baseline |
| 3 | Transformer | +0.063 | −0.180 | 0.243 | tuned | baseline |
| 4 | MLP | +0.055 | −0.106 | 0.162 | tuned | baseline |
| 5 | Naive persistence | 0.000 | 0.000 | 0.000 | none | reference |

Gradient boosting is pooled across datasets under the confound-control
configuration; the three neural rows are pooled across the transfer matrix.

**Nothing on this list generalizes.** Every learned method is positive in the
climate it was trained on and negative outside it. Closing that gap is the
open problem.

---

## Per-dataset, same-climate, one step

| Method | DKASC | HKUST | Ausgrid | PVDAQ |
|---|---|---|---|---|
| Gradient boosting | +0.128 | +0.064 | +0.075 | +0.024 |
| LSTM | +0.092 | +0.065 | +0.068 | −0.019 |
| Transformer | +0.099 | +0.078 | +0.062 | −0.010 |
| MLP | +0.063 | +0.064 | +0.055 | +0.036 |

---

## Horizon sweep, same-climate skill

| Method | 15 min | 30 min | 60 min | 180 min |
|---|---|---|---|---|
| LSTM | +0.046 | +0.131 | +0.217 | +0.373 |
| Transformer | +0.056 | +0.116 | +0.223 | +0.369 |
| MLP | +0.054 | +0.104 | +0.185 | +0.364 |

The 15-minute column averages three datasets, not four: Ausgrid's native
resolution is 30 minutes and interpolating a 15-minute point would fabricate
readings.

Skill rises with horizon in 12 of 12 dataset-model combinations, from +0.052
pooled at the shortest available horizon to +0.369 at three hours. A
single-step result says nothing about a three-hour forecast.

---

## How to submit

Open a pull request that adds one row to the table above and one directory
under `submissions/`.

**1. Use the frozen protocol without modification.**

- Splits exactly as given in `protocol/splits_<dataset>.csv`. No reshuffling,
  no re-splitting, no tuning on test.
- Capacity normalization from `protocol/capacity_<dataset>.csv`. Do not
  recompute it.
- The common feature set only: normalized power, plus sin/cos of hour and
  sin/cos of day-of-year. Weather channels are excluded by design, because
  cross-climate transfer needs a shared input width. If your method requires
  weather, submit to the ablation table instead and say so.
- Training budget capped at 100,000 windows, matching the baselines.
- Three seeds: 42, 7, 123. Report the mean and the standard deviation.

**2. Report both columns.** Same-climate and transferred. Transferred means
every source-target pair where source ≠ target.

**3. Include in `submissions/<your-method>/`:**

- `results.csv` — one row per site, model, seed, source and target, with the
  same column names as `results/phase6_transfer_matrix_results_tuned.csv`
- `README.md` — method description, parameter count, hardware, wall-clock
  training time, and a link to the code that produced the numbers
- `environment.txt` or `requirements.txt` — exact versions

**4. State any deviation explicitly.** A deviation does not disqualify a
submission; an undisclosed one does. Entries that change the protocol are
listed in a separate table rather than removed.

**5. Significance.** For any claim that a method beats another, report the
Wilcoxon signed-rank test paired on site, Holm–Bonferroni corrected within
the family of comparisons, with the matched-pairs rank-biserial effect size.
`scripts/` contains the same test used in the paper.

Submissions are verified against the frozen protocol before merge. A
submission that cannot be reproduced from its own stated configuration is not
merged.
