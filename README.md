# SolarBench

A harmonized cross-climate benchmark for photovoltaic power forecasting.
Four public archives, **five Köppen climate zones**, 345 scored sites,
23,312,924 rows, one evaluation protocol.

> **The headline result.** Every deep learning model tested is useful in the
> climate it was trained on and worse than useless outside it. Pooled across
> targets, skill against naive persistence goes from **+0.055 to −0.106**
> (MLP), **+0.066 to −0.143** (LSTM) and **+0.063 to −0.180** (Transformer).
> Negative skill means the model is beaten by assuming the next value equals
> the current one.

**Status:** research code accompanying a manuscript under review. The protocol
is frozen at v1.0.0 and will not change under this major version.

---

## Why this exists

Most PV forecasting papers validate on one site, or on several sites from one
provider in one climate. That design cannot detect a failure to generalize,
because there is nothing to generalize to. SolarBench harmonizes four
independent archives into a single schema and a single protocol so that a
model can be trained in one climate and scored in another.

The four archives were not built to be compared. Making them comparable is
most of the work: different sampling intervals, different weather coverage,
different metering conventions, capacities spanning three orders of magnitude.

## The datasets

| Dataset | Location | Köppen | Sites available | Sites scored | Native res. | Rows |
|---|---|---|---|---|---|---|
| DKASC | Alice Springs, AU | BWh | 1 | 1 | 15 min | 423,510 |
| HKUST | Hong Kong | Cwa | 60 | 37 | 15 min | 4,711,971 |
| Ausgrid | New South Wales, AU | Cfa | 300 | 300 | 30 min | 15,778,512 |
| PVDAQ | United States | BSk, BWh, Cfa, Dfb | 8 | 7 | 15 min | 2,398,931 |
| **Total** | | **5 distinct classes** | **369** | **345** | | **23,312,924** |

Five distinct Köppen classes across four datasets — BWh, Cwa, Cfa, BSk and
Dfb. PVDAQ spans four classes on its own, two of which overlap DKASC's and
Ausgrid's, so its distinct contribution is the continental and cold-semi-arid
zones.

**No source data is redistributed here.** Each archive must be obtained from
its original provider. See [LICENSE-DATA](LICENSE-DATA) for terms and
attribution.

## The protocol

- **Splits** — chronological 70/15/15, no shuffling. PVDAQ uses a per-site
  split because its systems were commissioned across 2007–2023; this is a
  disclosed asymmetry.
- **Capacity normalization** — per site, `c_i = percentile_99.5(p_i > 0)`.
  Positive by construction; sites with no positive reading are excluded rather
  than divided by zero.
- **Common feature set** — five features: normalized power, plus sin/cos of
  hour and sin/cos of day-of-year. Weather is deliberately excluded, because
  cross-climate transfer is not computable without a shared input width. The
  cost of that decision is measured, not assumed — see the weather ablation.
- **Metric** — forecast skill score against naive persistence,
  `SS = 1 − RMSE_model / RMSE_persistence`, on identical test rows.
- **Rare events** — labelled by an inverter-clipping proxy and a
  cloud-transient proxy, with a 405-setting sensitivity sweep behind the
  chosen thresholds.
- **Budgets** — 100,000 training windows for every model in every condition,
  three seeds (42, 7, 123).

## What it measures

| Experiment | Design |
|---|---|
| Transfer matrix | 4 sources × 4 targets, 3 models, 3 seeds |
| Leave-one-climate-out | train on 3 pooled, evaluate on the held-out 4th |
| Horizon sweep | 15 / 30 / 60 / 180 minutes |
| Rare-event sensitivity | 405 threshold settings per dataset |
| Weather ablation | common features vs. common + weather |
| Confound controls | five controls isolating each candidate explanation |

## Results

See [leaderboard.md](leaderboard.md) for the full tables and submission rules.

**Skill rises with horizon in 12 of 12 dataset-model combinations**, from
+0.052 pooled at the shortest available horizon to +0.369 at three hours.
Persistence's dominance is a property of the one-step interval, not of the
method.

**Rare-event degradation is universal** — 60 of 60 transfer cells are worse on
rare events. The effect is carried entirely by cloud transients, positive in
540 of 540 threshold settings. The clipping proxy runs the other way: models
are *better* on clipped rows, because a flat ceiling at rated output is
trivially predictable.

**The gap is not an artifact.** Climate is collinear with sampling interval,
installation class and site count across four archives, so five controls test
each candidate explanation directly:

| Control | Gap | Verdict |
|---|---|---|
| Baseline | 0.109 | — |
| Matched resolution, all at 30 min | 0.200 | excluded — gap doubles |
| Matched site count, Ausgrid cut to 37 | 0.132 | excluded — gap widens |
| Smart persistence baseline | 13–140% worse than naive | excluded — naive was the harder baseline |
| Within-dataset, only site identity differs | 0.006 | excluded — 18× smaller |
| Capacity class within PVDAQ | 0.029 | partially attributed, about a quarter |

Four explanations excluded, one quantified. The remainder is the joint
contribution of climate zone and the data provenance that travels with it, and
four datasets cannot separate those two. Doing so needs one installation class
and one instrumentation standard deployed across multiple Köppen zones, which
no public PV dataset currently provides.

## Repository layout

```
protocol/     frozen split definitions, capacity constants, rare-event counts
results/      per-site results for every model, seed and experiment
models/       baseline model definitions
evaluation/   metrics, statistical tests, figure generation
data/         empty — place your own copies of the source archives here
*.m           MATLAB pipeline, in run order (see ENVIRONMENT.md)
```

## Reproducing

1. Obtain the four archives from their original providers (see
   [LICENSE-DATA](LICENSE-DATA)).
2. Place them under `data/` and run `run_harmonization.m`. A correct
   harmonization gives 23,312,924 rows across 369 sites.
3. Pin the evaluation with the files in `protocol/` — do not recompute the
   splits or the capacity constants.
4. Run the pipeline in the order given in [ENVIRONMENT.md](ENVIRONMENT.md).

MATLAB R2025b with the Deep Learning Toolbox, and Python 3.11. The Statistics
and Machine Learning Toolbox is not required — every statistical test runs in
Python.

## Submitting a method

Pull requests welcome. Report **both** columns: same-climate and transferred.
A method that only reports the first has not been evaluated on what this
benchmark measures. Rules in [leaderboard.md](leaderboard.md).

## Citing

See [CITATION.cff](CITATION.cff).

> Ismael Hassen, S. *SolarBench: Quantifying the Cross-Climate Generalization
> Gap in Deep Learning Photovoltaic Power Forecasting Under Rare Events.*
> Manuscript under review, 2026.

## Licence

Code is Apache-2.0 ([LICENSE](LICENSE)). Derived protocol and result files are
CC BY 4.0, subject to each source archive's own terms
([LICENSE-DATA](LICENSE-DATA)). No source data is redistributed.
