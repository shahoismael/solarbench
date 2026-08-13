#!/usr/bin/env python3
"""
phase8_transfer_significance.py

Phase 8. Significance testing on the Phase 6 transfer matrix.

Phase 6 reports means. This script asks whether those means are separable.
Three questions, three tests, all corrected for multiple comparisons.

Q1  Is the generalization gap real?
    Paired test over sites: same-climate skill vs. transferred skill, per model.
    Paired on Site_ID, because the same site is scored under both conditions,
    which removes site difficulty from the comparison.

Q2  Do the models differ from each other under transfer?
    Pairwise MLP / LSTM / Transformer on transferred cells.

Q3  Does any model beat persistence out of climate?
    One-sample test of skill score against zero, per model per condition.
    Skill > 0 means it beats persistence; the test asks whether the sign holds.

Wilcoxon signed-rank is used rather than a t-test: site-level skill scores are
heavy-tailed and n is large enough that the paired non-parametric test is the
safer choice. Holm-Bonferroni controls the family-wise error rate within each
question. Effect sizes are reported as matched-pairs rank-biserial correlation,
because with thousands of sites a p-value alone will always look significant.

Reads   phase6_transfer_matrix_results_tuned.csv  (falls back to untuned)
Writes  phase8_significance_tests.csv
        phase8_significance_summary.txt
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = "_tuned" if os.path.isfile(
    os.path.join(HERE, "phase6_transfer_matrix_results_tuned.csv")) else ""
RES = os.path.join(HERE, f"phase6_transfer_matrix_results{SUFFIX}.csv")
if not os.path.isfile(RES):
    sys.exit(f"Missing {RES}")

CLIMATES = ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]
MODELS = ["MLP", "LSTM", "Transformer"]
ALPHA = 0.05

df = pd.read_csv(RES)
single = df[~df["Source"].str.startswith("LOCO_")].copy()
loco = df[df["Source"].str.startswith("LOCO_")].copy()
loco["HeldOut"] = loco["Source"].str.replace("LOCO_", "", regex=False)

single["Condition"] = np.where(single["Source"] == single["Target"],
                               "same-climate", "transferred")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation for Wilcoxon signed-rank.

    r = (sum of positive ranks - sum of negative ranks) / total rank sum.
    Sign follows x - y; magnitude is bounded by 1.
    """
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    pos = r[d > 0].sum()
    neg = r[d < 0].sum()
    return (pos - neg) / r.sum()


def wilcoxon_safe(a, b=None):
    """Wilcoxon signed-rank guarded against degenerate input."""
    try:
        if b is None:
            if np.all(np.asarray(a) == 0):
                return np.nan, 1.0
            return stats.wilcoxon(a, alternative="two-sided")
        d = np.asarray(a) - np.asarray(b)
        if np.all(d == 0):
            return np.nan, 1.0
        return stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        return np.nan, 1.0


rows = []
skipped = []   # climates dropped for insufficient sites, reported not hidden
MIN_SITES = 6

# ---------------------------------------------------------------- Q1: the gap
# Per model, per target climate: pair each site's same-climate skill against
# its mean transferred skill on the same target. Averaging over seeds first
# stops seed noise from inflating n.
for m in MODELS:
    sub = single[single["Model"] == m]
    for tgt in CLIMATES:
        t = sub[sub["Target"] == tgt]
        same = (t[t["Condition"] == "same-climate"]
                .groupby("Site_ID")["SkillScore"].mean())
        tran = (t[t["Condition"] == "transferred"]
                .groupby("Site_ID")["SkillScore"].mean())
        common = same.index.intersection(tran.index)
        if len(common) < MIN_SITES:
            skipped.append((m, tgt, len(common)))
            continue
        a, b = same.loc[common].values, tran.loc[common].values
        stat, p = wilcoxon_safe(a, b)
        rows.append(dict(Question="Q1_gap", Model=m, Target=tgt,
                         Comparison="same-climate vs transferred",
                         n_sites=len(common),
                         mean_A=a.mean(), mean_B=b.mean(),
                         diff=a.mean() - b.mean(),
                         effect_r=rank_biserial(a, b), p_raw=p))

# ------------------------------------------------- Q2: models under transfer
for tgt in CLIMATES:
    t = single[(single["Target"] == tgt) & (single["Condition"] == "transferred")]
    per_site = (t.groupby(["Model", "Site_ID"])["SkillScore"].mean()
                 .unstack(level=0))
    for m1, m2 in itertools.combinations(MODELS, 2):
        if m1 not in per_site.columns or m2 not in per_site.columns:
            continue
        pair = per_site[[m1, m2]].dropna()
        if len(pair) < 6:
            continue
        a, b = pair[m1].values, pair[m2].values
        stat, p = wilcoxon_safe(a, b)
        rows.append(dict(Question="Q2_models", Model=f"{m1} vs {m2}", Target=tgt,
                         Comparison="transferred skill",
                         n_sites=len(pair),
                         mean_A=a.mean(), mean_B=b.mean(),
                         diff=a.mean() - b.mean(),
                         effect_r=rank_biserial(a, b), p_raw=p))

# ------------------------------------------- Q3: vs persistence (skill vs 0)
for m in MODELS:
    for cond, frame in [("same-climate", single[(single["Model"] == m) &
                                                (single["Condition"] == "same-climate")]),
                        ("transferred", single[(single["Model"] == m) &
                                               (single["Condition"] == "transferred")]),
                        ("LOCO", loco[loco["Model"] == m])]:
        if frame.empty:
            continue
        key = "HeldOut" if cond == "LOCO" else "Target"
        for tgt in CLIMATES:
            t = frame[frame[key] == tgt]
            if t.empty:
                continue
            s = t.groupby("Site_ID")["SkillScore"].mean().dropna()
            if len(s) < 6:
                continue
            stat, p = wilcoxon_safe(s.values)
            rows.append(dict(Question="Q3_vs_persistence", Model=m, Target=tgt,
                             Comparison=f"{cond} skill vs 0",
                             n_sites=len(s),
                             mean_A=s.mean(), mean_B=0.0,
                             diff=s.mean(),
                             effect_r=rank_biserial(s.values, np.zeros(len(s))),
                             p_raw=p))

res = pd.DataFrame(rows)
if res.empty:
    sys.exit("No tests could be run.")

# Holm correction applied within each question family
res["p_holm"] = np.nan
for q, idx in res.groupby("Question").groups.items():
    res.loc[idx, "p_holm"] = holm(res.loc[idx, "p_raw"].values)
res["significant"] = res["p_holm"] < ALPHA

res.to_csv(os.path.join(HERE, f"phase8_significance_tests{SUFFIX}.csv"), index=False)

# ------------------------------------------------------------------- summary
L = []
L.append("PHASE 8 - SIGNIFICANCE TESTS ON THE TRANSFER MATRIX")
L.append("=" * 70)
L.append(f"Source: {os.path.basename(RES)}")
L.append("Wilcoxon signed-rank, paired on site. Holm-Bonferroni within each family.")
L.append(f"alpha = {ALPHA}. effect_r = matched-pairs rank-biserial correlation.")
if skipped:
    L.append("")
    L.append(f"NOT TESTED (fewer than {MIN_SITES} sites, so no paired test is possible):")
    seen = {}
    for m, tgt, n in skipped:
        seen.setdefault(tgt, n)
    for tgt, n in seen.items():
        L.append(f"  {tgt}: n={n} site(s). Excluded from Q1 for every model.")
    L.append("  This is a data-coverage limit of the source dataset, not a result.")
L.append("")

q1 = res[res.Question == "Q1_gap"]
L.append("-- Q1  Is the generalization gap real? (same-climate vs transferred) --")
for m in MODELS:
    s = q1[q1.Model == m]
    if s.empty:
        continue
    sig = int(s.significant.sum())
    L.append(f"  {m:<12} significant on {sig}/{len(s)} target climates")
    for r in s.itertuples():
        mark = "*" if r.significant else " "
        L.append(f"     {mark} {r.Target:<8} n={r.n_sites:<4} same {r.mean_A:+.3f} "
                 f"vs transferred {r.mean_B:+.3f}  drop {r.diff:+.3f}  "
                 f"r={r.effect_r:+.2f}  p={r.p_holm:.2e}")
L.append("")

q2 = res[res.Question == "Q2_models"]
L.append("-- Q2  Do the models differ under transfer? --")
for c in q2.Model.unique():
    s = q2[q2.Model == c]
    sig = int(s.significant.sum())
    L.append(f"  {c:<26} significant on {sig}/{len(s)} climates")
L.append("")

q3 = res[res.Question == "Q3_vs_persistence"]
L.append("-- Q3  Does the model beat naive persistence? (skill vs 0) --")
for cond in ["same-climate", "transferred", "LOCO"]:
    L.append(f"  [{cond}]")
    for m in MODELS:
        s = q3[(q3.Model == m) & (q3.Comparison.str.startswith(cond))]
        if s.empty:
            continue
        # s["diff"] not s.diff -- DataFrame.diff is a method, not this column
        beats = int(((s["diff"] > 0) & s["significant"]).sum())
        loses = int(((s["diff"] < 0) & s["significant"]).sum())
        ns = int((~s["significant"]).sum())
        L.append(f"    {m:<12} beats {beats}/{len(s)}   loses {loses}/{len(s)}   "
                 f"indistinguishable {ns}/{len(s)}")
L.append("")
L.append("Full per-test detail: phase8_significance_tests%s.csv" % SUFFIX)

txt = "\n".join(L)
with open(os.path.join(HERE, f"phase8_significance_summary{SUFFIX}.txt"),
          "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print(txt)
