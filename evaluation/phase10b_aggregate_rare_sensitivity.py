#!/usr/bin/env python3
"""
phase10b_aggregate_rare_sensitivity.py

Phase 10, stage B. Answers one question: does the rare-event finding depend on
the thresholds that were chosen for it?

Reads   phase10_rare_sensitivity.csv
Writes  phase10_table_sensitivity_summary.csv
        phase10_summary.txt
        fig19_rare_sensitivity.png

Run:  python phase10b_aggregate_rare_sensitivity.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "phase10_rare_sensitivity.csv")
if not os.path.isfile(SRC):
    sys.exit(f"Missing {SRC}\nRun phase10_rare_event_sensitivity.m first.")

DATASETS = ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]
RULES = ["clipping", "cloud", "combined"]

df = pd.read_csv(SRC)
df = df.dropna(subset=["Persist_pct_worse"])

# A rule that flags a large share of rows is not isolating rare conditions.
# 25% is generous; anything above it is reported but excluded from the
# robustness claim.
PREV_CAP = 25.0
df["plausible"] = df["Prevalence_pct"] <= PREV_CAP

summary = (df.groupby(["Dataset", "Rule"], as_index=False)
             .agg(settings=("Persist_pct_worse", "size"),
                  prevalence_min=("Prevalence_pct", "min"),
                  prevalence_max=("Prevalence_pct", "max"),
                  persist_worse_min=("Persist_pct_worse", "min"),
                  persist_worse_med=("Persist_pct_worse", "median"),
                  persist_worse_max=("Persist_pct_worse", "max"),
                  mlp_worse_med=("MLP_pct_worse", "median"),
                  frac_positive=("Persist_pct_worse", lambda s: float((s > 0).mean()))))
summary.to_csv(os.path.join(HERE, "phase10_table_sensitivity_summary.csv"), index=False)

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 4.0), sharey=True)
for ax, ds in zip(np.atleast_1d(axes), DATASETS):
    d = df[df.Dataset == ds]
    # "combined" is drawn first and "cloud" last: the two nearly coincide,
    # because the cloud rule flags almost every row the combined rule flags.
    # Plotting cloud on top as an open marker keeps it visible instead of
    # letting it disappear under combined, which is itself the finding.
    order = [("combined", "tab:green", dict(s=26, alpha=0.55, marker="o")),
             ("clipping", "tab:blue",  dict(s=18, alpha=0.75, marker="o")),
             ("cloud",    "tab:orange", dict(s=30, alpha=0.95, marker="x", linewidths=1.1))]
    for rule, colour, kw in order:
        r = d[d.Rule == rule]
        if r.empty:
            continue
        ax.scatter(r.Prevalence_pct, r.Persist_pct_worse, label=rule, color=colour, **kw)
    base = d[d.IsBaseline == 1]
    if not base.empty:
        ax.scatter(base.Prevalence_pct, base.Persist_pct_worse, s=120,
                   facecolors="none", edgecolors="black", linewidths=1.8,
                   label="paper's setting", zorder=5)
    ax.axhline(0, color="black", lw=1.2)
    ax.axvline(PREV_CAP, color="grey", ls=":", lw=1)
    ax.set_xlabel("Share of test rows flagged (%)")
    ax.set_title(ds, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel("Rare-event RMSE increase (%)")
np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
fig.suptitle("Rare-event degradation across every threshold tested (persistence)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig19_rare_sensitivity.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ summary
L = ["PHASE 10 - RARE-EVENT THRESHOLD SENSITIVITY", "=" * 70]
L.append(f"Settings tested per dataset: {int(df.groupby('Dataset').size().max())}")
L.append(f"Prevalence cap for the robustness claim: {PREV_CAP}% of test rows")
L.append("")

ok = df[df.plausible]
L.append("-- Does degradation survive every threshold? (persistence) --")
for ds in DATASETS:
    d = ok[ok.Dataset == ds]
    if d.empty:
        continue
    pos = int((d.Persist_pct_worse > 0).sum())
    L.append(f"  {ds:<8} positive in {pos}/{len(d)} settings   "
             f"range {d.Persist_pct_worse.min():+.1f}% .. {d.Persist_pct_worse.max():+.1f}%   "
             f"median {d.Persist_pct_worse.median():+.1f}%")
L.append("")

L.append("-- Which proxy carries the effect? (median increase, plausible settings) --")
for ds in DATASETS:
    d = ok[ok.Dataset == ds]
    if d.empty:
        continue
    parts = []
    for rule in RULES:
        r = d[d.Rule == rule]
        if r.empty:
            continue
        parts.append(f"{rule} {r.Persist_pct_worse.median():+.1f}%")
    L.append(f"  {ds:<8} " + "   ".join(parts))
L.append("")

L.append("-- Prevalence range by rule (all settings) --")
for rule in RULES:
    r = df[df.Rule == rule]
    L.append(f"  {rule:<10} {r.Prevalence_pct.min():.2f}% .. {r.Prevalence_pct.max():.2f}%")
L.append("")

base = df[df.IsBaseline == 1]
if not base.empty:
    L.append("-- The setting used in the paper --")
    for r in base.itertuples():
        L.append(f"  {r.Dataset:<8} prevalence {r.Prevalence_pct:.2f}%   "
                 f"persistence {r.Persist_pct_worse:+.1f}%   MLP {r.MLP_pct_worse:+.1f}%")
    L.append("")

allpos = int((ok.Persist_pct_worse > 0).sum())
L.append(f"OVERALL: degradation positive in {allpos}/{len(ok)} plausible settings "
         f"({100*allpos/len(ok):.1f}%).")
L.append("Figure: fig19_rare_sensitivity.png")

txt = "\n".join(L)
with open(os.path.join(HERE, "phase10_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print(txt)
