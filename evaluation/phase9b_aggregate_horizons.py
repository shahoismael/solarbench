#!/usr/bin/env python3
"""
phase9b_aggregate_horizons.py

Phase 9, stage B. Aggregates the horizon sweep into the table and figure that
settle the persistence question.

Reads   phase9_horizon_results.csv
Writes  phase9_table_horizon_rmse.csv     mean RMSE (kW) by dataset, model, horizon
        phase9_table_horizon_skill.csv    mean skill vs persistence, same layout
        phase9_table_crossover.csv        the horizon at which each model overtakes persistence
        phase9_summary.txt
        fig17_horizon_skill.png           skill vs horizon, one panel per dataset
        fig18_horizon_rmse.png            RMSE growth vs horizon, per dataset

Run:  python phase9b_aggregate_horizons.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "phase9_horizon_results.csv")
if not os.path.isfile(SRC):
    sys.exit(f"Missing {SRC}\nRun phase9_multihorizon.m first.")

DATASETS = ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]
LEARNED = ["MLP", "LSTM", "Transformer"]
LABELS = {"DKASC": "DKASC (desert)", "HKUST": "HKUST (subtropical)",
          "Ausgrid": "Ausgrid (temperate)", "PVDAQ": "PVDAQ (mixed)"}

df = pd.read_csv(SRC)
horizons = sorted(df["Horizon_min"].unique())


def agg(frame, value):
    """Mean over sites within a seed, then mean over seeds."""
    per_seed = frame.groupby(["Dataset", "Model", "Horizon_min", "Seed"],
                             as_index=False)[value].mean()
    return (per_seed.groupby(["Dataset", "Model", "Horizon_min"])[value]
                    .agg(["mean", "std"])
                    .reset_index())


rmse = agg(df, "RMSE_kW").rename(columns={"mean": "RMSE_mean", "std": "RMSE_std"})
skill = agg(df[df.Model != "Persistence"], "SkillScore").rename(
    columns={"mean": "Skill_mean", "std": "Skill_std"})

rmse.to_csv(os.path.join(HERE, "phase9_table_horizon_rmse.csv"), index=False)
skill.to_csv(os.path.join(HERE, "phase9_table_horizon_skill.csv"), index=False)

# ----------------------------------------------------------------- crossover
# The shortest horizon at which a model's mean skill turns positive, i.e. where
# learning starts to beat doing nothing.
rows = []
for ds in DATASETS:
    for m in LEARNED:
        s = skill[(skill.Dataset == ds) & (skill.Model == m)].sort_values("Horizon_min")
        if s.empty:
            continue
        pos = s[s.Skill_mean > 0]
        rows.append({
            "Dataset": ds, "Model": m,
            "crossover_min": int(pos.Horizon_min.iloc[0]) if not pos.empty else np.nan,
            "skill_at_shortest": s.Skill_mean.iloc[0],
            "skill_at_longest": s.Skill_mean.iloc[-1],
            "shortest_min": int(s.Horizon_min.iloc[0]),
            "longest_min": int(s.Horizon_min.iloc[-1]),
            "beats_at_all": bool(not pos.empty),
        })
cross = pd.DataFrame(rows)
cross.to_csv(os.path.join(HERE, "phase9_table_crossover.csv"), index=False)

# ------------------------------------------------------------------- figures
fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 4.0), sharey=True)
for ax, ds in zip(np.atleast_1d(axes), DATASETS):
    for m in LEARNED:
        s = skill[(skill.Dataset == ds) & (skill.Model == m)].sort_values("Horizon_min")
        if s.empty:
            continue
        ax.errorbar(s.Horizon_min, s.Skill_mean, yerr=s.Skill_std,
                    marker="o", capsize=3, label=m)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xscale("log")
    # A log axis draws its own minor ticks (2x10^1, 4x10^1 ...), which collide
    # with the explicit horizon labels and make the axis unreadable.
    ax.minorticks_off()
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(h)) for h in horizons])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_title(LABELS[ds], fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel("Skill score vs. persistence")
np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
fig.suptitle("Above the line, learning beats doing nothing", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig17_horizon_skill.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 4.0))
for ax, ds in zip(np.atleast_1d(axes), DATASETS):
    for m in ["Persistence"] + LEARNED:
        s = rmse[(rmse.Dataset == ds) & (rmse.Model == m)].sort_values("Horizon_min")
        if s.empty:
            continue
        style = dict(marker="s", ls="--", color="black") if m == "Persistence" else dict(marker="o")
        ax.plot(s.Horizon_min, s.RMSE_mean, label=m, **style)
    ax.set_xscale("log")
    # A log axis draws its own minor ticks (2x10^1, 4x10^1 ...), which collide
    # with the explicit horizon labels and make the axis unreadable.
    ax.minorticks_off()
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(h)) for h in horizons])
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Mean test RMSE (kW)")
    ax.set_title(LABELS[ds], fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
fig.suptitle("Error growth with horizon: persistence dashed", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig18_horizon_rmse.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- summary
L = ["PHASE 9 - FORECAST HORIZON SWEEP", "=" * 70]
L.append(f"Horizons tested: {', '.join(str(int(h)) + ' min' for h in horizons)}")
L.append(f"Rows: {len(df)}   seeds: {sorted(df[df.Model!='Persistence'].Seed.unique())}")
L.append("")
L.append("-- Skill vs. persistence by horizon (mean across sites and seeds) --")
for ds in DATASETS:
    s = skill[skill.Dataset == ds]
    if s.empty:
        continue
    L.append(f"  {LABELS[ds]}")
    for m in LEARNED:
        r = s[s.Model == m].sort_values("Horizon_min")
        if r.empty:
            continue
        cells = "  ".join(f"{int(h)}min {v:+.3f}" for h, v in
                          zip(r.Horizon_min, r.Skill_mean))
        L.append(f"    {m:<12} {cells}")
L.append("")
L.append("-- Where learning overtakes persistence --")
for r in cross.itertuples():
    if r.beats_at_all:
        L.append(f"  {r.Dataset:<8} {r.Model:<12} first positive at {r.crossover_min} min   "
                 f"({r.shortest_min}min {r.skill_at_shortest:+.3f} -> "
                 f"{r.longest_min}min {r.skill_at_longest:+.3f})")
    else:
        L.append(f"  {r.Dataset:<8} {r.Model:<12} never beats persistence   "
                 f"({r.shortest_min}min {r.skill_at_shortest:+.3f} -> "
                 f"{r.longest_min}min {r.skill_at_longest:+.3f})")
L.append("")
L.append("Figures: fig17_horizon_skill.png, fig18_horizon_rmse.png")

txt = "\n".join(L)
with open(os.path.join(HERE, "phase9_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print(txt)
