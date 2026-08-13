#!/usr/bin/env python3
"""
phase16_missing_figures.py

Generates the three figures the manuscript still lacks.

  Figure 7  — Confound controls (Section 4.6).
              The section that answers the audit's CRITICAL finding currently
              has no figure at all. This is the most important of the three.

  Figure 8  — Weather ablation (Section 4.5).

  Figure 9  — Study design schematic (Section 3.7).
              Drawn from the protocol constants, not from data, so the numbers
              on it are hard-coded from Methodology rather than read from a CSV.

Every plotted value is read from the released result files. Nothing is typed in
by hand except the schematic's structural labels.

Run from the results directory:
    python phase16_missing_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_new")
os.makedirs(OUT, exist_ok=True)

# Colourblind-safe (Okabe-Ito)
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GREY, PURPLE = "#666666", "#CC79A7"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def need(fname):
    p = os.path.join(HERE, fname)
    if not os.path.isfile(p):
        sys.exit(f"Missing input: {p}")
    return p


# =====================================================================
# FIGURE 7 — Confound controls
# =====================================================================
d13 = pd.read_csv(need("phase13_confound_results.csv"))
d14 = pd.read_csv(need("phase14_within_results.csv"))

g = d13[d13.Model == "GBM"].copy()
g["cond"] = np.where(g.Source == g.Target, "same", "transfer")

rows = []
label = {"baseline_15min": "Baseline\n(nothing matched)",
         "matched_30min": "Resolution\nmatched (30 min)",
         "ausgrid_37sites": "Site count\nmatched (37)"}
for c in ["baseline_15min", "matched_30min", "ausgrid_37sites"]:
    x = g[g.Condition == c]
    same = x[x.cond == "same"].groupby("Seed").Skill_vs_naive.mean().mean()
    tr = x[x.cond == "transfer"].groupby("Seed").Skill_vs_naive.mean().mean()
    rows.append((label[c], same, tr, same - tr))

w = d14[d14.Control.str.startswith("within")]
same = w[w.Control == "within_same_half"].groupby("Seed").Skill.mean().mean()
tr = w[w.Control == "within_cross_half"].groupby("Seed").Skill.mean().mean()
rows.append(("Within dataset\n(only site identity)", same, tr, same - tr))

cap = d14[d14.Control.str.startswith("capacity")]
same = cap[cap.Control == "capacity_same"].groupby("Seed").Skill.mean().mean()
tr = cap[cap.Control == "capacity_cross"].groupby("Seed").Skill.mean().mean()
rows.append(("Capacity class\n(PVDAQ, n=4+4)", same, tr, same - tr))

names = [r[0] for r in rows]
same_v = [r[1] for r in rows]
tr_v = [r[2] for r in rows]
gap_v = [r[3] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                               gridspec_kw={"width_ratios": [1.3, 1]})

x = np.arange(len(names))
bw = 0.38
ax1.bar(x - bw/2, same_v, bw, label="Same climate", color=BLUE)
ax1.bar(x + bw/2, tr_v, bw, label="Transferred", color=ORANGE)
ax1.axhline(0, color="black", lw=1.1)
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=7.0)
ax1.set_ylim(min(tr_v) * 1.45, max(same_v) * 1.45)
ax1.set_ylabel("Skill score vs. naive persistence")
ax1.set_title("Skill by condition", fontweight="bold")
ax1.legend(frameon=False, loc="lower left")
ax1.grid(axis="y", alpha=0.25)
for xi, (s, t) in enumerate(zip(same_v, tr_v)):
    ax1.text(xi - bw/2, s + 0.004, f"{s:+.3f}", ha="center", va="bottom", fontsize=7)
    ax1.text(xi + bw/2, t + (0.004 if t >= 0 else -0.012), f"{t:+.3f}",
             ha="center", va="bottom" if t >= 0 else "top", fontsize=7)

colours = [GREY, RED, RED, GREEN, PURPLE]
bars = ax2.barh(np.arange(len(names)), gap_v, color=colours)
ax2.axvline(gap_v[0], color=GREY, ls="--", lw=1.2)
ax2.text(gap_v[0] + 0.004, -0.62, "baseline gap 0.109",
         fontsize=7.5, color=GREY, va="center")
ax2.set_yticks(np.arange(len(names)))
ax2.set_yticklabels(names, fontsize=7.5)
ax2.invert_yaxis()
ax2.set_xlabel("Generalization gap (same − transferred)")
ax2.set_title("Does the gap survive the control?", fontweight="bold")
ax2.grid(axis="x", alpha=0.25)
for i, v in enumerate(gap_v):
    ax2.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5)
ax2.set_xlim(0, max(gap_v) * 1.28)
ax2.set_ylim(len(names) - 0.4, -1.1)

fig.suptitle("Controls on the cross-dataset gap: a bar at or beyond the dashed line means the control did not explain it",
             fontsize=9.5, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_confound_controls.png"))
plt.close(fig)
print("fig7_confound_controls.png")
for n, s, t, gp in rows:
    print(f"   {n[:28]:30s} same {s:+.3f}  transferred {t:+.3f}  gap {gp:.3f}")


# =====================================================================
# FIGURE 8 — Weather ablation
# =====================================================================
d11 = pd.read_csv(need("phase11_gbm_weather_results.csv"))
gb = d11[d11.Model == "GBM"]
CLIM = ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]
LAB = {"DKASC": "DKASC\n(desert)", "HKUST": "HKUST\n(subtropical)",
       "Ausgrid": "Ausgrid\n(temperate)", "PVDAQ": "PVDAQ\n(mixed)"}

rm = (gb.groupby(["Dataset", "Features", "Seed"], as_index=False).RMSE_kW.mean()
        .groupby(["Dataset", "Features"], as_index=False).RMSE_kW.mean())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

x = np.arange(len(CLIM))
bw = 0.36
com, wea = [], []
for c in CLIM:
    s = rm[(rm.Dataset == c) & (rm.Features == "common")].RMSE_kW
    w_ = rm[(rm.Dataset == c) & (rm.Features == "common+weather")].RMSE_kW
    com.append(float(s.iloc[0]) if len(s) else np.nan)
    wea.append(float(w_.iloc[0]) if len(w_) else np.nan)

a1.bar(x - bw/2, com, bw, label="Common features", color=BLUE)
a1.bar(x + bw/2, wea, bw, label="Common + weather", color=GREEN)
a1.set_xticks(x); a1.set_xticklabels([LAB[c] for c in CLIM], fontsize=7.5)
a1.set_ylabel("Mean test RMSE (kW)")
a1.set_yscale("log")
a1.set_title("Error with and without weather (log scale)", fontweight="bold")
a1.legend(frameon=False)
a1.grid(axis="y", alpha=0.25)
a1.text(2, min(v for v in com if np.isfinite(v)) * 1.1, "no weather\navailable",
        ha="center", fontsize=7, style="italic", color=GREY)

pct = [100*(w_-c)/c if (np.isfinite(w_) and np.isfinite(c)) else np.nan
       for c, w_ in zip(com, wea)]
cols = [GREEN if (np.isfinite(p) and p < 0) else RED for p in pct]
a2.bar(x, [p if np.isfinite(p) else 0 for p in pct], 0.55, color=cols)
a2.axhline(0, color="black", lw=1.1)
a2.set_xticks(x); a2.set_xticklabels([LAB[c] for c in CLIM], fontsize=7.5)
a2.set_ylabel("RMSE change from adding weather (%)")
a2.set_title("What weather buys", fontweight="bold", pad=12)
a2.grid(axis="y", alpha=0.25)

finite = [p for p in pct if np.isfinite(p)]
lo, hi = min(finite), max(finite)
a2.set_ylim(lo - 4.5, hi + 3.5)          # headroom so no label meets the frame

for xi, p in enumerate(pct):
    if not np.isfinite(p):
        a2.text(xi, 0.35, "no weather\navailable", ha="center", va="bottom",
                fontsize=7, style="italic", color=GREY)
        continue
    # Long negative bars get their label inside; short bars outside.
    inside = p < lo * 0.55
    a2.text(xi, p * 0.5 if inside else p - 0.9,
            f"{p:+.1f}%", ha="center",
            va="center" if inside else "top",
            fontsize=8.5, fontweight="bold",
            color="white" if inside else "black")

a2.text(0.5, -0.30, "green = weather helps      red = weather hurts",
        transform=a2.transAxes, ha="center", fontsize=7.5, color=GREY)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_weather_ablation.png"))
plt.close(fig)
print("\nfig8_weather_ablation.png")
for c, a, b, p in zip(CLIM, com, wea, pct):
    print(f"   {c:9s} common {a:.4f}  +weather {b if np.isfinite(b) else float('nan'):.4f}  {p:+.1f}%"
          if np.isfinite(b) else f"   {c:9s} common {a:.4f}  no weather")


# =====================================================================
# FIGURE 9 — Study design schematic
# =====================================================================
fig, ax = plt.subplots(figsize=(10.2, 5.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")


def box(x, y, w, h, text, fc, ec=None, fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=fc, ec=ec or "#333333", lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color="#333333"))


# Row 1 — sources
ax.text(50, 55.6, "Four independent public datasets — five Köppen zones, 345 sites, 23.3 M records",
        ha="center", fontsize=8.6, fontweight="bold")
src = [("DKASC\nBWh · 1 site\n15 min", 2), ("HKUST\nCwa · 37 sites\n15 min", 26),
       ("Ausgrid\nCfa · 300 sites\n30 min", 50), ("PVDAQ\nBSk/BWh/Cfa/Dfb\n7 sites · 15 min", 74)]
for t, xp in src:
    box(xp, 46.5, 22, 7, t, "#D9E8F5")
    arrow(xp + 11, 46.5, xp + 11, 43.5)

# Row 2 — harmonization
box(2, 37, 94, 6.5,
    "Harmonization  ·  common schema  ·  15-min resampling where native resolution is finer  ·  per-site capacity normalization  $c_i$ = P99.5",
    "#FDEBD0")
arrow(50, 37, 50, 34)

# Row 3 — protocol
box(2, 27.5, 94, 6.5,
    "Protocol  ·  chronological 70/15/15 split, no shuffling  ·  common feature set: normalized power + 4 cyclical time features\n"
    "rare-event labels: clipping proxy and cloud-transient proxy",
    "#FDEBD0")
arrow(50, 27.5, 50, 24.5)

# Row 4 — models
box(2, 19, 94, 5.5,
    "Five baselines under matched budgets (100,000 windows, 3 seeds)  ·  naive persistence · MLP · LSTM · Transformer · gradient boosting\n"
    "hyperparameters: 12 configurations each, selected once on HKUST validation only, then frozen",
    "#E8F5E9")
for xp in (14, 38, 62, 86):
    arrow(xp, 19, xp, 16)

# Row 5 — experiments
exp = [("Transfer matrix\n4 sources × 4 targets\n§4.2", 2, "#E3F2E1"),
       ("Leave-one-\nclimate-out\n§4.2", 26, "#E3F2E1"),
       ("Horizon sweep\n15/30/60/180 min\n§4.3", 50, "#E3F2E1"),
       ("Rare-event sweep\n405 thresholds\n§4.4", 74, "#E3F2E1")]
for t, xp, c in exp:
    box(xp, 9, 22, 6.8, t, c)
    arrow(xp + 11, 9, xp + 11, 6.2)

# Row 6 — controls
box(2, 0.5, 94, 5.5,
    "Confound controls (§4.6)  ·  matched resolution · matched site count · smart-persistence baseline · within-dataset floor · capacity class\n"
    "Evaluation metric throughout: forecast skill score against naive persistence, computed per site on identical test rows",
    "#F3E5F5")

fig.savefig(os.path.join(OUT, "fig9_study_design.png"))
plt.close(fig)
print("\nfig9_study_design.png")

print(f"\nAll three written to {OUT}")
print("Copy into final_for_submission/figures/ once checked.")
