#!/usr/bin/env python3
"""
phase6c_aggregate_transfer.py

Phase 6, stage C. Turns the per-site output of phase6b_run_transfer_matrix.m
into the paper-ready transfer tables and figures.

Reads (from this directory):
    phase6_transfer_matrix_results.csv
    phase6_persistence_reference.csv

Writes (to this directory):
    phase6_table_transfer_rmse.csv        source x target, mean RMSE (kW), per model
    phase6_table_transfer_nrmse.csv       source x target, mean nRMSE, per model
    phase6_table_transfer_skill.csv       source x target, mean skill score, per model
    phase6_table_transfer_loss.csv        off-diagonal penalty vs. own-climate training
    phase6_table_loco.csv                 leave-one-climate-out summary
    phase6_table_rare_in_transfer.csv     rare-event degradation inside every cell
    phase6_summary.txt                    headline numbers, ready to quote

    fig14_transfer_heatmap.png            skill score, source x target, one panel per model
    fig15_transfer_loss.png               transfer penalty by model
    fig16_loco_skill.png                  LOCO skill score by held-out climate

Run:  python phase6c_aggregate_transfer.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Prefer the tuned run when phase7 has been executed; fall back to the
# untuned run otherwise. SUFFIX also tags every output so the two runs
# never overwrite each other.
SUFFIX = "_tuned" if os.path.isfile(
    os.path.join(HERE, "phase6_transfer_matrix_results_tuned.csv")) else ""
RES = os.path.join(HERE, f"phase6_transfer_matrix_results{SUFFIX}.csv")
PER = os.path.join(HERE, f"phase6_persistence_reference{SUFFIX}.csv")
print(f"Reading {'TUNED' if SUFFIX else 'UNTUNED'} results\n")

CLIMATES = ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]
MODELS = ["MLP", "LSTM", "Transformer"]
LABELS = {
    "DKASC": "DKASC\n(desert)",
    "HKUST": "HKUST\n(subtropical)",
    "Ausgrid": "Ausgrid\n(temperate)",
    "PVDAQ": "PVDAQ\n(mixed)",
}

for p in (RES, PER):
    if not os.path.isfile(p):
        sys.exit(f"Missing input: {p}\nRun phase6a then phase6b in MATLAB first.")

df = pd.read_csv(RES)
persist = pd.read_csv(PER)

# Split single-source rows from leave-one-climate-out rows
single = df[~df["Source"].str.startswith("LOCO_")].copy()
loco = df[df["Source"].str.startswith("LOCO_")].copy()
loco["HeldOut"] = loco["Source"].str.replace("LOCO_", "", regex=False)


def cell_mean(frame, value, index, columns):
    """Site-level values -> mean per seed, then mean across seeds.

    Averaging per seed first stops a 300-site dataset's seed noise from being
    weighted differently than a 1-site dataset's.
    """
    per_seed = frame.groupby([index, columns, "Seed"], as_index=False)[value].mean()
    return per_seed.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")


# ---------------------------------------------------------------- transfer tables
tables = {}
for metric, fname in [
    ("RMSE_kW", "phase6_table_transfer_rmse.csv"),
    ("nRMSE", "phase6_table_transfer_nrmse.csv"),
    ("SkillScore", "phase6_table_transfer_skill.csv"),
]:
    out = []
    for m in MODELS:
        sub = single[single["Model"] == m]
        if sub.empty:
            continue
        piv = cell_mean(sub, metric, "Source", "Target")
        piv = piv.reindex(index=CLIMATES, columns=CLIMATES)
        piv.insert(0, "Model", m)
        piv.index.name = "TrainedOn"
        out.append(piv.reset_index())
    tbl = pd.concat(out, ignore_index=True)
    tbl.to_csv(os.path.join(HERE, fname.replace(".csv", SUFFIX + ".csv")), index=False)
    tables[metric] = tbl

# ------------------------------------------------------------------ transfer loss
# For each target climate: how much worse is a model trained elsewhere than the
# same model trained on that climate? This is the number the paper's central
# claim rests on.
rows = []
for m in MODELS:
    sub = single[single["Model"] == m]
    if sub.empty:
        continue
    piv = cell_mean(sub, "nRMSE", "Source", "Target").reindex(index=CLIMATES, columns=CLIMATES)
    for tgt in CLIMATES:
        if tgt not in piv.columns:
            continue
        own = piv.loc[tgt, tgt]
        others = piv.loc[[c for c in CLIMATES if c != tgt], tgt]
        rows.append({
            "Model": m,
            "Target": tgt,
            "OwnClimate_nRMSE": own,
            "Transferred_mean_nRMSE": others.mean(),
            "Transferred_worst_nRMSE": others.max(),
            "Transfer_penalty_pct": 100.0 * (others.mean() - own) / own if own else np.nan,
            "Worst_penalty_pct": 100.0 * (others.max() - own) / own if own else np.nan,
        })
loss = pd.DataFrame(rows)
loss.to_csv(os.path.join(HERE, f"phase6_table_transfer_loss{SUFFIX}.csv"), index=False)

# --------------------------------------------------------------------------- LOCO
loco_rows = []
for m in MODELS:
    sub = loco[loco["Model"] == m]
    if sub.empty:
        continue
    per_seed = sub.groupby(["HeldOut", "Seed"], as_index=False).agg(
        nRMSE=("nRMSE", "mean"), RMSE_kW=("RMSE_kW", "mean"), Skill=("SkillScore", "mean")
    )
    agg = per_seed.groupby("HeldOut", as_index=False).agg(
        nRMSE_mean=("nRMSE", "mean"), nRMSE_std=("nRMSE", "std"),
        RMSE_kW_mean=("RMSE_kW", "mean"),
        Skill_mean=("Skill", "mean"), Skill_std=("Skill", "std"),
    )
    agg.insert(0, "Model", m)
    loco_rows.append(agg)

loco_tbl = pd.concat(loco_rows, ignore_index=True) if loco_rows else pd.DataFrame()
if not loco_tbl.empty:
    # Compare LOCO against the same model trained on the held-out climate itself
    ref = {}
    for m in MODELS:
        sub = single[single["Model"] == m]
        if sub.empty:
            continue
        piv = cell_mean(sub, "nRMSE", "Source", "Target").reindex(index=CLIMATES, columns=CLIMATES)
        for c in CLIMATES:
            ref[(m, c)] = piv.loc[c, c]
    loco_tbl["OwnClimate_nRMSE"] = [ref.get((r.Model, r.HeldOut), np.nan) for r in loco_tbl.itertuples()]
    loco_tbl["LOCO_penalty_pct"] = 100.0 * (loco_tbl["nRMSE_mean"] - loco_tbl["OwnClimate_nRMSE"]) / loco_tbl["OwnClimate_nRMSE"]
loco_tbl.to_csv(os.path.join(HERE, f"phase6_table_loco{SUFFIX}.csv"), index=False)

# ------------------------------------------------------ rare events inside cells
rare = df.dropna(subset=["RareRMSE_kW"]).copy()
rare["RarePct"] = 100.0 * (rare["RareRMSE_kW"] - rare["RMSE_kW"]) / rare["RMSE_kW"]
rare_tbl = (rare.groupby(["Source", "Target", "Model"], as_index=False)
                .agg(Overall_RMSE_kW=("RMSE_kW", "mean"),
                     Rare_RMSE_kW=("RareRMSE_kW", "mean"),
                     Rare_pct_worse=("RarePct", "mean"),
                     N_sites=("Site_ID", "nunique")))
rare_tbl.to_csv(os.path.join(HERE, f"phase6_table_rare_in_transfer{SUFFIX}.csv"), index=False)

# ------------------------------------------------------------------------ figures
def heat(ax, mat, title, cmap, vmin, vmax, fmt="{:.2f}"):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CLIMATES)))
    ax.set_yticks(range(len(CLIMATES)))
    ax.set_xticklabels([LABELS[c] for c in CLIMATES], fontsize=7)
    ax.set_yticklabels([LABELS[c] for c in CLIMATES], fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            lum = (v - vmin) / (vmax - vmin + 1e-12)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if lum > 0.6 or lum < 0.15 else "black")
    for k in range(len(CLIMATES)):
        ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False, lw=2, ec="black"))
    return im


# Fig 14 - skill score heatmap
fig, axes = plt.subplots(1, len(MODELS), figsize=(4.6 * len(MODELS), 4.6))
if len(MODELS) == 1:
    axes = [axes]
for k, (ax, m) in enumerate(zip(axes, MODELS)):
    sub = single[single["Model"] == m]
    piv = cell_mean(sub, "SkillScore", "Source", "Target").reindex(index=CLIMATES, columns=CLIMATES)
    im = heat(ax, piv.values, m, "RdYlGn", -1.0, 1.0)
    # Only the leftmost panel carries the y-axis label; on the inner panels
    # matplotlib places it between the subplots, where it overlaps the
    # neighbouring heatmap cells.
    if k == 0:
        ax.set_ylabel("Trained on", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Tested on", fontsize=9)
fig.subplots_adjust(wspace=0.25)
fig.suptitle("Forecast skill score vs. naive persistence  (>0 beats persistence; boxed cells are same-climate)",
             fontsize=11, fontweight="bold")
fig.colorbar(im, ax=axes, shrink=0.8, label="Skill score")
fig.savefig(os.path.join(HERE, f"fig14_transfer_heatmap{SUFFIX}.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Fig 15 - transfer penalty
if not loss.empty:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(MODELS)
    x = np.arange(len(CLIMATES))
    for i, m in enumerate(MODELS):
        sub = loss[loss["Model"] == m].set_index("Target").reindex(CLIMATES)
        ax.bar(x + i * width, sub["Transfer_penalty_pct"], width, label=m)
    ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
    ax.set_xticklabels([LABELS[c] for c in CLIMATES], fontsize=8)
    ax.set_ylabel("nRMSE increase vs. training on that climate (%)")
    ax.set_title("Cost of training in the wrong climate", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(HERE, f"fig15_transfer_loss{SUFFIX}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# Fig 16 - LOCO skill
if not loco_tbl.empty:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(MODELS)
    x = np.arange(len(CLIMATES))
    for i, m in enumerate(MODELS):
        sub = loco_tbl[loco_tbl["Model"] == m].set_index("HeldOut").reindex(CLIMATES)
        ax.bar(x + i * width, sub["Skill_mean"], width, yerr=sub["Skill_std"],
               capsize=3, label=m)
    ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
    ax.set_xticklabels([LABELS[c] for c in CLIMATES], fontsize=8)
    ax.set_ylabel("Skill score vs. persistence")
    ax.set_title("Leave-one-climate-out: trained on three climates, tested on the fourth",
                 fontweight="bold")
    ax.axhline(0, color="black", lw=1.2)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(HERE, f"fig16_loco_skill{SUFFIX}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------------ summary
lines = []
lines.append("PHASE 6 - CROSS-CLIMATE TRANSFER SUMMARY")
lines.append("=" * 70)
lines.append(f"Result rows: {len(df)}   models: {sorted(df['Model'].unique())}   "
             f"seeds: {sorted(df['Seed'].unique())}")
lines.append("")

skill = tables["SkillScore"]
lines.append("-- Skill score vs. persistence (mean across seeds and sites) --")
for m in MODELS:
    sub = skill[skill["Model"] == m].set_index("TrainedOn")
    diag = [sub.loc[c, c] for c in CLIMATES if c in sub.index and c in sub.columns]
    off = [sub.loc[a, b] for a in CLIMATES for b in CLIMATES
           if a != b and a in sub.index and b in sub.columns]
    diag = [v for v in diag if pd.notna(v)]
    off = [v for v in off if pd.notna(v)]
    if not diag or not off:
        continue
    lines.append(f"  {m:<12} same-climate {np.mean(diag):+.3f} | transferred {np.mean(off):+.3f}")
    lines.append(f"               cells beating persistence: same-climate "
                 f"{sum(v > 0 for v in diag)}/{len(diag)}, transferred "
                 f"{sum(v > 0 for v in off)}/{len(off)}")
lines.append("")

if not loss.empty:
    lines.append("-- Transfer penalty (nRMSE increase vs. training on the target climate) --")
    for m in MODELS:
        sub = loss[loss["Model"] == m]
        if sub.empty:
            continue
        lines.append(f"  {m:<12} mean {sub['Transfer_penalty_pct'].mean():+.1f}%   "
                     f"worst {sub['Worst_penalty_pct'].max():+.1f}% "
                     f"({sub.loc[sub['Worst_penalty_pct'].idxmax(), 'Target']})")
    lines.append("")

if not loco_tbl.empty:
    lines.append("-- Leave-one-climate-out --")
    for m in MODELS:
        sub = loco_tbl[loco_tbl["Model"] == m]
        if sub.empty:
            continue
        beat = (sub["Skill_mean"] > 0).sum()
        lines.append(f"  {m:<12} mean skill {sub['Skill_mean'].mean():+.3f}   "
                     f"beats persistence on {beat}/{len(sub)} held-out climates   "
                     f"mean penalty {sub['LOCO_penalty_pct'].mean():+.1f}%")
    lines.append("")

lines.append("-- Rare events inside transfer cells --")
pos = (rare_tbl["Rare_pct_worse"] > 0).sum()
lines.append(f"  cells where rare-event RMSE is worse than overall: {pos}/{len(rare_tbl)}")
lines.append(f"  range: {rare_tbl['Rare_pct_worse'].min():+.1f}% to "
             f"{rare_tbl['Rare_pct_worse'].max():+.1f}%")
lines.append("")
lines.append("Figures: fig14_transfer_heatmap.png, fig15_transfer_loss.png, fig16_loco_skill.png")

text = "\n".join(lines)
with open(os.path.join(HERE, f"phase6_summary{SUFFIX}.txt"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
print(text)
