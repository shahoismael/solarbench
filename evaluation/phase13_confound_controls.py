#!/usr/bin/env python3
"""
phase13_confound_controls.py

Phase 13. Breaks the collinearity that the hostile audit flagged as CRITICAL.

THE PROBLEM
Across the four datasets, Köppen climate is perfectly confounded with three
other things: sampling interval (Ausgrid 30 min, the rest 15 min), installation
class (research array / campus rooftops / residential meters / utility systems)
and site count (1 / 37 / 300 / 7). The transfer matrix's worst cells are also
the pairs with the largest non-climatic mismatch. A rival explanation therefore
fits the whole matrix without invoking climate at all: models transfer badly
between sampling intervals and installation classes.

With four datasets that confound cannot be removed. It can be tested.

THREE CONTROLS

  A. Resolution control.
     Aggregate DKASC, HKUST and PVDAQ down to Ausgrid's native 30 minutes, so
     every dataset shares one sampling interval, then re-run the transfer
     matrix. Aggregation is legitimate; upsampling Ausgrid to 15 minutes would
     fabricate readings the sensors never took. If the gap survives at matched
     resolution, resolution is not what produces it.

  B. Site-count control.
     Subsample Ausgrid to 37 sites, matching HKUST exactly, and re-run. If
     Ausgrid's poor transfer is an artefact of having 300 sites, it should move.

  C. Smart persistence.
     Same-time-yesterday, alongside the naive last-value baseline. Naive
     persistence is close to unbeatable at one step by construction, which makes
     it a weak reference and a soft target. Smart persistence carries the
     diurnal cycle and is the harder naive comparator, particularly at long
     horizons. It requires no training.

METHOD
Gradient boosting is used as the single learner. It was the strongest one-step
model in Phase 11, trains in seconds rather than minutes, and needs no GPU, so
all three controls run in one pass. This is a control experiment, not a
re-ranking of architectures: what matters is whether the *gap* survives, not
which model is best.

Everything else is held identical to Phase 6: common feature set, per-site
capacity normalization, fixed chronological splits, matched training budget,
three seeds, identical test rows within each condition.

Writes  phase13_confound_results.csv
        phase13_summary.txt
"""

import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    sys.exit("scikit-learn required:  pip install scikit-learn")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "protocol"))

FILES = [
    ("DKASC",   "dataset1_DKASC_15min_labeled.csv",            15),
    ("HKUST",   "dataset2_HKUST_15min_labeled.csv",            15),
    ("Ausgrid", "dataset3_Ausgrid_raw_harmonized_labeled.csv", 30),
    ("PVDAQ",   "dataset4_PVDAQ_15min_labeled.csv",            15),
]
CLIM = [c for c, _, _ in FILES]

N_LAGS = 4
SEEDS = [42, 7, 123]
MAX_TRAIN = 100_000
MAX_TEST_PER_SITE = 2_000
RNG = 0
GBM = dict(max_iter=400, learning_rate=0.06, early_stopping=True,
           validation_fraction=0.15, n_iter_no_change=25)


def load(tag, fname, native, target_res, site_cap=None):
    """Build per-site arrays. target_res=30 aggregates 15-min data down."""
    df = pd.read_csv(os.path.join(PROTO, fname), dtype={"Site_ID": str}, low_memory=False)
    df["Site_ID"] = df["Site_ID"].str.replace("^ID_", "", regex=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if df["IsRareEvent"].dtype != bool:
        df["IsRareEvent"] = df["IsRareEvent"].astype(bool)

    if site_cap is not None:
        keep = sorted(df["Site_ID"].unique())
        rng = np.random.default_rng(RNG)
        if len(keep) > site_cap:
            keep = set(rng.choice(keep, site_cap, replace=False))
            df = df[df["Site_ID"].isin(keep)]

    if target_res > native:
        # Mean-aggregate onto the coarser grid. Split and rare-event labels are
        # taken as the majority/any within each bin so no label is invented.
        df = df.set_index("Timestamp")
        out = []
        for site, g in df.groupby("Site_ID", sort=False):
            r = g["Power_kW"].resample(f"{target_res}min").mean()
            sp = g["Split"].resample(f"{target_res}min").first()
            rare = g["IsRareEvent"].resample(f"{target_res}min").max()
            o = pd.DataFrame({"Power_kW": r, "Split": sp, "IsRareEvent": rare})
            o["Site_ID"] = site
            out.append(o.reset_index())
        df = pd.concat(out, ignore_index=True).dropna(subset=["Split"])
        df["IsRareEvent"] = df["IsRareEvent"].fillna(False).astype(bool)

    step = target_res
    per_day = int(round(24 * 60 / step))
    sites = []
    for site, g in df.groupby("Site_ID", sort=False):
        g = g.sort_values("Timestamp")
        p = g["Power_kW"].to_numpy(dtype=float)
        n = len(p)
        if n <= per_day + N_LAGS + 1:
            continue
        pos = p[p > 0]
        cap = np.percentile(pos, 99.5) if pos.size else np.nan
        if not np.isfinite(cap) or cap <= 0:
            continue
        pn = p / cap
        ts = g["Timestamp"]
        hf = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
        doy = ts.dt.dayofyear.to_numpy()
        tf = np.column_stack([np.sin(2*np.pi*hf/24), np.cos(2*np.pi*hf/24),
                              np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)])
        # start at per_day so same-time-yesterday is always available
        idx = np.arange(per_day, n)
        X = np.column_stack([np.column_stack([pn[idx - k] for k in range(1, N_LAGS+1)]), tf[idx]])
        sites.append(dict(
            tag=tag, site=str(site), cap=cap,
            X=X, y=pn[idx],
            naive=pn[idx-1], smart=pn[idx-per_day],
            split=g["Split"].to_numpy()[idx],
            rare=g["IsRareEvent"].to_numpy()[idx].astype(bool)))
    return sites


def assemble(sites, rng):
    tr_X, tr_y = [], []
    te = dict(X=[], y=[], naive=[], smart=[], cap=[], site=[], rare=[])
    for s in sites:
        ok = np.isfinite(s["X"]).all(axis=1) & np.isfinite(s["y"]) \
             & np.isfinite(s["naive"]) & np.isfinite(s["smart"])
        i_tr = np.flatnonzero((s["split"] == "train") & ok)
        i_te = np.flatnonzero((s["split"] == "test") & ok)
        if i_tr.size:
            tr_X.append(s["X"][i_tr]); tr_y.append(s["y"][i_tr])
        if i_te.size:
            if i_te.size > MAX_TEST_PER_SITE:
                i_te = np.sort(rng.choice(i_te, MAX_TEST_PER_SITE, replace=False))
            te["X"].append(s["X"][i_te]);       te["y"].append(s["y"][i_te])
            te["naive"].append(s["naive"][i_te]); te["smart"].append(s["smart"][i_te])
            te["cap"].append(np.full(i_te.size, s["cap"]))
            te["site"].append(np.full(i_te.size, s["site"], dtype=object))
            te["rare"].append(s["rare"][i_te])
    if not tr_X or not te["X"]:
        return None, None
    return (np.vstack(tr_X), np.concatenate(tr_y)), {k: np.concatenate(v) for k, v in te.items()}


def rmse_by_site(y, yhat, cap, site):
    out = {}
    for s in np.unique(site):
        m = site == s
        if m.sum() < 2:
            continue
        e = (y[m] - yhat[m]) * cap[m]
        out[s] = float(np.sqrt(np.nanmean(e ** 2)))
    return out


def run_condition(name, target_res, ausgrid_sites, rows, t0):
    print(f"\n{'='*66}\nCONDITION: {name}\n{'='*66}", flush=True)
    data = {}
    for tag, fname, native in FILES:
        cap = ausgrid_sites if (tag == "Ausgrid" and ausgrid_sites) else None
        s = load(tag, fname, native, target_res, site_cap=cap)
        data[tag] = s
        print(f"  {tag:<9} sites {len(s):3d}  step {target_res} min", flush=True)

    rng = np.random.default_rng(RNG)
    tr, te = {}, {}
    for tag in CLIM:
        a, b = assemble(data[tag], rng)
        tr[tag], te[tag] = a, b
    del data

    # naive and smart persistence references on identical rows
    ref = {}
    for tag in CLIM:
        t = te[tag]
        ref[tag] = dict(
            naive=rmse_by_site(t["y"], t["naive"], t["cap"], t["site"]),
            smart=rmse_by_site(t["y"], t["smart"], t["cap"], t["site"]))
        for kind in ("naive", "smart"):
            for s, r in ref[tag][kind].items():
                rows.append(dict(Condition=name, Source="-", Target=tag,
                                 Model=f"Persistence_{kind}", Seed=0, Site_ID=s,
                                 RMSE_kW=r, Skill_vs_naive=0.0 if kind == "naive" else
                                 (1 - r/ref[tag]["naive"][s] if ref[tag]["naive"].get(s) else np.nan)))

    for src in CLIM:
        if tr[src] is None:
            continue
        Xtr, ytr = tr[src]
        for seed in SEEDS:
            r0 = np.random.default_rng(seed)
            idx = np.arange(len(ytr))
            if idx.size > MAX_TRAIN:
                idx = r0.choice(idx, MAX_TRAIN, replace=False)
            g = HistGradientBoostingRegressor(random_state=seed, **GBM)
            g.fit(Xtr[idx], ytr[idx])
            line = []
            for tgt in CLIM:
                t = te[tgt]
                yp = g.predict(t["X"])
                per = rmse_by_site(t["y"], yp, t["cap"], t["site"])
                sk = []
                for s, r in per.items():
                    rn = ref[tgt]["naive"].get(s)
                    rs = ref[tgt]["smart"].get(s)
                    ss = 1 - r/rn if rn else np.nan
                    sm = 1 - r/rs if rs else np.nan
                    rows.append(dict(Condition=name, Source=src, Target=tgt,
                                     Model="GBM", Seed=seed, Site_ID=s,
                                     RMSE_kW=r, Skill_vs_naive=ss, Skill_vs_smart=sm))
                    sk.append(ss)
                line.append(f"{tgt[:4]} {np.nanmean(sk):+.3f}")
            print(f"    {src:<9} seed {seed:<4} " + "  ".join(line), flush=True)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)


rows = []
t0 = time.time()

run_condition("baseline_15min",       15, None, rows, t0)
run_condition("matched_30min",        30, None, rows, t0)
run_condition("ausgrid_37sites",      15, 37,   rows, t0)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "phase13_confound_results.csv"), index=False)

# ------------------------------------------------------------------- summary
L = ["PHASE 13 - CONFOUND CONTROLS", "=" * 70]
g = res[res.Model == "GBM"].copy()
g["cond"] = np.where(g.Source == g.Target, "same", "transfer")

L.append("\n-- A/B. Does the gap survive matched resolution and matched site count? --")
L.append(f"{'condition':<20} {'same-climate':>13} {'transferred':>13} {'gap':>8}")
for c in res.Condition.unique():
    d = g[g.Condition == c]
    a = d[d.cond == "same"].groupby("Seed").Skill_vs_naive.mean().mean()
    b = d[d.cond == "transfer"].groupby("Seed").Skill_vs_naive.mean().mean()
    L.append(f"{c:<20} {a:>13.3f} {b:>13.3f} {a-b:>8.3f}")

L.append("\n-- Per-target transferred skill, by condition --")
for c in res.Condition.unique():
    d = g[(g.Condition == c) & (g.cond == "transfer")]
    parts = [f"{t} {d[d.Target==t].Skill_vs_naive.mean():+.3f}" for t in CLIM]
    L.append(f"  {c:<20} " + "  ".join(parts))

L.append("\n-- C. Smart persistence (same time yesterday) vs naive --")
p = res[res.Model.str.startswith("Persistence")]
for c in res.Condition.unique():
    d = p[p.Condition == c]
    parts = []
    for t in CLIM:
        n_ = d[(d.Target == t) & (d.Model == "Persistence_naive")].RMSE_kW.mean()
        s_ = d[(d.Target == t) & (d.Model == "Persistence_smart")].RMSE_kW.mean()
        parts.append(f"{t} {100*(s_-n_)/n_:+.0f}%")
    L.append(f"  {c:<20} " + "  ".join(parts))
L.append("  (positive = smart persistence is worse than naive at this horizon)")

L.append("\n-- GBM skill against smart persistence, same-climate --")
for c in res.Condition.unique():
    d = g[(g.Condition == c) & (g.cond == "same")]
    L.append(f"  {c:<20} {d.Skill_vs_smart.mean():+.3f}")

txt = "\n".join(L)
open(os.path.join(HERE, "phase13_summary.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt)
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Rows: {len(res)}")
