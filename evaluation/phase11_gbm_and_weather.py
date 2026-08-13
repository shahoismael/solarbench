#!/usr/bin/env python3
"""
phase11_gbm_and_weather.py

Phase 11. Closes the last two gaps in one run.

GAP 1 - the missing gradient-boosting baseline.
The original design called for XGBoost. It was replaced by an MLP because
MATLAB's Statistics and Machine Learning Toolbox was not licensed. A reviewer
will read that as a convenience substitution. This script runs a real
histogram-based gradient boosting regressor (scikit-learn's
HistGradientBoostingRegressor, the same algorithm family as LightGBM and
XGBoost's hist mode) so the baseline set is defensible.

GAP 2 - the two-protocol problem.
Phases 3 and 4 trained with weather features. Phases 6, 9 and 10 use a common
power-plus-time feature set, because Ausgrid has no weather and cross-climate
transfer is impossible with mismatched input widths. That leaves two different
protocols in one paper. Rather than discard either, this script runs both
feature sets side by side on identical splits, which converts the
inconsistency into a stated ablation: what does weather actually buy?

PROTOCOL
  - In-climate only, one step ahead, the same chronological splits as Phase 2.
  - Per-site capacity normalization identical to every other phase.
  - Persistence is recomputed here on exactly these rows as the skill reference.
  - Three seeds. GBM is seeded; persistence is deterministic.

Requires: pandas, numpy, scikit-learn
    pip install scikit-learn

Reads   ..\\protocol\\dataset*_labeled.csv
Writes  phase11_gbm_weather_results.csv
        phase11_summary.txt
"""

import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    sys.exit("scikit-learn is required:  pip install scikit-learn")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "protocol"))

FILES = [
    ("DKASC",   "dataset1_DKASC_15min_labeled.csv"),
    ("HKUST",   "dataset2_HKUST_15min_labeled.csv"),
    ("Ausgrid", "dataset3_Ausgrid_raw_harmonized_labeled.csv"),
    ("PVDAQ",   "dataset4_PVDAQ_15min_labeled.csv"),
]
WEATHER = ["Irradiance_Wm2", "Temp_C", "Humidity_pct", "Wind_ms"]
N_LAGS = 4
SEEDS = [42, 7, 123]
MAX_TRAIN = 100_000   # matched to the neural baselines in Phase 6; an unequal
                      # budget would make the GBM comparison unfair and the
                      # methodology audit flagged it as such
MAX_TEST_PER_SITE = 2_000
RNG = 0


def build(df, wcols_global):
    """Per-site lag/time features, capacity-normalised target, persistence.

    The weather block always uses wcols_global, the columns that carry data
    somewhere in this dataset. A site missing one of them gets a NaN column
    rather than a narrower matrix; PVDAQ has sites with and without weather,
    and per-site column lists made the stacked matrices incompatible.
    HistGradientBoostingRegressor handles NaN natively, so no imputation is
    invented here.
    """
    out = []
    for site, g in df.groupby("Site_ID", sort=False):
        g = g.sort_values("Timestamp")
        p = g["Power_kW"].to_numpy(dtype=float)
        n = len(p)
        if n <= N_LAGS + 1:
            continue
        pos = p[p > 0]
        cap = np.percentile(pos, 99.5) if pos.size else np.nan
        if not np.isfinite(cap) or cap <= 0:
            cap = np.nanmax(p) if np.isfinite(np.nanmax(p)) else np.nan
        if not np.isfinite(cap) or cap <= 0:
            continue
        pn = p / cap

        ts = g["Timestamp"]
        hf = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
        doy = ts.dt.dayofyear.to_numpy()
        tfeat = np.column_stack([np.sin(2 * np.pi * hf / 24), np.cos(2 * np.pi * hf / 24),
                                 np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])

        idx = np.arange(N_LAGS, n)
        lags = np.column_stack([pn[idx - k] for k in range(1, N_LAGS + 1)])
        base = np.column_stack([lags, tfeat[idx]])

        if wcols_global:
            W = np.column_stack([
                g[c].to_numpy(dtype=float)[idx] if c in g.columns
                else np.full(len(idx), np.nan)
                for c in wcols_global
            ])
        else:
            W = np.empty((len(idx), 0))
        wcols = wcols_global

        d = dict(
            y=pn[idx], persist=pn[idx - 1], cap=np.full(len(idx), cap),
            site=np.full(len(idx), str(site)), split=g["Split"].to_numpy()[idx],
            rare=g["IsRareEvent"].to_numpy()[idx].astype(bool),
            yraw=p[idx], base=base, weather=W, wcols=wcols,
        )
        out.append(d)
    return out


def metrics(y, yhat, cap):
    yk = y * cap
    pk = yhat * cap
    e = yk - pk
    rmse = float(np.sqrt(np.nanmean(e ** 2)))
    mae = float(np.nanmean(np.abs(e)))
    nrmse = float(np.sqrt(np.nanmean((y - yhat) ** 2)))
    m = yk > 0.01
    mape = float(100 * np.nanmean(np.abs(e[m] / yk[m]))) if m.sum() else np.nan
    return rmse, mae, nrmse, mape


rows = []
t0 = time.time()

for tag, fname in FILES:
    path = os.path.join(PROTO, fname)
    if not os.path.isfile(path):
        print(f"missing {path} -- skipped")
        continue
    print(f"\n######## {tag} ########", flush=True)

    df = pd.read_csv(path, dtype={"Site_ID": str}, low_memory=False)
    df["Site_ID"] = df["Site_ID"].str.replace("^ID_", "", regex=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if df["IsRareEvent"].dtype != bool:
        df["IsRareEvent"] = df["IsRareEvent"].astype(bool)

    # Decided once for the whole dataset, not per site.
    wcols_global = [c for c in WEATHER if c in df.columns and df[c].notna().any()]
    sites = build(df, wcols_global)
    del df
    if not sites:
        print("  no usable sites")
        continue
    has_weather = len(wcols_global) > 0
    print(f"  sites: {len(sites)} | weather columns: "
          f"{wcols_global if has_weather else 'none'}", flush=True)

    rng = np.random.default_rng(RNG)

    # assemble train / test once
    Xtr_b, Xtr_w, ytr = [], [], []
    Xte_b, Xte_w, yte, pte, cte, ste, rte = [], [], [], [], [], [], []
    for s in sites:
        istr = s["split"] == "train"
        iste = s["split"] == "test"
        valid = np.isfinite(s["base"]).all(axis=1) & np.isfinite(s["y"]) & np.isfinite(s["persist"])
        istr &= valid
        iste &= valid
        if istr.any():
            k = np.flatnonzero(istr)
            Xtr_b.append(s["base"][k])
            Xtr_w.append(np.hstack([s["base"][k], s["weather"][k]]) if has_weather
                         else s["base"][k])
            ytr.append(s["y"][k])
        if iste.any():
            k = np.flatnonzero(iste)
            if k.size > MAX_TEST_PER_SITE:
                k = np.sort(rng.choice(k, MAX_TEST_PER_SITE, replace=False))
            Xte_b.append(s["base"][k])
            Xte_w.append(np.hstack([s["base"][k], s["weather"][k]]) if has_weather
                         else s["base"][k])
            yte.append(s["y"][k]); pte.append(s["persist"][k])
            cte.append(s["cap"][k]); ste.append(s["site"][k]); rte.append(s["rare"][k])
    del sites

    Xtr_b = np.vstack(Xtr_b); Xtr_w = np.vstack(Xtr_w); ytr = np.concatenate(ytr)
    Xte_b = np.vstack(Xte_b); Xte_w = np.vstack(Xte_w); yte = np.concatenate(yte)
    pte = np.concatenate(pte); cte = np.concatenate(cte)
    ste = np.concatenate(ste); rte = np.concatenate(rte)
    print(f"  train {len(ytr)} | test {len(yte)}", flush=True)

    # ---- persistence reference, per site ----
    pref = {}
    for site in np.unique(ste):
        m = ste == site
        if m.sum() < 2:
            continue
        r, a, nr, mp = metrics(yte[m], pte[m], cte[m])
        rr = np.nan
        if rte[m].sum() >= 2:
            rr, _, _, _ = metrics(yte[m][rte[m]], pte[m][rte[m]], cte[m][rte[m]])
        pref[site] = r
        rows.append(dict(Dataset=tag, Features="n/a", Model="Persistence", Seed=0,
                         Site_ID=site, N_test=int(m.sum()), N_rare=int(rte[m].sum()),
                         RMSE_kW=r, MAE_kW=a, nRMSE=nr, MAPE_pct=mp,
                         RareRMSE_kW=rr, SkillScore=0.0))

    # ---- GBM on each feature set ----
    for fname_set, Xtr, Xte in (("common", Xtr_b, Xte_b), ("common+weather", Xtr_w, Xte_w)):
        if fname_set == "common+weather" and not has_weather:
            continue
        for seed in SEEDS:
            r0 = np.random.default_rng(seed)
            idx = np.arange(len(ytr))
            if idx.size > MAX_TRAIN:
                idx = r0.choice(idx, MAX_TRAIN, replace=False)
            t1 = time.time()
            gbm = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, max_depth=None,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=seed)
            gbm.fit(Xtr[idx], ytr[idx])
            yp = gbm.predict(Xte)

            accR, accS = [], []
            for site in np.unique(ste):
                m = ste == site
                if m.sum() < 2:
                    continue
                r, a, nr, mp = metrics(yte[m], yp[m], cte[m])
                rr = np.nan
                if rte[m].sum() >= 2:
                    rr, _, _, _ = metrics(yte[m][rte[m]], yp[m][rte[m]], cte[m][rte[m]])
                ss = 1 - r / pref[site] if pref.get(site, 0) else np.nan
                rows.append(dict(Dataset=tag, Features=fname_set, Model="GBM", Seed=seed,
                                 Site_ID=site, N_test=int(m.sum()), N_rare=int(rte[m].sum()),
                                 RMSE_kW=r, MAE_kW=a, nRMSE=nr, MAPE_pct=mp,
                                 RareRMSE_kW=rr, SkillScore=ss))
                accR.append(r); accS.append(ss)
            print(f"    GBM {fname_set:<15} seed {seed:<4} {time.time()-t1:6.1f}s "
                  f"RMSE {np.nanmean(accR):.4f}  skill {np.nanmean(accS):+.3f}", flush=True)

    del Xtr_b, Xtr_w, ytr, Xte_b, Xte_w, yte, pte, cte, ste, rte

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "phase11_gbm_weather_results.csv"), index=False)

# ------------------------------------------------------------------- summary
L = ["PHASE 11 - GRADIENT BOOSTING BASELINE AND WEATHER ABLATION", "=" * 70]
L.append(f"Rows: {len(res)}   seeds: {SEEDS}")
L.append("")

g = res[res.Model == "GBM"]
L.append("-- GBM skill vs. persistence, by feature set --")
for ds in ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]:
    d = g[g.Dataset == ds]
    if d.empty:
        continue
    parts = []
    for fs in ["common", "common+weather"]:
        s = d[d.Features == fs]
        if s.empty:
            continue
        per_seed = s.groupby("Seed")["SkillScore"].mean()
        parts.append(f"{fs} {per_seed.mean():+.3f}")
    L.append(f"  {ds:<8} " + "   ".join(parts))
L.append("")

L.append("-- What does weather buy? (RMSE change, common -> common+weather) --")
for ds in ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]:
    d = g[g.Dataset == ds]
    a = d[d.Features == "common"].groupby("Seed")["RMSE_kW"].mean().mean()
    b = d[d.Features == "common+weather"].groupby("Seed")["RMSE_kW"].mean().mean()
    if np.isnan(a) or np.isnan(b):
        L.append(f"  {ds:<8} no weather available")
        continue
    L.append(f"  {ds:<8} {a:.4f} -> {b:.4f} kW   ({100*(b-a)/a:+.1f}%)")
L.append("")

L.append("-- Rare-event degradation (GBM, common features) --")
for ds in ["DKASC", "HKUST", "Ausgrid", "PVDAQ"]:
    d = g[(g.Dataset == ds) & (g.Features == "common")].dropna(subset=["RareRMSE_kW"])
    if d.empty:
        continue
    pct = 100 * (d.RareRMSE_kW.mean() - d.RMSE_kW.mean()) / d.RMSE_kW.mean()
    L.append(f"  {ds:<8} {pct:+.1f}%")

txt = "\n".join(L)
with open(os.path.join(HERE, "phase11_summary.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print("\n" + txt)
print(f"\nDone in {(time.time()-t0)/60:.1f} min")
