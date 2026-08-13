#!/usr/bin/env python3
"""
phase14_within_dataset_controls.py

Phase 14. Two further controls on the confound the hostile audit raised.

Phase 13 excluded sampling interval and site count as explanations for the
cross-dataset gap. Two possibilities remained: installation class, and the
trivial possibility that transferring to *any* unseen site costs something,
regardless of what changes.

CONTROL D — within-dataset transfer floor.
Split each dataset's own sites into two disjoint halves. Train on half A, test
on half B, and the reverse. Climate, sampling interval, installation class,
instrumentation, metering convention and split strategy are all identical; the
only thing that differs is which specific sites the model saw.

This measures the floor: what transfer costs when nothing changes but identity.
If the within-dataset floor is near zero while cross-dataset transfer is
strongly negative, the cross-dataset gap is not a generic cost of unseen sites.
If the floor is itself strongly negative, the paper's central claim is about
site identity rather than climate, and would have to be rewritten.

DKASC is excluded: one site cannot be split.

CONTROL E — installation class within one archive.
PVDAQ's systems span 6 kW to 408 kW under one provider, one instrumentation
standard and one metering convention. Splitting at the median capacity gives a
small-system group and a large-system group, and transferring between them
varies installation scale while holding data provenance fixed. Climate is not
fully held constant, because PVDAQ's systems sit in four Köppen zones, so this
is weaker than Control D and is reported as indicative rather than conclusive.
With roughly three or four scored sites per side it is underpowered by
construction. It is reported anyway: a reviewer who asks whether installation
class was tested should receive a measurement and its limitations, not a
sentence in Limitations.

Learner, features, normalization, splits, budget and seeds are identical to
Phase 13.

Writes  phase14_within_results.csv
        phase14_summary.txt
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
    ("DKASC",   "dataset1_DKASC_15min_labeled.csv"),
    ("HKUST",   "dataset2_HKUST_15min_labeled.csv"),
    ("Ausgrid", "dataset3_Ausgrid_raw_harmonized_labeled.csv"),
    ("PVDAQ",   "dataset4_PVDAQ_15min_labeled.csv"),
]
N_LAGS = 4
SEEDS = [42, 7, 123]
MAX_TRAIN = 100_000
MAX_TEST_PER_SITE = 2_000
RNG = 0
GBM = dict(max_iter=400, learning_rate=0.06, early_stopping=True,
           validation_fraction=0.15, n_iter_no_change=25)


def build_sites(tag, fname):
    df = pd.read_csv(os.path.join(PROTO, fname), dtype={"Site_ID": str}, low_memory=False)
    df["Site_ID"] = df["Site_ID"].str.replace("^ID_", "", regex=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if df["IsRareEvent"].dtype != bool:
        df["IsRareEvent"] = df["IsRareEvent"].astype(bool)

    sites = []
    for site, g in df.groupby("Site_ID", sort=False):
        g = g.sort_values("Timestamp")
        p = g["Power_kW"].to_numpy(dtype=float)
        n = len(p)
        if n <= N_LAGS + 2:
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
        idx = np.arange(N_LAGS, n)
        X = np.column_stack([np.column_stack([pn[idx-k] for k in range(1, N_LAGS+1)]), tf[idx]])
        sites.append(dict(tag=tag, site=str(site), cap=float(cap), X=X, y=pn[idx],
                          naive=pn[idx-1], split=g["Split"].to_numpy()[idx]))
    return sites


def pack(subset, rng, want):
    """want='train' or 'test'."""
    X, y, naive, cap, sid = [], [], [], [], []
    for s in subset:
        ok = np.isfinite(s["X"]).all(axis=1) & np.isfinite(s["y"]) & np.isfinite(s["naive"])
        i = np.flatnonzero((s["split"] == want) & ok)
        if not i.size:
            continue
        if want == "test" and i.size > MAX_TEST_PER_SITE:
            i = np.sort(rng.choice(i, MAX_TEST_PER_SITE, replace=False))
        X.append(s["X"][i]); y.append(s["y"][i]); naive.append(s["naive"][i])
        cap.append(np.full(i.size, s["cap"])); sid.append(np.full(i.size, s["site"], dtype=object))
    if not X:
        return None
    return dict(X=np.vstack(X), y=np.concatenate(y), naive=np.concatenate(naive),
                cap=np.concatenate(cap), site=np.concatenate(sid))


def rmse_by_site(y, yhat, cap, site):
    out = {}
    for s in np.unique(site):
        m = site == s
        if m.sum() < 2:
            continue
        e = (y[m] - yhat[m]) * cap[m]
        out[s] = float(np.sqrt(np.nanmean(e ** 2)))
    return out


def transfer(train_sites, test_sites, label_src, label_tgt, control, rows, rng):
    tr = pack(train_sites, rng, "train")
    te = pack(test_sites, rng, "test")
    if tr is None or te is None:
        return None
    ref = rmse_by_site(te["y"], te["naive"], te["cap"], te["site"])
    skills = []
    for seed in SEEDS:
        r0 = np.random.default_rng(seed)
        idx = np.arange(len(tr["y"]))
        if idx.size > MAX_TRAIN:
            idx = r0.choice(idx, MAX_TRAIN, replace=False)
        g = HistGradientBoostingRegressor(random_state=seed, **GBM)
        g.fit(tr["X"][idx], tr["y"][idx])
        per = rmse_by_site(te["y"], g.predict(te["X"]), te["cap"], te["site"])
        ss = []
        for s, r in per.items():
            v = 1 - r / ref[s] if ref.get(s) else np.nan
            rows.append(dict(Control=control, Source=label_src, Target=label_tgt,
                             Seed=seed, Site_ID=s, RMSE_kW=r, Skill=v))
            ss.append(v)
        skills.append(np.nanmean(ss))
    return float(np.nanmean(skills))


rows = []
t0 = time.time()
rng = np.random.default_rng(RNG)

print("=" * 66)
print("CONTROL D — within-dataset transfer floor")
print("=" * 66, flush=True)

store = {}
for tag, fname in FILES:
    s = build_sites(tag, fname)
    store[tag] = s
    print(f"  {tag:<9} usable sites {len(s)}", flush=True)

for tag in ["HKUST", "Ausgrid", "PVDAQ"]:
    s = store[tag]
    names = sorted({x["site"] for x in s})
    if len(names) < 4:
        print(f"  {tag}: too few sites to split")
        continue
    r = np.random.default_rng(RNG)
    perm = r.permutation(len(names))
    half = len(names) // 2
    A = {names[i] for i in perm[:half]}
    B = {names[i] for i in perm[half:]}
    sA = [x for x in s if x["site"] in A]
    sB = [x for x in s if x["site"] in B]

    within = transfer(sA, sA, f"{tag}-A", f"{tag}-A", "within_same_half", rows, rng)
    ab = transfer(sA, sB, f"{tag}-A", f"{tag}-B", "within_cross_half", rows, rng)
    ba = transfer(sB, sA, f"{tag}-B", f"{tag}-A", "within_cross_half", rows, rng)
    print(f"  {tag:<9} |A|={len(A)} |B|={len(B)} | same-half {within:+.3f} | "
          f"A→B {ab:+.3f} | B→A {ba:+.3f}", flush=True)

print()
print("=" * 66)
print("CONTROL E — installation class within PVDAQ (capacity split)")
print("=" * 66, flush=True)

s = store["PVDAQ"]
caps = sorted(((x["site"], x["cap"]) for x in s), key=lambda z: z[1])
med = np.median([c for _, c in caps])
small = {n for n, c in caps if c <= med}
large = {n for n, c in caps if c > med}
print("  capacities (kW): " + ", ".join(f"{n}:{c:.0f}" for n, c in caps))
print(f"  median {med:.0f} kW | small {len(small)} sites | large {len(large)} sites", flush=True)

sS = [x for x in s if x["site"] in small]
sL = [x for x in s if x["site"] in large]
if len(small) >= 2 and len(large) >= 2:
    ss = transfer(sS, sS, "PVDAQ-small", "PVDAQ-small", "capacity_same", rows, rng)
    ll = transfer(sL, sL, "PVDAQ-large", "PVDAQ-large", "capacity_same", rows, rng)
    sl = transfer(sS, sL, "PVDAQ-small", "PVDAQ-large", "capacity_cross", rows, rng)
    ls = transfer(sL, sS, "PVDAQ-large", "PVDAQ-small", "capacity_cross", rows, rng)
    print(f"  small→small {ss:+.3f} | large→large {ll:+.3f} | "
          f"small→large {sl:+.3f} | large→small {ls:+.3f}", flush=True)
else:
    print("  too few sites per side")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "phase14_within_results.csv"), index=False)

# ------------------------------------------------------------------- summary
L = ["PHASE 14 - WITHIN-DATASET AND INSTALLATION-CLASS CONTROLS", "=" * 70]

L.append("\n-- Control D: what does transfer cost when only site identity changes? --")
L.append(f"{'dataset':<10} {'same half':>11} {'cross half':>12} {'floor':>8}")
for tag in ["HKUST", "Ausgrid", "PVDAQ"]:
    d = res[res.Source.str.startswith(tag)]
    if d.empty:
        continue
    a = d[d.Control == "within_same_half"].groupby("Seed").Skill.mean().mean()
    b = d[d.Control == "within_cross_half"].groupby("Seed").Skill.mean().mean()
    L.append(f"{tag:<10} {a:>11.3f} {b:>12.3f} {a-b:>8.3f}")

d = res[res.Control.str.startswith("within")]
a = d[d.Control == "within_same_half"].groupby("Seed").Skill.mean().mean()
b = d[d.Control == "within_cross_half"].groupby("Seed").Skill.mean().mean()
L.append(f"{'POOLED':<10} {a:>11.3f} {b:>12.3f} {a-b:>8.3f}")
L.append("\nCompare against the cross-dataset gap of 0.109 measured in Phase 13")
L.append("under identical learner, features, budget and seeds.")

L.append("\n-- Control E: installation class within PVDAQ --")
d = res[res.Control.str.startswith("capacity")]
if not d.empty:
    a = d[d.Control == "capacity_same"].groupby("Seed").Skill.mean().mean()
    b = d[d.Control == "capacity_cross"].groupby("Seed").Skill.mean().mean()
    L.append(f"  same capacity class  {a:+.3f}")
    L.append(f"  cross capacity class {b:+.3f}")
    L.append(f"  gap                  {a-b:+.3f}")
    L.append(f"  sites per side: {len(small)} small, {len(large)} large — underpowered, indicative only")

txt = "\n".join(L)
open(os.path.join(HERE, "phase14_summary.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt)
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Rows: {len(res)}")
