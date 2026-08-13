#!/usr/bin/env python3
"""
phase17_export_release_protocol.py

Exports the small, redistributable half of the SolarBench protocol so that the
GitHub repository can reproduce the evaluation without shipping any source data.

The four *_labeled.csv files under dataset/protocol/ are the harmonized corpus
(1.6 GB, 23.3 M rows). They cannot go in a git repository and they are not ours
to redistribute. What CAN be released is the derived protocol: which site went
in which split, where each split's date boundary falls, the per-site capacity
normalization constant, and the rare-event label counts. Those are a few hundred
rows in total, and they are what a third party needs in order to rebuild the
identical evaluation from their own copy of the source data.

Writes to solarbench/protocol/:
    splits_<dataset>.csv       one row per site x split: rows, date range
    capacity_<dataset>.csv     one row per site: c_i = P99.5 of positive power
    rare_events_<dataset>.csv  one row per site x split: label counts and rate
    protocol_summary.csv       one row per dataset, for the README table

Run:  python phase17_export_release_protocol.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROTOCOL_IN = HERE.parent / "protocol"
OUT = HERE.parent.parent.parent / "solarbench" / "protocol"

DATASETS = {
    "DKASC":   "dataset1_DKASC_15min_labeled.csv",
    "HKUST":   "dataset2_HKUST_15min_labeled.csv",
    "Ausgrid": "dataset3_Ausgrid_raw_harmonized_labeled.csv",
    "PVDAQ":   "dataset4_PVDAQ_15min_labeled.csv",
}

KOPPEN = {
    "DKASC":   "BWh",
    "HKUST":   "Cwa",
    "Ausgrid": "Cfa",
    "PVDAQ":   "BSk, BWh, Cfa, Dfb",
}
NATIVE_RES_MIN = {"DKASC": 15, "HKUST": 15, "Ausgrid": 30, "PVDAQ": 15}

USECOLS = ["Timestamp", "Power_kW", "Site_ID", "Split", "IsRareEvent"]
CHUNK = 2_000_000

# The capacity constant is the 99.5th percentile of strictly positive power.
# Streaming an exact percentile over 15 M rows is not worth the memory, so
# positive values are accumulated per site and the percentile taken at the end.
# Ausgrid is the only file where that matters and it stays under ~1 GB of
# float32 because most rows are zero (night) and are discarded.
CAPACITY_PCTILE = 99.5


def process(name: str, path: Path):
    split_rows = {}
    rare_rows = {}
    pos_power = {}
    n_total = 0

    reader = pd.read_csv(
        path, usecols=USECOLS, chunksize=CHUNK,
        parse_dates=["Timestamp"], dtype={"Site_ID": "str", "Split": "str"},
    )
    for chunk in reader:
        n_total += len(chunk)

        g = chunk.groupby(["Site_ID", "Split"], sort=False)
        for key, sub in g:
            ts = sub["Timestamp"]
            if key in split_rows:
                prev = split_rows[key]
                split_rows[key] = (
                    prev[0] + len(sub),
                    min(prev[1], ts.min()),
                    max(prev[2], ts.max()),
                )
                rare_rows[key] = (
                    rare_rows[key][0] + int(sub["IsRareEvent"].sum()),
                    rare_rows[key][1] + len(sub),
                )
            else:
                split_rows[key] = (len(sub), ts.min(), ts.max())
                rare_rows[key] = (int(sub["IsRareEvent"].sum()), len(sub))

        pos = chunk.loc[chunk["Power_kW"] > 0, ["Site_ID", "Power_kW"]]
        for site, sub in pos.groupby("Site_ID", sort=False):
            pos_power.setdefault(site, []).append(
                sub["Power_kW"].to_numpy(dtype=np.float32))

    splits = pd.DataFrame(
        [(s, sp, n, a, b) for (s, sp), (n, a, b) in split_rows.items()],
        columns=["site_id", "split", "n_rows", "start", "end"],
    ).sort_values(["site_id", "split"])

    rare = pd.DataFrame(
        [(s, sp, r, n) for (s, sp), (r, n) in rare_rows.items()],
        columns=["site_id", "split", "n_rare", "n_rows"],
    ).sort_values(["site_id", "split"])
    rare["rare_rate_pct"] = (100 * rare["n_rare"] / rare["n_rows"]).round(4)

    cap = pd.DataFrame(
        [(s, float(np.percentile(np.concatenate(v), CAPACITY_PCTILE)), len(np.concatenate(v)))
         for s, v in pos_power.items()],
        columns=["site_id", "capacity_kw", "n_positive_rows"],
    ).sort_values("site_id")
    cap["capacity_kw"] = cap["capacity_kw"].round(6)

    assert (cap["capacity_kw"] > 0).all(), f"{name}: non-positive capacity constant"

    OUT.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUT / f"splits_{name}.csv", index=False)
    cap.to_csv(OUT / f"capacity_{name}.csv", index=False)
    rare.to_csv(OUT / f"rare_events_{name}.csv", index=False)

    summary = {
        "dataset": name,
        "koppen": KOPPEN[name],
        "native_resolution_min": NATIVE_RES_MIN[name],
        "sites": splits["site_id"].nunique(),
        "rows": n_total,
        "train_rows": int(splits.loc[splits["split"] == "train", "n_rows"].sum()),
        "val_rows": int(splits.loc[splits["split"] == "val", "n_rows"].sum()),
        "test_rows": int(splits.loc[splits["split"] == "test", "n_rows"].sum()),
        "start": splits["start"].min(),
        "end": splits["end"].max(),
        "rare_rate_pct": round(100 * rare["n_rare"].sum() / rare["n_rows"].sum(), 4),
        "capacity_min_kw": round(cap["capacity_kw"].min(), 4),
        "capacity_max_kw": round(cap["capacity_kw"].max(), 4),
    }
    print(f"  {name:8s} sites={summary['sites']:4d} rows={n_total:,} "
          f"rare={summary['rare_rate_pct']}% "
          f"cap=[{summary['capacity_min_kw']}, {summary['capacity_max_kw']}] kW")
    return summary


def main() -> None:
    rows = []
    for name, fn in DATASETS.items():
        path = PROTOCOL_IN / fn
        if not path.exists():
            sys.exit(f"missing: {path}")
        print(f"scanning {fn} ...")
        rows.append(process(name, path))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "protocol_summary.csv", index=False)
    print(f"\ntotal rows: {summary['rows'].sum():,}")
    print(f"total sites: {summary['sites'].sum()}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
