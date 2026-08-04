# SolarBench

A harmonized, cross-climate benchmark for photovoltaic power forecasting — four Köppen zones, one protocol, open baselines.

## Status

Research in progress. Code, harmonized data, protocol files, and baseline results will be released here on publication, alongside a Zenodo DOI and a citation entry.

## Data

Raw datasets are not redistributed. Each must be obtained from its original public source; the harmonization scripts document the expected folder and file structure.

## License

Code is released under Apache-2.0. Derived data and protocol files will be released under CC-BY-4.0, subject to each source dataset's own terms.
## Overview

SolarBench harmonizes four public PV datasets spanning four Köppen climate zones into a single schema and evaluation protocol, so forecasting models can be compared across climates rather than on a single convenience site.

| Dataset | Location | Köppen zone | Sites |
|---|---|---|---|
| DKASC | Alice Springs, Australia | BWh — hot desert | 1 |
| HKUST Rooftop | Hong Kong | Cwa — humid subtropical | 60 |
| Ausgrid | Sydney, Australia | Cfa — temperate | 300 |
| PVDAQ | Continental United States | BSk, BWh, Cfa, Dfb | 8 |

## What the benchmark provides

- A common schema across all four datasets: timestamp, power (kW), irradiance, temperature, humidity, wind speed, site ID, Köppen label
- Fixed chronological train/validation/test splits — no random shuffling
- A labeled rare-event subset covering inverter clipping and sudden cloud transients
- Reference baselines: persistence, MLP, LSTM, and a self-attention Transformer
- Cross-climate transfer evaluation: train on one climate, test on all four

## Repository structure
