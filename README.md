# SolarBench

A harmonized, cross-climate benchmark for photovoltaic power forecasting — four Köppen zones, one protocol, open baselines.

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
