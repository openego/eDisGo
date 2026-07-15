# `spatial_complexity_reduction(aggregation_mode=True)` doesn't aggregate `electromobility.flexibility_bands`, breaking `optimize` on merged charging points

**Type:** bug
**Found:** 2026-07-15, running the real `uc6_spatial_reduction.yaml` pipeline
on grid 32377 with `spatial_reduction.aggregation_mode: true`.
**Affects:** `spatial_complexity_reduction`/`apply_busmap`
(`edisgo/tools/spatial_complexity_reduction.py`), specifically the load
aggregation step; consumed by `_build_electromobility`
(`edisgo/io/powermodels_io.py`).

## Problem

When `aggregation_mode=True`, `apply_busmap` merges loads at the same bus
(`aggregate_loads_df`, `spatial_complexity_reduction.py:1587-1599`) and
correctly aggregates their **time series** via `aggregate_timeseries`
(`spatial_complexity_reduction.py:1718`, called for `loads_active_power` /
`loads_reactive_power`). This works fine for DSM loads and heat pumps, whose
OPF constraints are read directly from `loads_df`/`dsm.p_max`/
`heat_pump.heat_demand_df` keyed by whatever name is currently in `loads_df`.

Charging points are different: the OPF's own constraint builder,
`_build_electromobility` (`edisgo/io/powermodels_io.py:1212`), does **not**
read its upper-power bound from `loads_df` — it reads
`electromobility.flexibility_bands["upper_power"][cp_name]`, a separate
DataFrame keyed by charging-point name. `spatial_complexity_reduction` never
touches `flexibility_bands` during aggregation, so it still only has entries
for the *original*, pre-merge charging-point names.

When a bus has 2+ charging points and gets merged into one representative
load (e.g. `Load_Bus_mvgd_32377_F2_B2_charging_point_hpc`), that
representative name has:
- a correctly-aggregated `loads_active_power` entry (summed from the
  originals), but
- **no** entry at all in `flexibility_bands["upper_power"]`.

`task_optimize` derives `flexible_cps` from `loads_df` (filtering
`type == "charging_point"`), so it passes the representative's name into
`pm_optimize` → `to_powermodels` → `_build_electromobility`, which then does
`flex_bands_df["upper_power"][emob_df.index[cp_i]]` and raises `KeyError`
for the representative name.

## Reproduction

Run `uc6_spatial_reduction.yaml` (or any preset with the spatial bracket) on
a grid with 2+ charging points sharing a bus, with
`spatial_reduction: {enabled: true, aggregation_mode: true,
load_aggregation_mode: bus}` and `optimize: {flexible: [...,
charging_points, ...]}`. Crashes inside `optimize`, before `spatial_restore`
ever runs:

```
KeyError: 'Load_Bus_mvgd_32377_F2_B2_charging_point_hpc'
  .../edisgo/io/powermodels_io.py:1212, in _build_electromobility
    * flex_bands_df["upper_power"][emob_df.index[cp_i]].iloc[0]
```

## Scope note

This is independent of the spatial-reduction pipeline-integration work
(`spatial_reduce`/`spatial_restore`/`apply_reduced_results_to_full_grid`) —
it's a gap in `spatial_complexity_reduction` itself (the reduction side),
already reachable via `EDisGo.spatial_complexity_reduction` +
`EDisGo.pm_optimize` directly, with no pipeline involved. It only manifests
under `aggregation_mode=True` with charging points present; `aggregation_
mode=False` is unaffected since every component keeps its original name
there, and `flexibility_bands` stays valid.

Also worth checking as part of the same fix: whether heat pumps have an
analogous issue if the OPF ever reads a heat-pump-specific band keyed
differently from `loads_df` (current understanding, from the spatial-
reduction design session, is that heat pumps use `heat_demand_df`/`cop_df`/
`loads_df.p_set` directly, which DO get aggregated correctly — but worth
double-checking with a merged multi-heat-pump bus once this issue is picked
up, in case some other heat-pump-specific structure has the same gap).

## Suggested fix

Extend `apply_busmap`'s load-aggregation step to also aggregate
`electromobility.flexibility_bands` (`upper_power`, `lower_energy`,
`upper_energy`) onto the same representative name, using the same
`old_name`-based grouping/summing already used for `aggregate_timeseries`
— summing `upper_power` per merged group is the natural analogue of summing
`p_set`/active power for the representative.
