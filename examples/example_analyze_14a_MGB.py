"""
§14a EnWG Heat Pump Curtailment Analysis - Monthly Simulation

This script performs a monthly optimization with §14a heat pump curtailment
and generates comprehensive analysis plots and statistics.

Usage:
    python analyze_14a_full_year.py --grid_path <path_to_ding0_grid> --scenario eGon2035 --num_days 30
"""

import os
import sys
import argparse
import pandas as pd
import geopandas as gpd
import pypsa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import matplotlib.colors as mcolors
import contextily as ctx
import imageio.v2 as imageio
import re

from edisgo import EDisGo
from edisgo.io.db import engine as egon_engine


def add_charging_points_manually(edisgo, num_cps=30, seed=42):
    """
    Add charging points manually to the grid with realistic size distribution.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object
    num_cps : int
        Number of charging points to add (default: 30)
    seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    EDisGo
        EDisGo object with added charging points
    """
    print(f"\n3a. Adding {num_cps} charging points manually...")

    np.random.seed(seed + 100)  # Different seed than HPs

    # Get random LV buses (different from HP buses if possible)
    lv_buses = edisgo.topology.buses_df[
        edisgo.topology.buses_df.v_nom < 1.0  # LV buses
    ]

    if len(lv_buses) < num_cps:
        print(
            f"  ⚠ Warning: Only {len(lv_buses)} LV buses available, using all"
        )
        num_cps = len(lv_buses)

    selected_buses = lv_buses.sample(n=num_cps, random_state=seed + 100)

    # Realistic distribution based on typical EV charging points:
    # - 50% home charging (3.7-11 kW, typically 11 kW)
    # - 30% work/public charging (11-22 kW)
    # - 20% fast charging (22-50 kW, but curtailed to grid limits)
    num_home = int(num_cps * 0.5)
    num_work = int(num_cps * 0.3)
    num_fast = num_cps - num_home - num_work

    cp_data = []
    cp_names = []

    # Home charging points (3.7-11 kW)
    for i in range(num_home):
        p_set = np.random.uniform(0.0037, 0.011)  # 3.7-11 kW
        cp_data.append(
            {
                "bus": selected_buses.index[i],
                "p_set": p_set,
                "type": "charging_point",
                "sector": "home",
            }
        )
        cp_names.append(f"CP_home_{i+1}")

    # Work/public charging points (11-22 kW)
    for i in range(num_work):
        p_set = np.random.uniform(0.011, 0.022)  # 11-22 kW
        cp_data.append(
            {
                "bus": selected_buses.index[num_home + i],
                "p_set": p_set,
                "type": "charging_point",
                "sector": "work",
            }
        )
        cp_names.append(f"CP_work_{i+1}")

    # Fast charging points (22-50 kW)
    for i in range(num_fast):
        p_set = np.random.uniform(0.022, 0.050)  # 22-50 kW
        cp_data.append(
            {
                "bus": selected_buses.index[num_home + num_work + i],
                "p_set": p_set,
                "type": "charging_point",
                "sector": "public",
            }
        )
        cp_names.append(f"CP_fast_{i+1}")

    # Add to topology
    cp_df = pd.DataFrame(cp_data, index=cp_names)
    edisgo.topology.loads_df = pd.concat([edisgo.topology.loads_df, cp_df])

    print(f"  ✓ Added {len(cp_names)} charging points:")
    print(f"    - {num_home} home (3.7-11 kW)")
    print(f"    - {num_work} work/public (11-22 kW)")
    print(f"    - {num_fast} fast charging (22-50 kW)")
    print(
        f"    - §14a eligible (>4.2 kW): {len(cp_df[cp_df['p_set'] > 0.0042])}"
    )

    return edisgo


def add_heat_pumps_manually(edisgo, num_hps=50, seed=42):
    """
    Add heat pumps manually to the grid with realistic size distribution.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object
    num_hps : int
        Number of heat pumps to add (default: 50)
    seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    EDisGo
        EDisGo object with added heat pumps
    """
    print(f"\n3. Adding {num_hps} heat pumps manually...")

    np.random.seed(seed)

    # Get random LV buses
    lv_buses = edisgo.topology.buses_df[
        edisgo.topology.buses_df.v_nom < 1.0  # LV buses
    ]

    if len(lv_buses) < num_hps:
        print(
            f"  ⚠ Warning: Only {len(lv_buses)} LV buses available, using all"
        )
        num_hps = len(lv_buses)

    selected_buses = lv_buses.sample(n=num_hps, random_state=seed)

    # Realistic distribution based on German residential heat pumps:
    # - 60% large (11-20 kW) - typical for older/larger houses
    # - 30% medium (5-11 kW) - typical for modern houses
    # - 10% small (3-5 kW) - typical for well-insulated new buildings
    num_large = int(num_hps * 0.6)
    num_medium = int(num_hps * 0.3)
    num_small = num_hps - num_large - num_medium

    hp_data = []
    hp_names = []

    # Large heat pumps (11-20 kW)
    for i in range(num_large):
        p_set = np.random.uniform(0.011, 0.020)  # 11-20 kW
        hp_data.append(
            {
                "bus": selected_buses.index[i],
                "p_set": p_set,
                "type": "heat_pump",
                "sector": "residential",
            }
        )
        hp_names.append(f"HP_large_{i+1}")

    # Medium heat pumps (5-11 kW)
    for i in range(num_medium):
        p_set = np.random.uniform(0.005, 0.011)  # 5-11 kW
        hp_data.append(
            {
                "bus": selected_buses.index[num_large + i],
                "p_set": p_set,
                "type": "heat_pump",
                "sector": "residential",
            }
        )
        hp_names.append(f"HP_medium_{i+1}")

    # Small heat pumps (3-5 kW)
    for i in range(num_small):
        p_set = np.random.uniform(0.003, 0.005)  # 3-5 kW
        hp_data.append(
            {
                "bus": selected_buses.index[num_large + num_medium + i],
                "p_set": p_set,
                "type": "heat_pump",
                "sector": "residential",
            }
        )
        hp_names.append(f"HP_small_{i+1}")

    # Add to topology
    hp_df = pd.DataFrame(hp_data, index=hp_names)
    edisgo.topology.loads_df = pd.concat([edisgo.topology.loads_df, hp_df])

    print(f"  ✓ Added {len(hp_names)} heat pumps:")
    print(f"    - {num_large} large (11-20 kW)")
    print(f"    - {num_medium} medium (5-11 kW)")
    print(f"    - {num_small} small (3-5 kW)")
    print(f"    - §14a eligible (>4.2 kW): {num_large + num_medium}")

    return edisgo


def create_hp_timeseries(edisgo, scenario="eGon2035", num_days=30):
    """
    Create realistic heat demand and COP time series for heat pumps.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with heat pumps
    scenario : str
        Scenario name for time index
    num_days : int
        Number of days to simulate (default: 30 for one month)

    Returns
    -------
    EDisGo
        EDisGo object with HP time series
    """
    print(f"\n4. Creating heat pump time series for {num_days} days...")

    # Get heat pumps
    heat_pumps = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "heat_pump"
    ]

    if len(heat_pumps) == 0:
        print("  ⚠ Warning: No heat pumps found")
        return edisgo
    timeindex = edisgo.timeseries.timeindex
    print(f"  Creating time series for {len(heat_pumps)} heat pumps...")
    print(f"  Timeindex: {len(timeindex)} timesteps (hourly, {num_days} days)")

    # Create realistic heat demand profile for winter month
    hour_of_day = timeindex.hour.values
    day_of_week = timeindex.dayofweek.values  # 0=Monday, 6=Sunday

    # Winter season - high base load (mid-January)
    seasonal_factor = 0.9  # High demand in winter

    # Daily pattern (higher demand morning and evening)
    daily_factor = 0.7 + 0.3 * (
        np.exp(-((hour_of_day - 7) ** 2) / 8)  # Morning peak
        + np.exp(-((hour_of_day - 19) ** 2) / 8)  # Evening peak
    )

    # Weekend pattern (slightly different - later morning, more evening)
    weekend_mask = day_of_week >= 5
    daily_factor[weekend_mask] = 0.6 + 0.4 * (
        np.exp(-((hour_of_day[weekend_mask] - 9) ** 2) / 10)  # Later morning
        + np.exp(-((hour_of_day[weekend_mask] - 20) ** 2) / 10)  # Evening peak
    )

    # Combine patterns
    base_profile = seasonal_factor * daily_factor

    # Create COP profile (winter - lower COP due to cold temperatures)
    # Typical air-source heat pump COP in winter: 2.5-3.5
    cop_profile = 3.0 + np.random.normal(0, 0.2, len(timeindex))
    cop_profile = np.clip(cop_profile, 2.5, 3.5)

    # Create individual profiles for each HP
    heat_demand_data = {}
    cop_data = {}

    for hp_name in heat_pumps.index:
        p_set = heat_pumps.loc[hp_name, "p_set"]

        # Heat demand: base profile scaled by nominal power with random variation
        # Assume average COP of 3.5, so thermal = electrical * 3.5
        base_thermal = p_set * 3.5

        # Add individual variation (±20%)
        individual_factor = 0.8 + 0.4 * np.random.random(len(timeindex))
        heat_demand = base_profile * base_thermal * individual_factor

        # Add some random noise
        heat_demand += np.random.normal(0, 0.001, len(timeindex))
        heat_demand = np.maximum(heat_demand, 0)  # No negative demand

        heat_demand_data[hp_name] = heat_demand

        # Individual COP with small variation
        individual_cop = cop_profile + np.random.normal(0, 0.1, len(timeindex))
        individual_cop = np.clip(individual_cop, 2.5, 4.5)
        cop_data[hp_name] = individual_cop

    # Set data
    edisgo.heat_pump.heat_demand_df = pd.DataFrame(
        heat_demand_data, index=timeindex
    )
    edisgo.heat_pump.cop_df = pd.DataFrame(cop_data, index=timeindex)

    print(f"  ✓ Created time series:")
    print(
        f"    Heat demand range: {edisgo.heat_pump.heat_demand_df.min().min():.6f} - {edisgo.heat_pump.heat_demand_df.max().max():.6f} MW"
    )
    print(
        f"    COP range: {edisgo.heat_pump.cop_df.min().min():.2f} - {edisgo.heat_pump.cop_df.max().max():.2f}"
    )

    return edisgo


def create_cp_timeseries(edisgo, scenario="eGon2035", num_days=30):
    """
    Create realistic charging demand time series for charging points.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with charging points
    scenario : str
        Scenario name for time index
    num_days : int
        Number of days to simulate (default: 30 for one month)

    Returns
    -------
    EDisGo
        EDisGo object with CP time series
    """
    print(f"\n4a. Creating charging point time series for {num_days} days...")

    # Get charging points
    charging_points = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "charging_point"
    ]

    if len(charging_points) == 0:
        print("  ⚠ Warning: No charging points found")
        return edisgo

    # Use same time index as heat pumps
    timeindex = edisgo.timeseries.timeindex

    print(
        f"  Creating time series for {len(charging_points)} charging points..."
    )
    print(f"  Timeindex: {len(timeindex)} timesteps (hourly, {num_days} days)")

    # Create realistic charging profiles based on use case
    hours = np.arange(len(timeindex))
    hour_of_day = timeindex.hour.values
    day_of_week = timeindex.dayofweek.values  # 0=Monday, 6=Sunday

    cp_load_data = {}

    for cp_name in charging_points.index:
        p_set = charging_points.loc[cp_name, "p_set"]
        sector = charging_points.loc[cp_name, "sector"]

        # Different profiles based on sector
        if sector == "home":
            # Home charging: evening/night (18:00-07:00), higher on weekends
            peak_hours = (hour_of_day >= 18) | (hour_of_day <= 7)
            base_profile = np.where(peak_hours, 0.7, 0.1)
            # Higher usage on weekends
            weekend_mask = day_of_week >= 5
            base_profile[weekend_mask] *= 1.3

        elif sector == "work":
            # Work charging: daytime on weekdays (08:00-17:00)
            work_hours = (hour_of_day >= 8) & (hour_of_day <= 17)
            weekday_mask = day_of_week < 5
            base_profile = np.where(work_hours & weekday_mask, 0.6, 0.05)

        else:  # public/fast charging
            # Public charging: distributed throughout day, peaks at noon and evening
            base_profile = 0.3 + 0.4 * (
                np.exp(-((hour_of_day - 12) ** 2) / 12)  # Noon peak
                + np.exp(-((hour_of_day - 18) ** 2) / 12)  # Evening peak
            )

        # Add randomness and individual variation
        random_factor = 0.7 + 0.6 * np.random.random(len(timeindex))
        cp_load = base_profile * p_set * random_factor

        # Add some random noise
        cp_load += np.random.normal(0, p_set * 0.05, len(timeindex))
        cp_load = np.maximum(cp_load, 0)  # No negative load
        cp_load = np.minimum(cp_load, p_set)  # Cap at nominal power

        cp_load_data[cp_name] = cp_load

    # Add CP loads to timeseries (they will be added to loads_active_power)
    cp_load_df = pd.DataFrame(cp_load_data, index=timeindex)

    # Store for later use
    if not hasattr(edisgo, "charging_point_loads"):
        edisgo.charging_point_loads = cp_load_df
    else:
        edisgo.charging_point_loads = cp_load_df

    print(f"  ✓ Created time series:")
    print(
        f"    Load range: {cp_load_df.min().min():.6f} - {cp_load_df.max().max():.6f} MW"
    )
    print(f"    Average load: {cp_load_df.mean().mean():.6f} MW")

    return edisgo


def setup_edisgo(
    grid_path, scenario="eGon2035", num_hps=50, num_cps=30, num_days=30
):
    """
    Load grid and setup components for time series analysis.

    Parameters
    ----------
    grid_path : str
        Path to ding0 grid folder
    scenario : str
        Scenario name (default: eGon2035)
    num_hps : int
        Number of heat pumps to add (default: 50)
    num_cps : int
        Number of charging points to add (default: 30)
    num_days : int
        Number of days to simulate (default: 30)

    Returns
    -------
    EDisGo
        Initialized EDisGo object with time series
    """
    print(f"\n{'='*80}")
    print(f"🔧 Setting up EDisGo Grid")
    print(f"{'='*80}")
    print(f"Grid path: {grid_path}")
    print(f"Scenario: {scenario}")

    # Load grid
    print("\n1. Loading ding0 grid...")

    edisgo = EDisGo(ding0_grid=grid_path, legacy_ding0_grids=False)

    edisgo.topology.loads_df = edisgo.topology.loads_df[
        ~edisgo.topology.loads_df.type.isin(["charging_point", "heat_pump"])
    ]

    # Set the timeindex
    num_timesteps = num_days * 24
    timeindex = pd.date_range(
        "2035-01-15", periods=num_timesteps, freq="h"
    )  # Mid-winter
    edisgo.timeseries.timeindex = timeindex

    # Add heat pumps manually
    edisgo = add_heat_pumps_manually(edisgo, num_hps=num_hps)

    # Add charging points manually
    edisgo = add_charging_points_manually(edisgo, num_cps=num_cps)

    # Create HP time series
    edisgo = create_hp_timeseries(edisgo, scenario=scenario, num_days=num_days)

    # Create CP time series
    edisgo = create_cp_timeseries(edisgo, scenario=scenario, num_days=num_days)

    # Store HP timeindex
    hp_timeindex = edisgo.timeseries.timeindex
    num_timesteps = len(hp_timeindex)

    # Set time series for other components (generators, loads)
    print("\n5. Setting time series for generators and loads...")

    # Create simple time series for generators (use nominal power)
    generators = edisgo.topology.generators_df
    p_max_pu = pd.read_csv(
        grid_path + "/timeseries/gen_p_max_pu_timeseries.csv",
        index_col="snapshot",
        parse_dates=["snapshot"],
    )
    # TODO: Check why the years do not match
    p_max_pu = p_max_pu.iloc[0 : len(edisgo.timeseries.timeindex), :]
    p_max_pu.index = edisgo.timeseries.timeindex
    edisgo.timeseries.active_power_p_max_pu(
        edisgo, ts_generators_p_max_pu=p_max_pu
    )

    gen_reactive = pd.DataFrame(
        data=0.0,
        columns=edisgo.topology.generators_df.index,
        index=edisgo.timeseries.timeindex,
    )
    edisgo.timeseries._generators_reactive_power = gen_reactive

    print(f"  ✓ Created time series for {len(generators)} generators")

    # Create simple time series for other loads (use nominal power)

    other_loads = edisgo.topology.loads_df[
        ~edisgo.topology.loads_df["type"].isin(["heat_pump", "charging_point"])
    ]
    p_set = pd.read_csv(
        grid_path + "/timeseries/load_timeseries.csv",
        index_col="snapshot",
        parse_dates=["snapshot"],
        usecols=lambda c: c == "snapshot" or c in other_loads.index,
    )
    p_set = p_set.iloc[0 : len(edisgo.timeseries.timeindex), :]
    p_set.index = edisgo.timeseries.timeindex
    edisgo.timeseries._loads_active_power = p_set

    edisgo.timeseries._loads_reactive_power = pd.DataFrame(
        0.0,
        index=edisgo.timeseries.timeindex,
        columns=edisgo.topology.loads_df.index,
    )
    print(f"  ✓ Created time series for {len(other_loads)} other loads")

    # Calculate HP loads from heat demand and COP
    print("6. Calculating heat pump electrical loads...")
    heat_pumps = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "heat_pump"
    ]

    # Initialize loads_active_power with existing data if any, or create empty
    if (
        not hasattr(edisgo.timeseries, "_loads_active_power")
        or edisgo.timeseries._loads_active_power is None
    ):
        edisgo.timeseries._loads_active_power = pd.DataFrame(
            index=hp_timeindex
        )

    if (
        not hasattr(edisgo.timeseries, "_loads_reactive_power")
        or edisgo.timeseries._loads_reactive_power is None
    ):
        edisgo.timeseries._loads_reactive_power = pd.DataFrame(
            index=hp_timeindex
        )

    # Add HP electrical loads
    for hp_name in heat_pumps.index:
        hp_load = (
            edisgo.heat_pump.heat_demand_df[hp_name]
            / edisgo.heat_pump.cop_df[hp_name]
        )
        edisgo.timeseries._loads_active_power[hp_name] = hp_load.values
        edisgo.timeseries._loads_reactive_power[hp_name] = 0.0

    print(f"  ✓ Calculated electrical loads for {len(heat_pumps)} heat pumps")

    # Add charging point loads
    if hasattr(edisgo, "charging_point_loads"):
        charging_points = edisgo.topology.loads_df[
            edisgo.topology.loads_df["type"] == "charging_point"
        ]
        for cp_name in charging_points.index:
            edisgo.timeseries._loads_active_power[cp_name] = (
                edisgo.charging_point_loads[cp_name].values
            )
            edisgo.timeseries._loads_reactive_power[cp_name] = 0.0
        print(f"  ✓ Added loads for {len(charging_points)} charging points")

    # Initial analysis
    print("7. Running initial power flow analysis...")
    # edisgo.topology.transformers_df.at["trafo_bus_319", "s_nom"] = 1000
    # edisgo.topology.lines_df.x = 0.01
    # edisgo.topology.lines_df.r = 0.01
    # edisgo.topology.lines_df.s_nom = 1000
    # edisgo.topology.generators_df

    pypsa_network = edisgo.to_pypsa()
    # pypsa_network.export_to_csv_folder("/home/carlos/LoMa/validation/solve_pf_problem")
    # breakpoint()
    pypsa_network.pf()
    edisgo.analyze()

    print("\n✓ Grid setup complete!")
    print(f"  Timeindex: {len(edisgo.timeseries.timeindex)} timesteps")
    print(f"  Start: {edisgo.timeseries.timeindex[0]}")
    print(f"  End: {edisgo.timeseries.timeindex[-1]}")

    # Heat pump statistics
    print(f"\n  Heat pumps: {len(heat_pumps)}")
    print(
        f"  Power range: {heat_pumps['p_set'].min()*1000:.1f} - {heat_pumps['p_set'].max()*1000:.1f} kW"
    )
    print(
        f"  §14a eligible (>4.2 kW): {len(heat_pumps[heat_pumps['p_set'] > 0.0042])}"
    )

    # Charging point statistics
    charging_points = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "charging_point"
    ]
    if len(charging_points) > 0:
        print(f"\n  Charging points: {len(charging_points)}")
        print(
            f"  Power range: {charging_points['p_set'].min()*1000:.1f} - {charging_points['p_set'].max()*1000:.1f} kW"
        )
        print(
            f"  §14a eligible (>4.2 kW): {len(charging_points[charging_points['p_set'] > 0.0042])}"
        )

        # DEBUG: Verify charging_points_df property works
        cp_via_property = edisgo.topology.charging_points_df
        print(
            f"  DEBUG: topology.charging_points_df returns {len(cp_via_property)} CPs"
        )
        if len(cp_via_property) != len(charging_points):
            print(f"  ⚠️ WARNING: Mismatch between direct query and property!")

    return edisgo


def run_optimization_14a(edisgo):
    """
    Run optimization with §14a curtailment enabled.

    Uses opf_version=3 which minimizes line losses, maximal line loading, and HV slacks.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with time series

    Returns
    -------
    EDisGo
        EDisGo object with optimization results
    """
    print(f"\n{'='*80}")
    print(f"⚡ Running OPF with §14a Curtailment")
    print(f"{'='*80}")
    print(f"\nUsing OPF version 3:")
    print(f"  - Minimize line losses")
    print(f"  - Minimize maximal line loading")
    print(f"  - Minimize HV slacks")
    print(f"  - §14a curtailment enabled for heat pumps and charging points")

    start_time = datetime.now()

    # Run optimization
    edisgo.pm_optimize(opf_version=5, curtailment_14a=True)

    duration = (datetime.now() - start_time).total_seconds()

    print(f"\n✓ Optimization complete!")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    return edisgo


def analyze_curtailment_results(edisgo, output_dir="results_14a"):
    """
    Analyze §14a curtailment results and generate statistics.

    Parameters
    ----------
    edisgo : EDisGo
        EDisGo object with optimization results
    output_dir : str
        Directory to save results

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"📊 Analyzing §14a Curtailment Results")
    print(f"{'='*80}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get curtailment data for both heat pumps and charging points
    hp_gen_cols = [
        col
        for col in edisgo.timeseries.generators_active_power.columns
        if "hp_14a_support" in col
    ]
    cp_gen_cols = [
        col
        for col in edisgo.timeseries.generators_active_power.columns
        if "cp_14a_support" in col or "charging_point_14a_support" in col
    ]

    all_gen_cols = hp_gen_cols + cp_gen_cols

    if len(all_gen_cols) == 0:
        print("⚠ WARNING: No §14a virtual generators found in results!")
        return {}

    curtailment = edisgo.timeseries.generators_active_power[all_gen_cols]

    # Get heat pump and charging point load data
    heat_pumps = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "heat_pump"
    ]
    charging_points = edisgo.topology.loads_df[
        edisgo.topology.loads_df["type"] == "charging_point"
    ]

    all_flexible_loads = pd.concat([heat_pumps, charging_points])
    flexible_loads = edisgo.timeseries.loads_active_power[
        all_flexible_loads.index
    ]

    # Separate for detailed analysis
    hp_loads = (
        edisgo.timeseries.loads_active_power[heat_pumps.index]
        if len(heat_pumps) > 0
        else pd.DataFrame()
    )
    cp_loads = (
        edisgo.timeseries.loads_active_power[charging_points.index]
        if len(charging_points) > 0
        else pd.DataFrame()
    )

    # Calculate statistics
    total_curtailment = curtailment.sum().sum()
    total_flexible_load = flexible_loads.sum().sum()
    total_hp_load = hp_loads.sum().sum() if len(hp_loads) > 0 else 0
    total_cp_load = cp_loads.sum().sum() if len(cp_loads) > 0 else 0
    curtailment_percentage = (
        (total_curtailment / total_flexible_load * 100)
        if total_flexible_load > 0
        else 0
    )

    flexible_curtailment_total = curtailment.sum()
    curtailed_units = flexible_curtailment_total[
        flexible_curtailment_total > 0
    ]

    # Separate HP and CP curtailment
    hp_curtailment_total = (
        curtailment[hp_gen_cols].sum() if len(hp_gen_cols) > 0 else pd.Series()
    )
    cp_curtailment_total = (
        curtailment[cp_gen_cols].sum() if len(cp_gen_cols) > 0 else pd.Series()
    )
    curtailed_hps = (
        hp_curtailment_total[hp_curtailment_total > 0]
        if len(hp_curtailment_total) > 0
        else pd.Series()
    )
    curtailed_cps = (
        cp_curtailment_total[cp_curtailment_total > 0]
        if len(cp_curtailment_total) > 0
        else pd.Series()
    )

    # Time series statistics
    curtailment_per_timestep = curtailment.sum(axis=1)
    max_curtailment_timestep = curtailment_per_timestep.idxmax()
    max_curtailment_value = curtailment_per_timestep.max()

    # Daily statistics
    curtailment_daily = curtailment_per_timestep.resample("D").sum()

    # Monthly statistics
    curtailment_monthly = curtailment_per_timestep.resample("M").sum()

    results = {
        "total_curtailment_MWh": total_curtailment,
        "total_flexible_load_MWh": total_flexible_load,
        "total_hp_load_MWh": total_hp_load,
        "total_cp_load_MWh": total_cp_load,
        "curtailment_percentage": curtailment_percentage,
        "num_virtual_gens": len(all_gen_cols),
        "num_hp_gens": len(hp_gen_cols),
        "num_cp_gens": len(cp_gen_cols),
        "num_curtailed_hps": len(curtailed_hps),
        "num_curtailed_cps": len(curtailed_cps),
        "max_curtailment_MW": curtailment.max().max(),
        "max_curtailment_timestep": max_curtailment_timestep,
        "max_curtailment_value_MW": max_curtailment_value,
        "curtailment_per_timestep": curtailment_per_timestep,
        "curtailment_daily": curtailment_daily,
        "curtailment_monthly": curtailment_monthly,
        "curtailment_data": curtailment,
        "hp_curtailment_data": (
            curtailment[hp_gen_cols]
            if len(hp_gen_cols) > 0
            else pd.DataFrame()
        ),
        "cp_curtailment_data": (
            curtailment[cp_gen_cols]
            if len(cp_gen_cols) > 0
            else pd.DataFrame()
        ),
        "hp_loads": hp_loads,
        "cp_loads": cp_loads,
        "flexible_loads": flexible_loads,
        "hp_curtailment_total": hp_curtailment_total,
        "cp_curtailment_total": cp_curtailment_total,
        "curtailed_hps": curtailed_hps,
        "curtailed_cps": curtailed_cps,
    }

    # Print summary
    print(f"\n📈 Summary Statistics:")
    print(
        f"  Virtual generators: {len(all_gen_cols)} (HPs: {len(hp_gen_cols)}, CPs: {len(cp_gen_cols)})"
    )
    print(f"  Heat pumps curtailed: {len(curtailed_hps)} / {len(heat_pumps)}")
    print(
        f"  Charging points curtailed: {len(curtailed_cps)} / {len(charging_points)}"
    )
    print(f"  Total curtailment: {total_curtailment:.2f} MWh")
    print(
        f"  Total flexible load: {total_flexible_load:.2f} MWh (HP: {total_hp_load:.2f}, CP: {total_cp_load:.2f})"
    )
    print(f"  Curtailment ratio: {curtailment_percentage:.2f}%")
    print(f"  Max curtailment: {curtailment.max().max():.4f} MW")
    print(
        f"  Max total curtailment (timestep): {max_curtailment_value:.4f} MW at {max_curtailment_timestep}"
    )

    if len(curtailed_hps) > 0:
        print(f"\n  Top 5 curtailed heat pumps:")
        for i, (hp, value) in enumerate(
            curtailed_hps.sort_values(ascending=False).head().items(), 1
        ):
            hp_name = hp.replace("hp_14a_support_", "")
            print(f"    {i}. {hp_name}: {value:.4f} MWh")

    if len(curtailed_cps) > 0:
        print(f"\n  Top 5 curtailed charging points:")
        for i, (cp, value) in enumerate(
            curtailed_cps.sort_values(ascending=False).head().items(), 1
        ):
            cp_name = cp.replace("cp_14a_support_", "").replace(
                "charging_point_14a_support_", ""
            )
            print(f"    {i}. {cp_name}: {value:.4f} MWh")

    # Save statistics to CSV
    stats_df = pd.DataFrame(
        {
            "Metric": [
                "Total Curtailment (MWh)",
                "Total Flexible Load (MWh)",
                "Total HP Load (MWh)",
                "Total CP Load (MWh)",
                "Curtailment Percentage (%)",
                "Virtual Generators (Total)",
                "Virtual Generators (HPs)",
                "Virtual Generators (CPs)",
                "Curtailed HPs",
                "Curtailed CPs",
                "Max Curtailment (MW)",
                "Max Total Curtailment (MW)",
            ],
            "Value": [
                f"{total_curtailment:.2f}",
                f"{total_flexible_load:.2f}",
                f"{total_hp_load:.2f}",
                f"{total_cp_load:.2f}",
                f"{curtailment_percentage:.2f}",
                len(all_gen_cols),
                len(hp_gen_cols),
                len(cp_gen_cols),
                len(curtailed_hps),
                len(curtailed_cps),
                f"{curtailment.max().max():.4f}",
                f"{max_curtailment_value:.4f}",
            ],
        }
    )
    stats_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    print(
        f"\n✓ Summary statistics saved to {output_dir}/summary_statistics.csv"
    )

    # Save detailed curtailment data
    curtailment.to_csv(f"{output_dir}/curtailment_timeseries.csv")
    curtailment_daily.to_csv(f"{output_dir}/curtailment_daily.csv")
    curtailment_monthly.to_csv(f"{output_dir}/curtailment_monthly.csv")
    if len(hp_curtailment_total) > 0:
        hp_curtailment_total.to_csv(f"{output_dir}/hp_curtailment_total.csv")
    if len(cp_curtailment_total) > 0:
        cp_curtailment_total.to_csv(f"{output_dir}/cp_curtailment_total.csv")

    print(f"✓ Detailed data saved to {output_dir}/")

    return results


def create_plots(results, output_dir="results_14a"):
    """
    Create comprehensive visualization plots.

    Parameters
    ----------
    results : dict
        Results dictionary from analyze_curtailment_results
    output_dir : str
        Directory to save plots
    """
    print(f"\n{'='*80}")
    print(f"📊 Creating Visualization Plots")
    print(f"{'='*80}")

    curtailment = results["curtailment_data"]
    hp_loads = results["hp_loads"]
    cp_loads = results.get("cp_loads", pd.DataFrame())
    curtailment_per_timestep = results["curtailment_per_timestep"]
    curtailment_daily = results["curtailment_daily"]
    curtailment_monthly = results["curtailment_monthly"]
    hp_curtailment_total = results["hp_curtailment_total"]
    cp_curtailment_total = results.get("cp_curtailment_total", pd.Series())
    curtailed_hps = results["curtailed_hps"]
    curtailed_cps = results.get("curtailed_cps", pd.Series())

    # Plot 1: Time series curtailment
    num_days = len(curtailment_per_timestep) // 24
    print(f"1. Creating {num_days}-day curtailment plot...")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        curtailment_per_timestep.index,
        curtailment_per_timestep.values,
        "r-",
        linewidth=1.5,
        alpha=0.7,
        marker="o",
        markersize=3,
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Total Curtailment (MW)", fontsize=12)
    ax.set_title(
        f"§14a Heat Pump & Charging Point Curtailment - {num_days} Days",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/01_curtailment_timeseries.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 2: Daily curtailment
    print("2. Creating daily curtailment plot...")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(
        curtailment_daily.index,
        curtailment_daily.values,
        width=1,
        color="red",
        alpha=0.7,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Daily Curtailment (MWh)", fontsize=12)
    ax.set_title("§14a Daily Curtailment", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/02_curtailment_daily.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Monthly curtailment
    print("3. Creating monthly curtailment plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    months = [d.strftime("%b %Y") for d in curtailment_monthly.index]
    ax.bar(months, curtailment_monthly.values, color="red", alpha=0.7)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Monthly Curtailment (MWh)", fontsize=12)
    ax.set_title("§14a Monthly Curtailment", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/03_curtailment_monthly.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 4: Top 10 curtailed units (HPs and CPs combined)
    print("4. Creating top curtailed units plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    all_curtailed_list = []
    if len(curtailed_hps) > 0:
        all_curtailed_list.append(curtailed_hps)
    if len(curtailed_cps) > 0:
        all_curtailed_list.append(curtailed_cps)

    if len(all_curtailed_list) > 0:
        all_curtailed = pd.concat(all_curtailed_list)
        top10 = all_curtailed.sort_values(ascending=False).head(10)
        unit_names = [
            name.replace("hp_14a_support_", "HP: ")
            .replace("cp_14a_support_", "CP: ")
            .replace("charging_point_14a_support_", "CP: ")
            for name in top10.index
        ]
        colors = ["blue" if "HP:" in name else "green" for name in unit_names]
        ax.barh(unit_names, top10.values, color=colors, alpha=0.7)
    else:
        ax.text(
            0.5,
            0.5,
            "No curtailed units found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Total Curtailment (MWh)", fontsize=12)
    ax.set_ylabel("Unit", fontsize=12)
    ax.set_title(
        "Top 10 Curtailed Units (Heat Pumps & Charging Points)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/04_top10_curtailed_hps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 5: Curtailment distribution (histogram)
    print("5. Creating curtailment distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    curtailment_nonzero = curtailment_per_timestep[
        curtailment_per_timestep > 0
    ]
    ax.hist(
        curtailment_nonzero.values,
        bins=50,
        color="red",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Curtailment (MW)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Distribution of Non-Zero Curtailment Events",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/05_curtailment_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Plot 6: Detailed view of most curtailed HP
    print("6. Creating detailed HP profile plot...")
    most_curtailed = hp_curtailment_total.idxmax()
    hp_original_name = most_curtailed.replace("hp_14a_support_", "")

    if hp_original_name in hp_loads.columns:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Full year
        original_load = hp_loads[hp_original_name]
        curtailment_power = curtailment[most_curtailed]
        net_load = original_load - curtailment_power

        ax1 = axes[0]
        ax1.plot(
            original_load.index,
            original_load.values,
            "b-",
            linewidth=0.5,
            label="Original Load",
            alpha=0.7,
        )
        ax1.plot(
            net_load.index,
            net_load.values,
            "g-",
            linewidth=0.5,
            label="Net Load (after curtailment)",
            alpha=0.7,
        )
        ax1.axhline(
            y=0.0042,
            color="orange",
            linestyle="--",
            linewidth=1,
            label="§14a Minimum (4.2 kW)",
        )
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("Power (MW)", fontsize=12)
        ax1.set_title(
            f"{hp_original_name} - Full Year Profile",
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Sample week (first week with curtailment)
        curtailment_weeks = curtailment_power.resample("W").sum()
        first_curtailment_week = curtailment_weeks[
            curtailment_weeks > 0
        ].index[0]
        week_start = first_curtailment_week
        week_end = week_start + pd.Timedelta(days=7)

        ax2 = axes[1]
        week_mask = (original_load.index >= week_start) & (
            original_load.index < week_end
        )
        ax2.plot(
            original_load.index[week_mask],
            original_load.values[week_mask],
            "b-",
            marker="o",
            linewidth=2,
            label="Original Load",
            markersize=3,
        )
        ax2.plot(
            net_load.index[week_mask],
            net_load.values[week_mask],
            "g-",
            marker="s",
            linewidth=2,
            label="Net Load",
            markersize=3,
        )
        ax2.fill_between(
            original_load.index[week_mask],
            net_load.values[week_mask],
            original_load.values[week_mask],
            alpha=0.3,
            color="red",
            label="Curtailed Power",
        )
        ax2.axhline(
            y=0.0042,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="§14a Minimum (4.2 kW)",
        )
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Power (MW)", fontsize=12)
        ax2.set_title(
            f"{hp_original_name} - Sample Week with Curtailment",
            fontsize=14,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/detailed_hp_profiles.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Plot 3: Charging Point Analysis (if CPs were curtailed)
    if "curtailed_cps" in results and len(results["curtailed_cps"]) > 0:
        print("3. Creating detailed charging point profile plot...")

        cp_curtailment = results["cp_curtailment_data"]
        cp_loads = results["cp_loads"]
        curtailed_cps = results["curtailed_cps"]

        # Most curtailed CP detail
        most_curtailed_cp = curtailed_cps.idxmax()
        cp_original_name = most_curtailed_cp.replace("cp_14a_support_", "")

        if cp_original_name in cp_loads.columns:
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))

            original_load = cp_loads[cp_original_name]
            curtailment_power = cp_curtailment[most_curtailed_cp]
            net_load = original_load - curtailment_power

            # Full period
            ax1 = axes[0]
            ax1.plot(
                original_load.index,
                original_load.values,
                "b-",
                linewidth=0.5,
                label="Original Load",
                alpha=0.7,
            )
            ax1.plot(
                net_load.index,
                net_load.values,
                "g-",
                linewidth=0.5,
                label="Net Load (after curtailment)",
                alpha=0.7,
            )
            ax1.axhline(
                y=0.0042,
                color="orange",
                linestyle="--",
                linewidth=1,
                label="§14a Minimum (4.2 kW)",
            )
            ax1.set_xlabel("Time", fontsize=12)
            ax1.set_ylabel("Power (MW)", fontsize=12)
            ax1.set_title(
                f"{cp_original_name} - Full Period Profile",
                fontsize=14,
                fontweight="bold",
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Sample week with curtailment
            curtailment_weeks = curtailment_power.resample("W").sum()
            first_curtailment_week = (
                curtailment_weeks[curtailment_weeks > 0].index[0]
                if any(curtailment_weeks > 0)
                else curtailment_weeks.index[0]
            )
            week_start = first_curtailment_week
            week_end = week_start + pd.Timedelta(days=7)

            ax2 = axes[1]
            week_mask = (original_load.index >= week_start) & (
                original_load.index < week_end
            )
            ax2.plot(
                original_load.index[week_mask],
                original_load.values[week_mask],
                "b-",
                marker="o",
                linewidth=2,
                label="Original Load",
                markersize=3,
            )
            ax2.plot(
                net_load.index[week_mask],
                net_load.values[week_mask],
                "g-",
                marker="s",
                linewidth=2,
                label="Net Load",
                markersize=3,
            )
            ax2.fill_between(
                original_load.index[week_mask],
                net_load.values[week_mask],
                original_load.values[week_mask],
                alpha=0.3,
                color="orange",
                label="Curtailed Power",
            )
            ax2.axhline(
                y=0.0042,
                color="orange",
                linestyle="--",
                linewidth=2,
                label="§14a Minimum (4.2 kW)",
            )
            ax2.set_xlabel("Time", fontsize=12)
            ax2.set_ylabel("Power (MW)", fontsize=12)
            ax2.set_title(
                f"{cp_original_name} - Sample Week",
                fontsize=14,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/detailed_cp_profiles.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    print(f"\n✓ All plots saved to {output_dir}/")


def main():
    # ============================================================================
    # CONFIGURATION - Edit these values directly
    # ============================================================================

    # Grid configuration
    # GRID_PATH = '/home/carlos/LoMa/eDisGo/30879'
    GRID_PATH = (
        "/home/student/Execution/LoMa_exe/results/MGB_Model_V3"  # MGB_Husum_model'
    )
    SCENARIO = "eGon2035"

    # Simulation parameters
    NUM_DAYS = 10 / 24  # Number of days to simulate (e.g., 7, 30, 365)
    NUM_HEAT_PUMPS = 20  # Number of heat pumps to add
    NUM_CHARGING_POINTS = 20  # Number of charging points to add

    # Output
    OUTPUT_DIR = "./"

    # ============================================================================
    # END CONFIGURATION
    # ============================================================================
    # Create directory name from configuration parameters
    output_dir = f"{OUTPUT_DIR}/results_{NUM_DAYS}d_HP{NUM_HEAT_PUMPS}_CP{NUM_CHARGING_POINTS}_14a"
    print(f"\n{'#'*80}")
    print(f"# §14a EnWG Heat Pump Curtailment Analysis - {NUM_DAYS} Days")
    print(f"{'#'*80}")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Setup grid and load data
        edisgo = setup_edisgo(
            GRID_PATH,
            scenario=SCENARIO,
            num_hps=NUM_HEAT_PUMPS,
            num_cps=NUM_CHARGING_POINTS,
            num_days=NUM_DAYS,
        )

        # Run optimization with §14a
        edisgo = run_optimization_14a(edisgo)
        
        #update line_loading and voltage values
        edisgo.analyze()

        # Analyze results
        results = analyze_curtailment_results(edisgo, output_dir=output_dir)

        if results:
            # Create plots
            create_plots(results, output_dir=output_dir)

            print(f"\n{'='*80}")
            print(f"✓ Analysis Complete!")
            print(f"{'='*80}")
            print(f"\nResults saved to: {output_dir}/")
            print(f"  - summary_statistics.csv")
            print(f"  - curtailment_timeseries.csv")
            print(f"  - curtailment_daily.csv")
            print(f"  - curtailment_monthly.csv")
            print(f"  - hp_curtailment_total.csv")
            print(f"  - cp_curtailment_total.csv")
            print(f"  - curtailment_timeseries.png")
            print(f"  - detailed_hp_profiles.png")
            print(f"  - detailed_cp_profiles.png (if CPs curtailed)")

        print(
            f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    return edisgo


def create_network_gif(
    folder_path="./plots", output_name="network_evolution.gif", duration=1
):
    """
    Creates a GIF from PNG files in a folder.
    duration: time in seconds between frames
    """
    images = []

    # Get all png files that start with 'grid_analysis_'
    files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".png") and f.startswith("grid_analysis_")
    ]

    # IMPORTANT: Sort files by time.
    # Since your files are named 'grid_analysis_YYYY-MM-DD HH:MM:SS.png',
    # standard string sorting works perfectly for chronological order.
    files.sort()

    print(f"Found {len(files)} frames. Processing...")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(file_path))
        print(f"Added: {filename}")

    # Save the GIF
    # loop=0 means it will loop forever
    imageio.mimsave(output_name, images, duration=duration, loop=0)
    print(f"Success! GIF saved as {output_name}")


def plot_network(
    edisgo,
    snapshot: str = "2035-01-15 09:00:00",
    show: bool = True,
    save: bool = True,
    base_bus_size = 0.000000002
):
    results = edisgo.results
    
    n = edisgo.to_pypsa()
    coords = edisgo.topology.buses_df[["x", "y"]]
    coords = coords.reindex(n.buses.index)  #secure that index is matching
    n.buses["x"] = coords["x"].values
    n.buses["y"] = coords["y"].values
    
    line_columns = n.lines.index
    lines_t = results.s_res.loc[:, line_columns]

    # 1. Define limits for line loading
    loading_relative = results.s_res.loc[snapshot, line_columns] / n.lines.s_nom

    # 1. Limits für Farbskala (jetzt auf 0% - 100% bezogen)
    v_min, v_max = 0.0, 1.0
    norm_lines = mcolors.Normalize(vmin=v_min, vmax=v_max)

    # 2. Prepare bus data
    # Calculating voltage deviation from nominal (1.0 p.u.)
    bus_colors = (1 - edisgo.results.v_res.T[snapshot]).apply(abs)

    # Voltage limits (adjust vmin/vmax based on your bus_colors results)
    norm_buses = mcolors.Normalize(
        vmin=0.0, vmax=0.3
    )

    # --- (Curtailment logic and bus_sizes calculation) ---
    curt_14a = analyze_curtailment_results(edisgo, output_dir="results_14a")[
        "curtailment_data"
    ].T

    # Clean up index names to match load names
    curt_14a["load"] = curt_14a.index
    curt_14a["load"] = curt_14a["load"].apply(
        lambda x: x.replace("cp_14a_support_", "").replace(
            "hp_14a_support_", ""
        )
    )

    # Map loads to their respective buses and aggregate curtailment per bus
    curt_14a["bus"] = curt_14a["load"].map(edisgo.topology.loads_df["bus"])
    grouped_14a = curt_14a.groupby("bus").sum()
    grouped_14a.columns = grouped_14a.columns.map(str)

    # Calculate bus sizes based on curtailment; reindex to include all buses in the network
    bus_sizes = base_bus_size + (grouped_14a[snapshot] * 0.000001)
    bus_sizes = bus_sizes.reindex(bus_colors.index, fill_value=base_bus_size)
    # -------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the grid
    n.plot(
        margin=0.05,
        ax=ax,
        geomap=False,
        bus_colors=bus_colors,
        bus_alpha=1,
        bus_sizes=bus_sizes,
        bus_cmap="jet",
        bus_norm=norm_buses,
        line_colors=loading_relative,
        line_widths=1.6,
        line_cmap="jet",
        line_norm=norm_lines,
        title=f"Grid Analysis: {snapshot}",
        geometry=False,
    )

    # Add background basemap
    ctx.add_basemap(ax, crs=4326, source=ctx.providers.OpenStreetMap.Mapnik)

    # --- COLORBAR 1: LINE LOADING (LEFT SIDE) ---
    sm_lines = plt.cm.ScalarMappable(cmap="jet", norm=norm_lines)
    # Use location='left' and a slightly larger pad to avoid overlap with axis labels
    cb_lines = fig.colorbar(
        sm_lines,
        ax=ax,
        orientation="vertical",
        location="left",
        pad=0.08,
        aspect=20,
    )
    cb_lines.set_label("Line Loading [relative]", fontsize=8)

    # --- COLORBAR 2: BUS VOLTAGE (RIGHT SIDE) ---
    sm_buses = plt.cm.ScalarMappable(cmap="jet", norm=norm_buses)
    # Default location is right
    cb_buses = fig.colorbar(
        sm_buses,
        ax=ax,
        orientation="vertical",
        location="right",
        pad=0.02,
        aspect=20,
    )
    cb_buses.set_label("Voltage Deviation |1 - V| [p.u.]", fontsize=8)

    if save:
        plt.savefig(
            f"plots/grid_analysis_{snapshot}.png", dpi=300, bbox_inches="tight"
        )

    if show:
        plt.show()
<<<<<<< HEAD
'''
edisgo = main()
for ts in edisgo.timeseries.timeindex:
    plot_network(edisgo, show=False, snapshot=str(ts))
create_network_gif(output_name='network_evolution.gif', duration=500)
'''
=======

edisgo = main()
for ts in edisgo.timeseries.timeindex:
    plot_network(edisgo, show=False, snapshot=str(ts))
create_network_gif(duration=500)
>>>>>>> project/411_LoMa_14aOptimization_with_virtual_generators
