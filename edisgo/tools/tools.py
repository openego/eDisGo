from __future__ import annotations

import logging
import os

from math import pi, sqrt

import networkx as nx
import numpy as np
import pandas as pd

from sqlalchemy import func

from edisgo.flex_opt import check_tech_constraints, exceptions
from edisgo.tools import session_scope

if "READTHEDOCS" not in os.environ:

    import geopandas as gpd

    from egoio.db_tables import climate
    from shapely.geometry.multipolygon import MultiPolygon
    from shapely.wkt import loads as wkt_loads


logger = logging.getLogger(__name__)


def select_worstcase_snapshots(edisgo_obj):
    """
    Select two worst-case snapshots from time series

    Two time steps in a time series represent worst-case snapshots. These are

    1. Maximum Residual Load: refers to the point in the time series where the
        (load - generation) achieves its maximum.
    2. Minimum Residual Load: refers to the point in the time series where the
        (load - generation) achieves its minimum.

    These two points are identified based on the generation and load time
    series. In case load or feed-in case don't exist None is returned.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :obj:`dict`
        Dictionary with keys 'min_residual_load' and 'max_residual_load'.
        Values are corresponding worst-case snapshots of type
        :pandas:`pandas.Timestamp<Timestamp>`.

    """
    residual_load = edisgo_obj.timeseries.residual_load

    timestamp = {
        "min_residual_load": residual_load.idxmin(),
        "max_residual_load": residual_load.idxmax(),
    }

    return timestamp


def calculate_relative_line_load(edisgo_obj, lines=None, timesteps=None):
    """
    Calculates relative line loading for specified lines and time steps.

    Line loading is calculated by dividing the current at the given time step
    by the allowed current.


    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    lines : list(str) or None, optional
        Line names/representatives of lines to calculate line loading for. If
        None, line loading is calculated for all lines in the network.
        Default: None.
    timesteps : :pandas:`pandas.Timestamp<Timestamp>` or \
        list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
        Specifies time steps to calculate line loading for. If timesteps is
        None, all time steps power flow analysis was conducted for are used.
        Default: None.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with relative line loading (unitless). Index of
        the dataframe is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`,
        columns are the line representatives.

    """
    if timesteps is None:
        timesteps = edisgo_obj.results.i_res.index
    # check if timesteps is array-like, otherwise convert to list
    if not hasattr(timesteps, "__len__"):
        timesteps = [timesteps]

    if lines is not None:
        line_indices = lines
    else:
        line_indices = edisgo_obj.topology.lines_df.index

    mv_lines_allowed_load = check_tech_constraints.lines_allowed_load(edisgo_obj, "mv")
    lv_lines_allowed_load = check_tech_constraints.lines_allowed_load(edisgo_obj, "lv")
    lines_allowed_load = pd.concat(
        [mv_lines_allowed_load, lv_lines_allowed_load], axis=1, sort=False
    ).loc[timesteps, line_indices]

    return check_tech_constraints.lines_relative_load(edisgo_obj, lines_allowed_load)


def calculate_line_reactance(line_inductance_per_km, line_length, num_parallel):
    """
    Calculates line reactance in Ohm.

    Parameters
    ----------
    line_inductance_per_km : float or array-like
        Line inductance in mH/km.
    line_length : float
        Length of line in km.
    num_parallel : int
        Number of parallel lines.

    Returns
    -------
    float
        Reactance in Ohm

    """
    return line_inductance_per_km / 1e3 * line_length * 2 * pi * 50 / num_parallel


def calculate_line_resistance(line_resistance_per_km, line_length, num_parallel):
    """
    Calculates line resistance in Ohm.

    Parameters
    ----------
    line_resistance_per_km : float or array-like
        Line resistance in Ohm/km.
    line_length : float
        Length of line in km.
    num_parallel : int
        Number of parallel lines.

    Returns
    -------
    float
        Resistance in Ohm

    """
    return line_resistance_per_km * line_length / num_parallel


def calculate_line_susceptance(line_capacitance_per_km, line_length, num_parallel):
    """
    Calculates line shunt susceptance in Siemens.

    Parameters
    ----------
    line_capacitance_per_km : float
        Line capacitance in uF/km.
    line_length : float
        Length of line in km.
    num_parallel : int
        Number of parallel lines.

    Returns
    -------
    float
        Shunt susceptance in Siemens.

    """
    return line_capacitance_per_km / 1e6 * line_length * 2 * pi * 50 * num_parallel


def calculate_apparent_power(nominal_voltage, current, num_parallel):
    """
    Calculates apparent power in MVA from given voltage and current.

    Parameters
    ----------
    nominal_voltage : float or array-like
        Nominal voltage in kV.
    current : float or array-like
        Current in kA.
    num_parallel : int or array-like
        Number of parallel lines.

    Returns
    -------
    float
        Apparent power in MVA.

    """
    return sqrt(3) * nominal_voltage * current * num_parallel


def drop_duplicated_indices(dataframe, keep="first"):
    """
    Drop rows of duplicate indices in dataframe.

    Parameters
    ----------
    dataframe::pandas:`pandas.DataFrame<DataFrame>`
        handled dataframe
    keep: str
        indicator of row to be kept, 'first', 'last' or False,
        see pandas.DataFrame.drop_duplicates() method
    """
    return dataframe[~dataframe.index.duplicated(keep=keep)]


def drop_duplicated_columns(df, keep="first"):
    """
    Drop columns of dataframe that appear more than once.

    Parameters
    ----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe of which columns are dropped.
    keep : str
        Indicator of whether to keep first ('first'), last ('last') or
        none (False) of the duplicated columns.
        See `drop_duplicates()` method of
        :pandas:`pandas.DataFrame<DataFrame>`.

    """
    return df.loc[:, ~df.columns.duplicated(keep=keep)]


def select_cable(edisgo_obj, level, apparent_power):
    """
    Selects suitable cable type and quantity using given apparent power.

    Cable is selected to be able to carry the given `apparent_power`, no load
    factor is considered. Overhead lines are not considered in choosing a
    suitable cable.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    level : str
        Grid level to get suitable cable for. Possible options are 'mv' or
        'lv'.
    apparent_power : float
        Apparent power the cable must carry in MVA.

    Returns
    -------
    :pandas:`pandas.Series<Series>`
        Series with attributes of selected cable as in equipment data and
        cable type as series name.
    int
        Number of necessary parallel cables.

    """

    cable_count = 1

    if level == "mv":
        cable_data = edisgo_obj.topology.equipment_data["mv_cables"]
        available_cables = cable_data[
            cable_data["U_n"] == edisgo_obj.topology.mv_grid.nominal_voltage
        ]
    elif level == "lv":
        available_cables = edisgo_obj.topology.equipment_data["lv_cables"]
    else:
        raise ValueError(
            "Specified voltage level is not valid. Must either be 'mv' or 'lv'."
        )

    suitable_cables = available_cables[
        calculate_apparent_power(
            available_cables["U_n"], available_cables["I_max_th"], cable_count
        )
        > apparent_power
    ]

    # increase cable count until appropriate cable type is found
    while suitable_cables.empty and cable_count < 7:
        cable_count += 1
        suitable_cables = available_cables[
            calculate_apparent_power(
                available_cables["U_n"],
                available_cables["I_max_th"],
                cable_count,
            )
            > apparent_power
        ]
    if suitable_cables.empty:
        raise exceptions.MaximumIterationError(
            "Could not find a suitable cable for apparent power of "
            "{} MVA.".format(apparent_power)
        )

    cable_type = suitable_cables.loc[suitable_cables["I_max_th"].idxmin()]

    return cable_type, cable_count


def assign_feeder(edisgo_obj, mode="mv_feeder"):
    """
    Assigns MV or LV feeder to each bus and line, depending on the `mode`.

    The feeder name is written to a new column `mv_feeder` or `lv_feeder`
    in :class:`~.network.topology.Topology`'s
    :attr:`~.network.topology.Topology.buses_df` and
    :attr:`~.network.topology.Topology.lines_df`. The MV respectively LV feeder
    name corresponds to the name of the first bus in the respective feeder.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
    mode : str
        Specifies whether to assign MV or LV feeder. Valid options are
        'mv_feeder' or 'lv_feeder'. Default: 'mv_feeder'.

    """

    def _assign_to_busses(graph, station):
        # get all buses in network and remove station to get separate subgraphs
        graph_nodes = list(graph.nodes())
        graph_nodes.remove(station)
        subgraph = graph.subgraph(graph_nodes)

        for neighbor in graph.neighbors(station):
            # get all nodes in that feeder by doing a DFS in the disconnected
            # subgraph starting from the node adjacent to the station
            # `neighbor`
            subgraph_neighbor = nx.dfs_tree(subgraph, source=neighbor)
            for node in subgraph_neighbor.nodes():

                edisgo_obj.topology.buses_df.at[node, mode] = neighbor

                # in case of an LV station, assign feeder to all nodes in that
                # LV network (only applies when mode is 'mv_feeder'
                if node.split("_")[0] == "BusBar" and node.split("_")[-1] == "MV":
                    lvgrid = edisgo_obj.topology.get_lv_grid(int(node.split("_")[-2]))
                    edisgo_obj.topology.buses_df.loc[
                        lvgrid.buses_df.index, mode
                    ] = neighbor

    def _assign_to_lines(lines):
        edisgo_obj.topology.lines_df.loc[
            lines, mode
        ] = edisgo_obj.topology.lines_df.loc[lines].apply(
            lambda _: edisgo_obj.topology.buses_df.at[_.bus0, mode], axis=1
        )
        tmp = edisgo_obj.topology.lines_df.loc[lines]
        lines_nan = tmp[tmp.loc[lines, mode].isna()].index
        edisgo_obj.topology.lines_df.loc[
            lines_nan, mode
        ] = edisgo_obj.topology.lines_df.loc[lines_nan].apply(
            lambda _: edisgo_obj.topology.buses_df.at[_.bus1, mode], axis=1
        )

    if mode == "mv_feeder":
        graph = edisgo_obj.topology.mv_grid.graph
        station = edisgo_obj.topology.mv_grid.station.index[0]
        _assign_to_busses(graph, station)
        lines = edisgo_obj.topology.lines_df.index
        _assign_to_lines(lines)

    elif mode == "lv_feeder":
        for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:
            graph = lv_grid.graph
            station = lv_grid.station.index[0]
            _assign_to_busses(graph, station)
            lines = lv_grid.lines_df.index
            _assign_to_lines(lines)

    else:
        raise ValueError(
            "Invalid mode. Mode must either be 'mv_feeder' or 'lv_feeder'."
        )


def get_path_length_to_station(edisgo_obj):
    """
    Determines path length from each bus to HV-MV station.

    The path length is written to a new column `path_length_to_station` in
    `buses_df` dataframe of :class:`~.network.topology.Topology` class.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.Series<Series>`
        Series with bus name in index and path length to station as value.

    """
    graph = edisgo_obj.topology.mv_grid.graph
    mv_station = edisgo_obj.topology.mv_grid.station.index[0]

    for bus in edisgo_obj.topology.mv_grid.buses_df.index:
        path = nx.shortest_path(graph, source=mv_station, target=bus)
        edisgo_obj.topology.buses_df.at[bus, "path_length_to_station"] = len(path) - 1
        if bus.split("_")[0] == "BusBar" and bus.split("_")[-1] == "MV":
            # check if there is an underlying LV grid
            lv_grid_id = int(bus.split("_")[-2])
            if lv_grid_id in edisgo_obj.topology._lv_grid_ids:
                lvgrid = edisgo_obj.topology.get_lv_grid(lv_grid_id)
                lv_graph = lvgrid.graph
                lv_station = lvgrid.station.index[0]
                for bus in lvgrid.buses_df.index:
                    lv_path = nx.shortest_path(lv_graph, source=lv_station, target=bus)
                    edisgo_obj.topology.buses_df.at[
                        bus, "path_length_to_station"
                    ] = len(path) + len(lv_path)
    return edisgo_obj.topology.buses_df.path_length_to_station


def assign_voltage_level_to_component(df, buses_df):
    """
    Adds column with specification of voltage level component is in.

    The voltage level ('mv' or 'lv') is determined based on the nominal
    voltage of the bus the component is connected to. If the nominal voltage
    is smaller than 1 kV, voltage level 'lv' is assigned, otherwise 'mv' is
    assigned.

    Parameters
    ----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names in the index. Only required column is
        column 'bus', giving the name of the bus the component is connected to.
    buses_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with bus information. Bus names are in the index. Only required column
        is column 'v_nom', giving the nominal voltage of the voltage level the
        bus is in.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Same dataframe as given in parameter `df` with new column
        'voltage_level' specifying the voltage level the component is in
        (either 'mv' or 'lv').

    """
    df["voltage_level"] = df.apply(
        lambda _: "lv" if buses_df.at[_.bus, "v_nom"] < 1 else "mv",
        axis=1,
    )
    return df


def get_weather_cells_intersecting_with_grid_district(edisgo_obj):
    """
    Get all weather cells that intersect with the grid district.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    set
        Set with weather cell IDs

    """

    # Download geometries of weather cells
    srid = edisgo_obj.topology.grid_district["srid"]
    table = climate.Cosmoclmgrid
    with session_scope() as session:
        query = session.query(
            table.gid,
            func.ST_AsText(func.ST_Transform(table.geom, srid)).label("geometry"),
        )
    geom_data = pd.read_sql_query(query.statement, query.session.bind)
    geom_data.geometry = geom_data.apply(lambda _: wkt_loads(_.geometry), axis=1)
    geom_data = gpd.GeoDataFrame(geom_data, crs=f"EPSG:{srid}")

    # Make sure MV Geometry is MultiPolygon
    mv_geom = edisgo_obj.topology.grid_district["geom"]
    if mv_geom.geom_type == "Polygon":
        # Transform Polygon to MultiPolygon and overwrite geometry
        p = wkt_loads(str(mv_geom))
        m = MultiPolygon([p])
        edisgo_obj.topology.grid_district["geom"] = m
    elif mv_geom.geom_type == "MultiPolygon":
        m = mv_geom
    else:
        raise ValueError(
            f"Grid district geometry is of type {type(mv_geom)}."
            " Only Shapely Polygon or MultiPolygon are accepted."
        )
    mv_geom_gdf = gpd.GeoDataFrame(data={"geometry": [m]}, crs=f"EPSG:{srid}")

    return set(
        np.append(
            gpd.sjoin(
                geom_data, mv_geom_gdf, how="right", op="intersects"
            ).gid.unique(),
            edisgo_obj.topology.generators_df.weather_cell_id.dropna().unique(),
        )
    )


def get_directory_size(start_dir):
    """
    Calculates the size of all files within the start path.

    Walks through all files and sub-directories within a given directory and
    calculate the sum of size of all files in the directory.
    See also
    `stackoverflow <https://stackoverflow.com/questions/1392413/\
    calculating-a-directorys-size-using-python/1392549#1392549>`_.

    Parameters
    ----------
    start_dir : str
        Start path.

    Returns
    -------
    int
        Size of the directory.

    """
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(start_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def get_files_recursive(path, files=None):
    """
    Recursive function to get all files in a given path and its sub directories.

    Parameters
    ----------
    path : str
        Directory to start from.
    files : list, optional
        List of files to start with. Default: None.

    Returns
    -------

    """
    if files is None:
        files = []
    for f in os.listdir(path):
        file = os.path.join(path, f)
        if os.path.isdir(file):
            files = get_files_recursive(file, files)
        else:
            files.append(file)

    return files


def calculate_impedance_for_parallel_components(parallel_components, pu=False):
    """
    Method to calculate parallel impedance and power of parallel elements.
    """
    if pu:
        raise NotImplementedError(
            "Calculation in pu for parallel components not implemented yet."
        )
    else:
        if not (parallel_components.diff().dropna() < 1e-6).all().all():
            parallel_impedance = 1 / sum(
                1 / complex(comp.r, comp.x)
                for name, comp in parallel_components.iterrows()
            )
            # apply current devider and use minimum
            s_parallel = min(
                abs(
                    comp.s_nom
                    / (
                        1
                        / complex(comp.r, comp.x)
                        / sum(
                            1 / complex(comp.r, comp.x)
                            for name, comp in parallel_components.iterrows()
                        )
                    )
                )
                for name, comp in parallel_components.iterrows()
            )
            return pd.Series(
                {
                    "r": parallel_impedance.real,
                    "x": parallel_impedance.imag,
                    "s_nom": s_parallel,
                }
            )
        else:
            nr_components = len(parallel_components)
        return pd.Series(
            {
                "r": parallel_components.iloc[0].r / nr_components,
                "x": parallel_components.iloc[0].x / nr_components,
                "s_nom": parallel_components.iloc[0].s_nom * nr_components,
            }
        )


def add_line_susceptance(
    edisgo_obj,
    mode="mv_b",
):
    """
    Adds line susceptance information in Siemens to lines in existing grids.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object to which line susceptance information is added.

    mode : str
        Defines how the susceptance is added:

        * 'no_b'
            Susceptance is set to 0 for all lines.
        * 'mv_b' (Default)
            Susceptance is for the MV lines set according to the equipment parameters
            and for the LV lines it is set to zero.
        * 'all_b'
            Susceptance is for the MV lines set according to the equipment parameters
            and for the LV lines 0.25 uF/km is chosen.

    Returns
    -------
    :class:`~.EDisGo`

    """
    line_data_df = pd.concat(
        [
            edisgo_obj.topology.equipment_data["mv_overhead_lines"],
            edisgo_obj.topology.equipment_data["mv_cables"],
            edisgo_obj.topology.equipment_data["lv_cables"],
        ]
    )

    if mode == "no_b":
        line_data_df.loc[:, "C_per_km"] = 0
    elif mode == "mv_b":
        line_data_df.loc[
            edisgo_obj.topology.equipment_data["lv_cables"].index, "C_per_km"
        ] = 0
    elif mode == "all_b":
        line_data_df.loc[
            edisgo_obj.topology.equipment_data["lv_cables"].index, "C_per_km"
        ] = 0.25
    else:
        raise ValueError("Non-existing mode.")

    lines_df = edisgo_obj.topology.lines_df
    buses_df = edisgo_obj.topology.buses_df

    for index, bus0, type_info, length, num_parallel in lines_df[
        ["bus0", "type_info", "length", "num_parallel"]
    ].itertuples():
        v_nom = buses_df.loc[bus0].v_nom

        try:
            line_capacitance_per_km = (
                line_data_df.loc[line_data_df.U_n == v_nom].loc[type_info].C_per_km
            )
        except KeyError:
            line_capacitance_per_km = line_data_df.loc[type_info].C_per_km
            logger.warning(f"False voltage level for line {index}.")

        lines_df.loc[index, "b"] = calculate_line_susceptance(
            line_capacitance_per_km, length, num_parallel
        )

    return edisgo_obj


def mv_grid_gdf(edisgo_obj):
    return gpd.GeoDataFrame(
        geometry=[edisgo_obj.topology.grid_district["geom"]],
        crs=f"EPSG:{edisgo_obj.topology.grid_district['srid']}",
    )


def battery_storage_reference_operation(
    df,
    init_storage_charge,
    storage_max,
    charger_power,
    time_base,
    efficiency_charge=0.9,
    efficiency_discharge=0.9,
):
    """
    Reference operation of storage system where it directly charges
    Todo: Find original source
    Parameters
    -----------
    df : pandas.DataFrame
        Timeseries of house demand - PV generation
    init_storage_charge : float
        Initial state of energy of storage device
    storage_max : float
        Maximum energy level of storage device
    charger_power : float
        Nominal charging power of storage device
    time_base : float
        Timestep of inserted timeseries
    efficiency_charge: float
        Efficiency of storage system in case of charging
    efficiency_discharge: float
        Efficiency of storage system in case of discharging
    Returns
    ---------
    pandas.DataFrame
        Dataframe with storage operation timeseries
    """
    # Battery model handles generation positive, demand negative
    lst_storage_power = []
    lst_storage_charge = []
    storage_charge = init_storage_charge

    for i, d in df.iterrows():

        # If the house would feed electricity into the grid, charge the storage first.
        # No electricity exchange with grid as long as charger power is not exceeded
        if (d.house_demand > 0) & (storage_charge < storage_max):

            # Check if energy produced exceeds charger power
            if d.house_demand < charger_power:
                storage_charge = storage_charge + (
                    d.house_demand * efficiency_charge * time_base
                )
                storage_power = -d.house_demand
            # If it does, feed the rest to the grid
            else:
                storage_charge = storage_charge + (
                    charger_power * efficiency_charge * time_base
                )
                storage_power = -charger_power

            # If the storage would be overcharged, feed the 'rest' to the grid
            if storage_charge > storage_max:
                storage_power = storage_power + (storage_charge - storage_max) / (
                    efficiency_charge * time_base
                )
                storage_charge = storage_max

        # If the house needs electricity from the grid, discharge the storage first.
        # In this case d.house_demand is negative!
        # No electricity exchange with grid as long as demand does not exceed charger
        # power
        elif (d.house_demand < 0) & (storage_charge > 0):

            # Check if energy demand exceeds charger power
            if d.house_demand / efficiency_discharge < (charger_power * -1):
                storage_charge = storage_charge - (charger_power * time_base)
                storage_power = charger_power * efficiency_discharge

            else:
                storage_charge = storage_charge + (
                    d.house_demand / efficiency_discharge * time_base
                )
                storage_power = -d.house_demand

            # If the storage would be undercharged, take the 'rest' from the grid
            if storage_charge < 0:
                # since storage_charge is negative in this case it can be taken as
                # demand
                storage_power = (
                    storage_power + storage_charge * efficiency_discharge / time_base
                )
                storage_charge = 0

        # If the storage is full or empty, the demand is not affected
        # elif(storage_charge == 0) | (storage_charge == storage_max):
        else:
            storage_power = 0
        lst_storage_power.append(storage_power)
        lst_storage_charge.append(storage_charge)
    df["storage_power"] = lst_storage_power
    df["storage_charge"] = lst_storage_charge

    return df


def determine_observation_periods(edisgo_obj, window_days, idx="min", absolute=False):
    if absolute:
        residual_load = edisgo_obj.timeseries.residual_load.abs()
    else:
        residual_load = edisgo_obj.timeseries.residual_load
    residual_load = residual_load.rolling(window=window_days * 24, closed="both").mean()
    residual_load = residual_load.loc[::24]

    if idx == "min":
        timestep = residual_load.idxmin()
    elif idx == "max":
        timestep = residual_load.idxmax()
    elif idx == "load_max":
        timestep = edisgo_obj.timeseries.loads_active_power.sum(axis=1).idxmax()
    elif idx == "gen_max":
        timestep = edisgo_obj.timeseries.generators_active_power.sum(axis=1).idxmax()
    else:
        raise NotImplementedError

    timestep = timestep - pd.DateOffset(days=window_days)

    timeframe = pd.date_range(start=timestep, periods=window_days * 24, freq="h")

    return timeframe


def get_sample_using_time(
    edisgo_obj,
    start_date=None,
    periods=None,
    freq="1h",
    res_load=None,
    ts=True,
    bev=True,
    save_ev_soc_initial=True,
    hp=True,
    dsm=True,
):
    if periods is None:
        raise TypeError(
            "get_sample_using_time() missing required argument: " "'periods'"
        )
    if (res_load is None) & (start_date is None):
        raise TypeError(
            "get_sample_using_time() missing required argument:"
            "'start_date' or "
            "'res_load'"
        )
    elif start_date is not None:
        timeframe = pd.date_range(start=start_date, periods=periods, freq=freq)
    elif res_load is not None:
        if res_load == "balanced":
            timeframe = determine_observation_periods(
                edisgo_obj, int(periods / 24), idx="min", absolute=True
            )
        elif res_load in ["min", "max", "load_max", "gen_max"]:
            timeframe = determine_observation_periods(
                edisgo_obj, int(np.ceil(periods / 24)), idx=res_load
            )
        else:
            raise ValueError(
                "argument 'res_load' must be one of: 'min', 'max', 'load_max', "
                "'gen_max' or 'balanced'"
            )

    # generators, loads and storage units timeseries
    if ts:
        attributes = edisgo_obj.timeseries._attributes
        edisgo_obj.timeseries.timeindex = timeframe
        for attr in attributes:
            if not getattr(edisgo_obj.timeseries, attr).empty:
                setattr(
                    edisgo_obj.timeseries,
                    attr,
                    getattr(edisgo_obj.timeseries, attr).loc[timeframe],
                )
    # Battery electric vehicle timeseries
    if bev:
        if save_ev_soc_initial:
            # timestep EV SOC from timestep before if possible
            ts_before = pd.to_datetime(timeframe[0]) - pd.Timedelta(hours=1)
            try:
                initial_soc_cp = (
                    1
                    / 2
                    * (
                        edisgo_obj.electromobility.flexibility_bands[
                            "upper_energy"
                        ].loc[ts_before]
                        + edisgo_obj.electromobility.flexibility_bands[
                            "lower_energy"
                        ].loc[ts_before]
                    )
                )
            except KeyError:
                initial_soc_cp = (
                    1
                    / 2
                    * (
                        edisgo_obj.electromobility.flexibility_bands[
                            "upper_energy"
                        ].loc[start_date]
                        + edisgo_obj.electromobility.flexibility_bands[
                            "lower_energy"
                        ].loc[start_date]
                    )
                )
            edisgo_obj.electromobility.initial_soc_df = initial_soc_cp
        for key, df in edisgo_obj.electromobility.flexibility_bands.items():
            if not df.empty:
                df = df.loc[timeframe]
                edisgo_obj.electromobility.flexibility_bands.update({key: df})
    # Heat pumps timeseries
    if hp:
        for attr in ["cop_df", "heat_demand_df"]:
            if not getattr(edisgo_obj.heat_pump, attr).empty:
                setattr(
                    edisgo_obj.heat_pump,
                    attr,
                    getattr(edisgo_obj.heat_pump, attr).loc[timeframe],
                )
    # Demand Side Management timeseries
    if dsm:
        for attr in ["e_min", "e_max", "p_min", "p_max"]:
            if not getattr(edisgo_obj.dsm, attr).empty:
                setattr(
                    edisgo_obj.dsm,
                    attr,
                    getattr(edisgo_obj.dsm, attr).loc[timeframe],
                )


def resample(
    object, freq_orig, method: str = "ffill", freq: str | pd.Timedelta = "15min"
):
    """
    Resamples all time series data in given object to a desired resolution.

    Parameters
    ----------
    object : :class:`~.network.timeseries.TimeSeries`
        Object of which to resample time series data.
    freq_orig : :pandas:`pandas.Timedelta<Timedelta>`
        Frequency of original time series data.
    method : str, optional
        See `method` parameter in :attr:`~.EDisGo.resample_timeseries` for more
        information.
    freq : str, optional
        See `freq` parameter in :attr:`~.EDisGo.resample_timeseries` for more
        information.

    """

    # add time step at the end of the time series in case of up-sampling so that
    # last time interval in the original time series is still included
    df_dict = {}
    for attr in object._attributes:
        if not getattr(object, attr).empty:
            df_dict[attr] = getattr(object, attr)
            if pd.Timedelta(freq) < freq_orig:  # up-sampling
                new_dates = pd.DatetimeIndex([df_dict[attr].index[-1] + freq_orig])
            else:  # down-sampling
                new_dates = pd.DatetimeIndex([df_dict[attr].index[-1]])
            df_dict[attr] = (
                df_dict[attr]
                .reindex(df_dict[attr].index.union(new_dates).unique().sort_values())
                .ffill()
            )

    # resample time series
    if pd.Timedelta(freq) < freq_orig:  # up-sampling
        if method == "interpolate":
            for attr in df_dict.keys():
                setattr(
                    object,
                    attr,
                    df_dict[attr].resample(freq, closed="left").interpolate().iloc[:-1],
                )
        elif method == "ffill":
            for attr in df_dict.keys():
                setattr(
                    object,
                    attr,
                    df_dict[attr].resample(freq, closed="left").ffill().iloc[:-1],
                )
        elif method == "bfill":
            for attr in df_dict.keys():
                setattr(
                    object,
                    attr,
                    df_dict[attr].resample(freq, closed="left").bfill().iloc[:-1],
                )
        else:
            raise NotImplementedError(f"Resampling method {method} is not implemented.")
    else:  # down-sampling
        for attr in df_dict.keys():
            setattr(
                object,
                attr,
                df_dict[attr].resample(freq).mean(),
            )
