from __future__ import annotations

import logging
import os

from hashlib import md5
from math import pi, sqrt
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import saio

from sqlalchemy.engine.base import Engine

from edisgo.flex_opt import exceptions
from edisgo.io.db import session_scope_egon_data, sql_grid_geom, sql_intersects
from edisgo.tools import session_scope

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import climate

if TYPE_CHECKING:
    from edisgo import EDisGo

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


def drop_duplicated_indices(dataframe, keep="last"):
    """
    Drop rows of duplicate indices in dataframe.

    Be aware that this function changes the dataframe inplace. To avoid this behavior
    provide a copy of the dataframe to this function.

    Parameters
    ----------
    dataframe : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe to drop indices from.
    keep : str
        Indicator of whether to keep first ("first"), last ("last") or
        none (False) of the duplicated indices.
        See :pandas:`pandas.DataFrame.duplicated<DataFrame.duplicated>` for more
        information. Default: "last".

    """
    return dataframe[~dataframe.index.duplicated(keep=keep)]


def drop_duplicated_columns(df, keep="last"):
    """
    Drop columns of dataframe that appear more than once.

    Be aware that this function changes the dataframe inplace. To avoid this behavior
    provide a copy of the dataframe to this function.

    Parameters
    ----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe to drop columns from.
    keep : str
        Indicator of whether to keep first ("first"), last ("last") or
        none (False) of the duplicated columns.
        See :pandas:`pandas.DataFrame.duplicated<DataFrame.duplicated>` for more
        information. Default: "last".

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


def get_downstream_buses(edisgo_obj, comp_name, comp_type="bus"):
    """
    Returns all buses downstream (farther away from station) of the given bus or line.

    In case a bus is given, returns all buses downstream of the given bus plus the
    given bus itself.
    In case a line is given, returns all buses downstream of the bus that is closer to
    the station (thus only one bus of the line is included in the returned buses).

    Parameters
    ------------
    edisgo_obj : EDisGo object
    comp_name : str
        Name of bus or line (as in index of :attr:`~.network.topology.Topology.buses_df`
        or :attr:`~.network.topology.Topology.lines_df`) to get downstream buses for.
    comp_type : str
        Can be either 'bus' or 'line'. Default: 'bus'.

    Returns
    -------
    list(str)
        List of buses (as in index of :attr:`~.network.topology.Topology.buses_df`)
        downstream of the given component incl. the initial bus.

    """
    graph = edisgo_obj.topology.to_graph()
    station_node = edisgo_obj.topology.transformers_hvmv_df.bus1.values[0]

    if comp_type == "bus":
        # get upstream bus to determine which edge to remove to create subgraph
        bus = comp_name
        path_to_station = nx.shortest_path(graph, station_node, comp_name)
        bus_upstream = path_to_station[-2]
    elif comp_type == "line":
        # get bus further downstream to determine which buses downstream are affected
        bus0 = edisgo_obj.topology.lines_df.at[comp_name, "bus0"]
        bus1 = edisgo_obj.topology.lines_df.at[comp_name, "bus1"]
        path_to_station_bus0 = nx.shortest_path(graph, station_node, bus0)
        path_to_station_bus1 = nx.shortest_path(graph, station_node, bus1)
        bus = bus0 if len(path_to_station_bus0) > len(path_to_station_bus1) else bus1
        bus_upstream = bus0 if bus == bus1 else bus1
    else:
        raise ValueError(
            f"Component type needs to be either 'bus' or 'line'. Given {comp_type=} is "
            f"not valid."
        )

    # remove edge between bus and next bus upstream
    graph.remove_edge(bus, bus_upstream)

    # get subgraph containing relevant bus
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]

    for subgraph in subgraphs:
        if bus in subgraph.nodes():
            return list(subgraph.nodes())

    return [bus]


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


def determine_grid_integration_voltage_level(edisgo_object, power):
    """
    Gives voltage level component should be integrated into based on its nominal power.

    The voltage level is specified through an integer value from 4 to 7 with
    4 = MV busbar, 5 = MV grid, 6 = LV busbar and 7 = LV grid.

    The voltage level is determined using upper limits up to which capacity a component
    is integrated into a certain voltage level. These upper limits are set in the
    config section `grid_connection` through the parameters
    'upper_limit_voltage_level_{4:7}'.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    power : float
        Nominal power of component in MW.

    Returns
    --------
    int
        Voltage level component should be integrated into. Possible options are
        4 (MV busbar), 5 (MV grid), 6 (LV busbar) or 7 (LV grid).

    """
    cfg_max_p_nom = edisgo_object.config["grid_connection"]
    if (
        cfg_max_p_nom["upper_limit_voltage_level_5"]
        < power
        <= cfg_max_p_nom["upper_limit_voltage_level_4"]
    ):
        voltage_level = 4
    elif (
        cfg_max_p_nom["upper_limit_voltage_level_6"]
        < power
        <= cfg_max_p_nom["upper_limit_voltage_level_5"]
    ):
        voltage_level = 5
    elif (
        cfg_max_p_nom["upper_limit_voltage_level_7"]
        < power
        <= cfg_max_p_nom["upper_limit_voltage_level_6"]
    ):
        voltage_level = 6
    elif 0 < power <= cfg_max_p_nom["upper_limit_voltage_level_7"]:
        voltage_level = 7
    else:
        raise ValueError("Unsupported voltage level")
    return voltage_level


def determine_bus_voltage_level(edisgo_object, bus_name):
    """
    Gives voltage level as integer from 4 to 7 of given bus.

    The voltage level is specified through an integer value from 4 to 7 with
    4 = MV busbar, 5 = MV grid, 6 = LV busbar and 7 = LV grid.

    Buses that are directly connected to a station and not part of a longer feeder
    or half-ring, i.e. they are only part of one line, are as well considered as voltage
    level 4 or 6, depending on if they are connected to an HV/MV station or MV/LV
    station.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    bus_name : str
        Name of bus as in index of :attr:`~.network.topology.Topology.buses_df`.

    Returns
    --------
    int
        Voltage level of bus. Possible options are 4 (MV busbar), 5 (MV grid),
        6 (LV busbar) or 7 (LV grid).

    """
    v_nom = edisgo_object.topology.buses_df.at[bus_name, "v_nom"]
    if v_nom < 1:
        station_buses = edisgo_object.topology.transformers_df.bus1.values
        if bus_name in station_buses:
            voltage_level = 6
        else:
            # check if bus is directly connected to a station via a line that is not
            # connected to any other line - if that is the case it is considered as
            # voltage level 6
            connected_lines_df = edisgo_object.topology.get_connected_lines_from_bus(
                bus_name
            )
            if len(connected_lines_df) > 1:
                voltage_level = 7
            else:
                connected_line = connected_lines_df.iloc[0, :]
                if (
                    connected_line.at["bus0"] in station_buses
                    or connected_line.at["bus1"] in station_buses
                ):
                    voltage_level = 6
                else:
                    voltage_level = 7
    else:
        station_buses = edisgo_object.topology.transformers_hvmv_df.bus1.values
        if bus_name in station_buses:
            voltage_level = 4
        else:
            # check if bus is directly connected to a station via a line that is not
            # connected to any other line - if that is the case it is considered as
            # voltage level 4
            connected_lines_df = edisgo_object.topology.get_connected_lines_from_bus(
                bus_name
            )
            if len(connected_lines_df) > 1:
                voltage_level = 5
            else:
                connected_line = connected_lines_df.iloc[0, :]
                if (
                    connected_line.at["bus0"] in station_buses
                    or connected_line.at["bus1"] in station_buses
                ):
                    voltage_level = 4
                else:
                    voltage_level = 5
    return voltage_level


def get_weather_cells_intersecting_with_grid_district(
    edisgo_obj: EDisGo,
    engine: Engine | None = None,
) -> set:
    """
    Get all weather cells that intersect with the grid district.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine. Only needed when using new egon_data data.

    Returns
    -------
    set(int)
        Set with weather cell IDs.

    """
    # Download geometries of weather cells
    sql_geom = sql_grid_geom(edisgo_obj)
    srid = edisgo_obj.topology.grid_district["srid"]

    if edisgo_obj.legacy_grids is True:
        table = climate.Cosmoclmgrid
        with session_scope() as session:
            query = session.query(
                table.gid,
            ).filter(sql_intersects(table.geom, sql_geom, srid))
            weather_cells = pd.read_sql(sql=query.statement, con=query.session.bind).gid
    else:
        saio.register_schema("supply", engine)
        from saio.supply import egon_era5_weather_cells

        with session_scope_egon_data(engine=engine) as session:
            query = session.query(
                egon_era5_weather_cells.w_id,
            ).filter(sql_intersects(egon_era5_weather_cells.geom, sql_geom, srid))
            weather_cells = pd.read_sql(sql=query.statement, con=engine).w_id
    return set(
        np.append(
            weather_cells,
            edisgo_obj.topology.generators_df.weather_cell_id.dropna(),
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


def resample(
    object,
    freq_orig,
    method: str = "ffill",
    freq: str | pd.Timedelta = "15min",
    attr_to_resample=None,
):
    """
    Resamples time series data to a desired resolution.

    Both up- and down-sampling methods are possible.

    Parameters
    ----------
    object : :class:`~.network.timeseries.TimeSeries` or \
        :class:`~.network.heat.HeatPump`
        Object of which to resample time series data.
    freq_orig : :pandas:`pandas.Timedelta<Timedelta>`
        Frequency of original time series data.
    method : str, optional
        See `method` parameter in :attr:`~.EDisGo.resample_timeseries` for more
        information.
    freq : str, optional
        See `freq` parameter in :attr:`~.EDisGo.resample_timeseries` for more
        information.
    attr_to_resample : list(str), optional
        List of attributes to resample. Per default, all attributes specified in
        respective object's `_attributes` are resampled.

    """
    if attr_to_resample is None:
        attr_to_resample = object._attributes

    # add time step at the end of the time series in case of up-sampling so that
    # last time interval in the original time series is still included
    df_dict = {}
    for attr in attr_to_resample:
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


def reduce_memory_usage(df: pd.DataFrame, show_reduction: bool = False) -> pd.DataFrame:
    """
    Function to automatically check if columns of a pandas DataFrame can
    be reduced to a smaller data type.

    Source:
    https://www.mikulskibartosz.name/how-to-reduce-memory-usage-in-pandas/

    Parameters
    ----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame to reduce memory usage for.
    show_reduction : bool
        If True, print amount of memory reduced.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with decreased memory usage.

    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype("int16")
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype("int32")
                else:
                    df[col] = df[col].astype("int64")
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype("float32")
                else:
                    df[col] = df[col].astype("float64")

        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2

    if show_reduction is True:
        print(
            "Reduced memory usage of DataFrame by "
            f"{(1 - end_mem/start_mem) * 100:.2f} %."
        )

    return df


def get_year_based_on_timeindex(edisgo_obj):
    """
    Checks if :py:attr:`~.network.timeseries.TimeSeries.timeindex` is already set and
    if so, returns the year of the time index.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`

    Returns
    --------
    int or None
        If a time index is available returns the year of the time index,
        otherwise it returns None.

    """
    year = edisgo_obj.timeseries.timeindex.year
    if len(year) == 0:
        return None
    else:
        return year[0]


def get_year_based_on_scenario(scenario):
    """
    Returns the year the given scenario was set up for.

    Parameters
    ----------
    scenario : str
        Scenario for which to set year. Possible options are 'eGon2035' and 'eGon100RE'.

    Returns
    --------
    int or None
        Returns the year of the scenario (2035 in case of the 'eGon2035' scenario
        and 2045 in case of the 'eGon100RE' scenario). If another scenario name is
        provided it returns None.

    """
    if scenario == "eGon2035":
        return 2035
    elif scenario == "eGon100RE":
        return 2045
    else:
        return None


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Get hash of dataframe.

    Can be used to check if dataframes have the same content.

    Parameters
    -----------
    df : :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame to hash.

    Returns
    --------
    str
        Hash of dataframe as string.

    """
    s = df.to_json()
    return md5(s.encode()).hexdigest()
