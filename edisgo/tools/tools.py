import os

from math import pi, sqrt

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from sqlalchemy import func

from edisgo.flex_opt import check_tech_constraints, exceptions
from edisgo.network.grids import LVGrid
from edisgo.tools import session_scope

if "READTHEDOCS" not in os.environ:

    from egoio.db_tables import climate
    from shapely.geometry.multipolygon import MultiPolygon
    from shapely.wkt import loads as wkt_loads


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
                    lvgrid = LVGrid(id=int(node.split("_")[-2]), edisgo_obj=edisgo_obj)
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
            lv_grid_repr = "LVGrid_{}".format(int(bus.split("_")[-2]))
            if lv_grid_repr in edisgo_obj.topology._grids.keys():
                lvgrid = edisgo_obj.topology._grids[lv_grid_repr]
                lv_graph = lvgrid.graph
                lv_station = lvgrid.station.index[0]
                for bus in lvgrid.buses_df.index:
                    lv_path = nx.shortest_path(lv_graph, source=lv_station, target=bus)
                    edisgo_obj.topology.buses_df.at[
                        bus, "path_length_to_station"
                    ] = len(path) + len(lv_path)
    return edisgo_obj.topology.buses_df.path_length_to_station


def assign_voltage_level_to_component(edisgo_obj, df):
    """
    Adds column with specification of voltage level component is in.

    The voltage level ('mv' or 'lv') is determined based on the nominal
    voltage of the bus the component is connected to. If the nominal voltage
    is smaller than 1 kV, voltage level 'lv' is assigned, otherwise 'mv' is
    assigned.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names in the index. Only required column is
        column 'bus', giving the name of the bus the component is connected to.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Same dataframe as given in parameter `df` with new column
        'voltage_level' specifying the voltage level the component is in
        (either 'mv' or 'lv').

    """
    df["voltage_level"] = df.apply(
        lambda _: "lv" if edisgo_obj.topology.buses_df.at[_.bus, "v_nom"] < 1 else "mv",
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
    mv_geom_gdf = gpd.GeoDataFrame(m, crs=f"EPSG:{srid}", columns=["geometry"])

    return set(
        np.append(
            gpd.sjoin(
                geom_data, mv_geom_gdf, how="right", op="intersects"
            ).gid.unique(),
            edisgo_obj.topology.generators_df.weather_cell_id.dropna().unique(),
        )
    )


def aggregate_components(
    edisgo_obj,
    mode="by_component_type",
    aggregate_generators_by_cols=["bus"],
    aggregate_loads_by_cols=["bus"],
    aggregate_charging_points_by_cols=["bus"],
):
    """
    Aggregates generators, loads and charging points at the same bus.

    There are several options how to aggregate. By default all components
    of the same type are aggregated separately. You can specify further
    columns to consider in the aggregation, such as the generator type
    or the load sector.

    Be aware that by aggregating components you lose some information
    e.g. on load sector or charging point use case.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
    mode : str
        Valid options are 'by_component_type' and 'by_load_and_generation'.
        In case of aggregation 'by_component_type' generators, loads and
        charging points are aggregated separately, by the respectively
        specified columns, given in `aggregate_generators_by_cols`,
        `aggregate_loads_by_cols`, and `aggregate_charging_points_by_cols`.
        In case of aggregation 'by_load_and_generation', all loads and
        charging points at the same bus are aggregated. Input in
        `aggregate_loads_by_cols` and `aggregate_charging_points_by_cols`
        is ignored. Generators are aggregated by the columns specified in
        `aggregate_generators_by_cols`.
    aggregate_generators_by_cols : list(str)
        List of columns to aggregate generators at the same bus by. Valid
        columns are all columns in
        :attr:`~.network.topology.Topology.generators_df`.
    aggregate_loads_by_cols : list(str)
        List of columns to aggregate loads at the same bus by. Valid
        columns are all columns in
        :attr:`~.network.topology.Topology.loads_df`.
    aggregate_charging_points_by_cols : list(str)
        List of columns to aggregate charging points at the same bus by.
        Valid columns are all columns in
        :attr:`~.network.topology.Topology.charging_points_df`.

    """
    # aggregate generators at the same bus
    if mode not in ["by_component_type", "by_load_and_generation"]:
        raise ValueError(
            f"The given mode {mode} is not valid. Please see the docstring for more "
            f"information."
        )

    if not edisgo_obj.topology.generators_df.empty:
        gens_groupby = edisgo_obj.topology.generators_df.groupby(
            aggregate_generators_by_cols
        )
        naming = "Generators_{}"

        # set up new generators_df
        gens_df_grouped = gens_groupby.sum().reset_index()
        gens_df_grouped["name"] = gens_df_grouped.apply(
            lambda _: naming.format("_".join(_.loc[aggregate_generators_by_cols])),
            axis=1,
        )
        gens_df_grouped["control"] = "PQ"
        gens_df_grouped["control"] = "misc"
        if "weather_cell_id" in gens_df_grouped.columns:
            gens_df_grouped.drop(columns=["weather_cell_id"], inplace=True)
        edisgo_obj.topology.generators_df = gens_df_grouped.set_index("name")

        # set up new generator time series
        groups = gens_groupby.groups
        if isinstance(list(groups.keys())[0], tuple):
            edisgo_obj.timeseries.generators_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                "_".join(k)
                            ): edisgo_obj.timeseries.generators_active_power.loc[
                                :, v
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.generators_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                "_".join(k)
                            ): edisgo_obj.timeseries.generators_reactive_power.loc[
                                :, v
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
        else:
            edisgo_obj.timeseries.generators_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                k
                            ): edisgo_obj.timeseries.generators_active_power.loc[
                                :, v
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.generators_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                k
                            ): edisgo_obj.timeseries.generators_reactive_power.loc[
                                :, v
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )

    # aggregate all loads (conventional loads and charging points) at the
    # same bus
    if mode == "by_load_and_generation":
        aggregate_loads_by_cols = ["bus"]
        loads_groupby = edisgo_obj.topology.loads_df.loc[:, ["bus", "p_nom"]].groupby(
            aggregate_loads_by_cols
        )

        naming = "Loads_{}"
        # set up new loads_df
        loads_df_grouped = loads_groupby.sum().reset_index()
        loads_df_grouped["name"] = loads_df_grouped.apply(
            lambda _: naming.format("_".join(_.loc[aggregate_loads_by_cols])),
            axis=1,
        )

        loads_df_grouped = loads_df_grouped.assign(type="load")

        edisgo_obj.topology.loads_df = loads_df_grouped.set_index("name")

        # set up new loads time series
        groups = loads_groupby.groups
        ts_active = pd.concat(
            [
                edisgo_obj.timeseries.loads_active_power,
                edisgo_obj.timeseries.charging_points_active_power,
            ],
            axis=1,
        )
        ts_reactive = pd.concat(
            [
                edisgo_obj.timeseries.loads_reactive_power,
                edisgo_obj.timeseries.charging_points_reactive_power,
            ],
            axis=1,
        )
        if isinstance(list(groups.keys())[0], tuple):

            edisgo_obj.timeseries.loads_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {naming.format("_".join(k)): ts_active.loc[:, v].sum(axis=1)}
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.loads_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {naming.format("_".join(k)): ts_reactive.loc[:, v].sum(axis=1)}
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
        else:
            edisgo_obj.timeseries.loads_active_power = pd.concat(
                [
                    pd.DataFrame({naming.format(k): ts_active.loc[:, v].sum(axis=1)})
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.loads_reactive_power = pd.concat(
                [
                    pd.DataFrame({naming.format(k): ts_reactive.loc[:, v].sum(axis=1)})
                    for k, v in groups.items()
                ],
                axis=1,
            )
        # overwrite charging points
        edisgo_obj.timeseries.charging_points_active_power = pd.DataFrame(
            index=edisgo_obj.timeseries.timeindex
        )
        edisgo_obj.timeseries.charging_points_reactive_power = pd.DataFrame(
            index=edisgo_obj.timeseries.timeindex
        )

        return

    # aggregate conventional loads at the same bus and charging points
    # at the same bus separately

    # conventional loads
    if not edisgo_obj.topology.loads_df.empty:
        loads_df = edisgo_obj.topology.loads_df.loc[
            edisgo_obj.topology.loads_df.type.isin(["load", ""])
        ]
        loads_groupby = loads_df.groupby(aggregate_loads_by_cols)
        naming = "Loads_{}"

        # set up new loads_df
        loads_df_grouped = loads_groupby.sum().reset_index()
        loads_df_grouped["name"] = loads_df_grouped.apply(
            lambda _: naming.format("_".join(_.loc[aggregate_loads_by_cols])),
            axis=1,
        )

        loads_df_grouped = loads_df_grouped.assign(type="load")

        edisgo_obj.topology.loads_df.drop(index=loads_df.index, inplace=True)

        edisgo_obj.topology.loads_df = edisgo_obj.topology.loads_df.append(
            loads_df_grouped.set_index("name")
        )

        # set up new loads time series
        groups = loads_groupby.groups

        if isinstance(list(groups.keys())[0], tuple):
            edisgo_obj.timeseries.loads_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                "_".join(k)
                            ): edisgo_obj.timeseries.loads_active_power.loc[:, v].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.loads_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                "_".join(k)
                            ): edisgo_obj.timeseries.loads_reactive_power.loc[:, v].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
        else:
            edisgo_obj.timeseries.loads_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                k
                            ): edisgo_obj.timeseries.loads_active_power.loc[:, v].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.loads_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                k
                            ): edisgo_obj.timeseries.loads_reactive_power.loc[:, v].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )

    # charging points
    if not edisgo_obj.topology.charging_points_df.empty:
        loads_groupby = edisgo_obj.topology.charging_points_df.groupby(
            aggregate_charging_points_by_cols
        )
        naming = "ChargingPoints_{}"

        # set up new charging_points_df
        loads_df_grouped = loads_groupby.sum().reset_index()
        loads_df_grouped["name"] = loads_df_grouped.apply(
            lambda _: naming.format("_".join(_.loc[aggregate_charging_points_by_cols])),
            axis=1,
        )

        loads_df_grouped = loads_df_grouped.assign(type="charging_point")

        edisgo_obj.topology.loads_df.drop(
            index=edisgo_obj.topology.charging_points_df.index, inplace=True
        )

        edisgo_obj.topology.loads_df = edisgo_obj.topology.loads_df.append(
            loads_df_grouped.set_index("name")
        )

        # set up new charging points time series
        groups = loads_groupby.groups

        if isinstance(list(groups.keys())[0], tuple):
            edisgo_obj.timeseries.charging_points_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                "_".join(k)
                            ): edisgo_obj.timeseries.charging_points_active_power.loc[
                                :, v
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.charging_points_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                "_".join(k)
                            ): edisgo_obj.timeseries.charging_points_reactive_power.loc[
                                :,
                                v,
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
        else:
            edisgo_obj.timeseries.charging_points_active_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                k
                            ): edisgo_obj.timeseries.charging_points_active_power.loc[
                                :, v
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
            edisgo_obj.timeseries.charging_points_reactive_power = pd.concat(
                [
                    pd.DataFrame(
                        {
                            naming.format(
                                k
                            ): edisgo_obj.timeseries.charging_points_reactive_power.loc[
                                :,
                                v,
                            ].sum(
                                axis=1
                            )
                        }
                    )
                    for k, v in groups.items()
                ],
                axis=1,
            )
