from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from networkx.algorithms.shortest_paths.weighted import (
    _dijkstra as dijkstra_shortest_path_length,
)

from edisgo.network.grids import LVGrid, MVGrid
from edisgo.tools.tools import get_downstream_buses

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def reinforce_mv_lv_station_overloading(edisgo_obj, critical_stations):
    """
    Reinforce MV/LV substations due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV/LV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1_station': ['transformer_reinforced_1',
                                      ...,
                                      'transformer_reinforced_x'],
                   'Grid_10_station': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1_station': ['transformer_1']}
        }

    """
    transformers_changes = _reinforce_station_overloading(
        edisgo_obj, critical_stations, voltage_level="lv"
    )

    if transformers_changes["added"]:
        logger.debug(
            "==> {} LV station(s) has/have been reinforced due to "
            "overloading issues.".format(str(len(transformers_changes["added"])))
        )

    return transformers_changes


def reinforce_hv_mv_station_overloading(edisgo_obj, critical_stations):
    """
    Reinforce HV/MV station due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded HV/MV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.

    Returns
    -------
    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1_station': ['transformer_reinforced_1',
                                      ...,
                                      'transformer_reinforced_x'],
                   'Grid_10_station': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1_station': ['transformer_1']}
        }

    """
    transformers_changes = _reinforce_station_overloading(
        edisgo_obj, critical_stations, voltage_level="mv"
    )

    if transformers_changes["added"]:
        logger.debug("==> MV station has been reinforced due to overloading issues.")

    return transformers_changes


def _reinforce_station_overloading(edisgo_obj, critical_stations, voltage_level):
    """
    Reinforce stations due to overloading issues.

    In a first step a parallel transformer of the same kind is installed.
    If this is not sufficient as many standard transformers as needed are
    installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded MV/LV stations, their missing apparent
        power at maximal over-loading and the corresponding time step.
        Index of the dataframe are the representatives of the grids with
        over-loaded stations. Columns are 's_missing' containing the missing
        apparent power at maximal over-loading in MVA as float and 'time_index'
        containing the corresponding time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`.
    voltage_level : str
        Voltage level, over-loading is handled for. Possible options are
        "mv" or "lv".

    Returns
    -------
    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1_station': ['transformer_reinforced_1',
                                      ...,
                                      'transformer_reinforced_x'],
                   'Grid_10_station': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1_station': ['transformer_1']}
        }

    """
    if voltage_level == "lv":
        try:
            standard_transformer = edisgo_obj.topology.equipment_data[
                "lv_transformers"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "mv_lv_transformer"
                ]
            ]
        except KeyError:
            raise KeyError("Standard MV/LV transformer is not in equipment list.")
    elif voltage_level == "mv":
        try:
            standard_transformer = edisgo_obj.topology.equipment_data[
                "mv_transformers"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "hv_mv_transformer"
                ]
            ]
        except KeyError:
            raise KeyError("Standard HV/MV transformer is not in equipment list.")
    else:
        raise ValueError(
            "{} is not a valid option for input variable 'voltage_level' in "
            "function _station_overloading. Try 'mv' or "
            "'lv'.".format(voltage_level)
        )

    transformers_changes = {"added": {}, "removed": {}}
    for station in critical_stations.index:
        grid = critical_stations.at[station, "grid"]
        # list of maximum power of each transformer in the station
        s_max_per_trafo = grid.transformers_df.s_nom
        # missing capacity
        s_trafo_missing = critical_stations.at[station, "s_missing"]

        # check if second transformer of the same kind is sufficient
        # if true install second transformer, otherwise install as many
        # standard transformers as needed
        if max(s_max_per_trafo) >= s_trafo_missing:
            # if station has more than one transformer install a new
            # transformer of the same kind as the transformer that best
            # meets the missing power demand
            new_transformers = grid.transformers_df.loc[
                [
                    grid.transformers_df[s_max_per_trafo >= s_trafo_missing][
                        "s_nom"
                    ].idxmin()
                ]
            ]
            name = new_transformers.index[0].split("_")
            name.insert(-1, "reinforced")
            name[-1] = len(grid.transformers_df) + 1
            new_transformers.index = ["_".join([str(_) for _ in name])]

            # add new transformer to list of added transformers
            transformers_changes["added"][station] = [new_transformers.index[0]]
        else:
            # get any transformer to get attributes for new transformer from
            duplicated_transformer = grid.transformers_df.iloc[[0]]
            name = duplicated_transformer.index[0].split("_")
            name.insert(-1, "reinforced")
            duplicated_transformer.s_nom = standard_transformer.S_nom
            duplicated_transformer.type_info = standard_transformer.name
            if voltage_level == "lv":
                duplicated_transformer.r_pu = standard_transformer.r_pu
                duplicated_transformer.x_pu = standard_transformer.x_pu

            # set up as many new transformers as needed
            number_transformers = math.ceil(
                (s_trafo_missing + s_max_per_trafo.sum()) / standard_transformer.S_nom
            )

            index = []

            for i in range(number_transformers):
                name[-1] = i + 1
                index.append("_".join([str(_) for _ in name]))

            if number_transformers > 1:
                new_transformers = duplicated_transformer.iloc[
                    np.arange(len(duplicated_transformer)).repeat(number_transformers)
                ]
            else:
                new_transformers = duplicated_transformer.copy()

            new_transformers.index = index

            # add new transformer to list of added transformers
            transformers_changes["added"][station] = new_transformers.index.values
            # add previous transformers to list of removed transformers
            transformers_changes["removed"][station] = grid.transformers_df.index.values
            # remove previous transformers from topology
            if voltage_level == "lv":
                edisgo_obj.topology.transformers_df.drop(
                    grid.transformers_df.index.values, inplace=True
                )
            else:
                edisgo_obj.topology.transformers_hvmv_df.drop(
                    grid.transformers_df.index.values, inplace=True
                )

        # add new transformers to topology
        if voltage_level == "lv":
            edisgo_obj.topology.transformers_df = pd.concat(
                [
                    edisgo_obj.topology.transformers_df,
                    new_transformers,
                ]
            )
        else:
            edisgo_obj.topology.transformers_hvmv_df = pd.concat(
                [
                    edisgo_obj.topology.transformers_hvmv_df,
                    new_transformers,
                ]
            )
    return transformers_changes


def reinforce_mv_lv_station_voltage_issues(edisgo_obj, critical_stations):
    """
    Reinforce MV/LV substations due to voltage issues.

    A parallel standard transformer is installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with maximum deviations from allowed lower or upper voltage limits
        in p.u. for all MV-LV stations with voltage issues. For more information on
        dataframe see :attr:`~.flex_opt.check_tech_constraints.voltage_issues`.

    Returns
    -------
    :obj:`dict`
        Dictionary with added transformers in the form::

            {'added': {'Grid_1': ['transformer_reinforced_1',
                                  ...,
                                  'transformer_reinforced_x'],
                       'Grid_10': ['transformer_reinforced_10']
                       }
            }

    """

    # get parameters for standard transformer
    try:
        standard_transformer = edisgo_obj.topology.equipment_data[
            "lv_transformers"
        ].loc[
            edisgo_obj.config["grid_expansion_standard_equipment"]["mv_lv_transformer"]
        ]
    except KeyError:
        raise KeyError("Standard MV/LV transformer is not in equipment list.")

    transformers_changes = {"added": {}}
    for station in critical_stations.index:
        grid_id = critical_stations.at[station, "lv_grid_id"]
        grid = edisgo_obj.topology.get_lv_grid(int(grid_id))
        # get any transformer to get attributes for new transformer from
        duplicated_transformer = grid.transformers_df.iloc[[0]]
        # change transformer parameters
        name = duplicated_transformer.index[0].split("_")
        name.insert(-1, "reinforced")
        name[-1] = len(grid.transformers_df) + 1
        duplicated_transformer.index = ["_".join([str(_) for _ in name])]
        duplicated_transformer.s_nom = standard_transformer.S_nom
        duplicated_transformer.r_pu = standard_transformer.r_pu
        duplicated_transformer.x_pu = standard_transformer.x_pu
        duplicated_transformer.type_info = standard_transformer.name
        # add new transformer to topology
        edisgo_obj.topology.transformers_df = pd.concat(
            [
                edisgo_obj.topology.transformers_df,
                duplicated_transformer,
            ]
        )
        transformers_changes["added"][str(grid)] = duplicated_transformer.index.tolist()

    if transformers_changes["added"]:
        logger.debug(
            "==> {} LV station(s) has/have been reinforced due to voltage "
            "issues.".format(len(transformers_changes["added"]))
        )

    return transformers_changes


def get_standard_line(edisgo_obj, grid=None, nominal_voltage=None):
    """
    Get standard line type for given voltage level from config.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    nominal_voltage : float
        Nominal voltage of grid level to obtain standard line type for. Can be
        0.4, 10 or 20 kV.

    Returns
    ---------
    str
        Name of standard line, e.g. "NAYY 4x1x150".

    """
    if grid is not None:
        if isinstance(grid, LVGrid):
            nominal_voltage = 0.4
        elif isinstance(grid, MVGrid):
            nominal_voltage = grid.buses_df.v_nom.values[0]
        else:
            raise ValueError("Inserted grid is invalid.")
    if nominal_voltage == 0.4:
        standard_line_type = edisgo_obj.config["grid_expansion_standard_equipment"][
            "lv_line"
        ]
    else:
        standard_line_type = edisgo_obj.config["grid_expansion_standard_equipment"][
            f"mv_line_{int(nominal_voltage)}kv"
        ]
    return standard_line_type


def split_feeder_at_given_length(
    edisgo_obj, grid, feeder_name, crit_nodes_in_feeder, disconnect_length=2 / 3
):
    """
    Splits given feeder at specified length.

    This is a standard grid expansion measure in case of voltage issues. There, the
    feeder is usually disconnected at 2/3 of the feeder length.

    The feeder is split at 2/3 of the length between the station and the critical node
    farthest away from the station. A new standard line is installed, or if the line is
    already connected to the grid's station exchanged by standard line or a
    parallel standard line installed.
    In LV grids, feeder can only be split outside of buildings, i.e. loads and
    generators in buildings cannot be directly connected to the MV/LV station.
    In MV grids feeder can only be split at LV stations because they
    have switch disconnectors needed to operate the lines as half rings (loads
    in MV would be suitable as well because they have a switch bay (Schaltfeld)
    but this is currently not implemented).

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    feeder_name : str
        The feeder name corresponds to the name of the neighboring
        node of the respective grid's station.
    crit_nodes_in_feeder : list(str)
        List with names of buses that have voltage issues or should be considered
        when finding the point in the feeder where to split it. This is needed
        in order to find the critical node farthest away from the station.
    disconnect_length : float
        Relative length at which the feeder should be split. Default: 2/3.

    Returns
    -------
    dict{str: float}
        Dictionary with name of lines at which feeder was split as keys and the
        corresponding number of lines added as values.

    """
    standard_line = get_standard_line(edisgo_obj, grid=grid)
    lines_changes = {}

    # find path to each node in order to find node with voltage issues farthest
    # away from station
    graph = grid.graph
    station_node = grid.station.index[0]
    paths = {}
    for node in crit_nodes_in_feeder:
        path = nx.shortest_path(graph, station_node, node)
        paths[node] = path
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logger.error(
                "Voltage issues at busbar in LV network {} should have "
                "been solved in previous steps.".format(grid)
            )

    # find node farthest away
    get_weight = lambda u, v, data: data["length"]  # noqa: E731
    path_length = 0
    for n in crit_nodes_in_feeder:
        path_length_dict_tmp = dijkstra_shortest_path_length(
            graph, grid.station.index[0], get_weight, target=n
        )
        if path_length_dict_tmp[n] > path_length:
            node = n
            path_length = path_length_dict_tmp[n]
            path_length_dict = path_length_dict_tmp
    path = paths[node]

    # find first node in path that exceeds given length of the line length
    # from station to critical node farthest away from the station where feeder should
    # be separated
    disconnect_node = next(
        j
        for j in path
        if path_length_dict[j] >= path_length_dict[node] * disconnect_length
    )

    # if LVGrid: check if disconnect_node is outside of a house
    # and if not find next BranchTee outside the house
    if isinstance(grid, LVGrid):
        while (
            ~np.isnan(grid.buses_df.loc[disconnect_node].in_building)
            and grid.buses_df.loc[disconnect_node].in_building
        ):
            disconnect_node = path[path.index(disconnect_node) - 1]
            # break if node is station
            if disconnect_node is path[0]:
                logger.error("Could not reinforce voltage issue.")
                break

    # if MVGrid: check if disconnect_node is LV station and if not find
    # next LV station
    else:
        while disconnect_node not in edisgo_obj.topology.transformers_df.bus0.values:
            try:
                # try to find LVStation behind disconnect_node
                disconnect_node = path[path.index(disconnect_node) + 1]
            except IndexError:
                # if no LVStation between disconnect_node and node with
                # voltage problem, connect node directly to
                # MVStation
                disconnect_node = node
                break

    # if disconnect_node is a representative (meaning it is already
    # directly connected to the station), line cannot be
    # disconnected and must therefore be reinforced
    if disconnect_node == feeder_name:
        crit_line_name = graph.get_edge_data(station_node, disconnect_node)[
            "branch_name"
        ]
        crit_line = grid.lines_df.loc[crit_line_name]

        # if critical line is already a standard line install one
        # more parallel line
        if crit_line.type_info == standard_line:
            edisgo_obj.topology.update_number_of_parallel_lines(
                pd.Series(
                    index=[crit_line_name],
                    data=[
                        edisgo_obj.topology._lines_df.at[crit_line_name, "num_parallel"]
                        + 1
                    ],
                )
            )
            lines_changes[crit_line_name] = 1

        # if critical line is not yet a standard line replace old
        # line by a standard line
        else:
            # number of parallel standard lines could be calculated
            # following [2] p.103; for now number of parallel
            # standard lines is iterated
            edisgo_obj.topology.change_line_type([crit_line_name], standard_line)
            lines_changes[crit_line_name] = 1
        logger.debug(
            f"When solving voltage issues in grid {grid.id} in feeder "
            f"{feeder_name}, disconnection at 2/3 was tried but bus is already "
            f"connected to the station, wherefore line {crit_line_name} was "
            f"reinforced."
        )

    # if disconnect_node is not a representative, disconnect line
    else:
        # get line between disconnect_node and predecessor node (that is
        # closer to the station)
        pred_node = path[path.index(disconnect_node) - 1]
        crit_line_name = graph.get_edge_data(disconnect_node, pred_node)["branch_name"]
        if grid.lines_df.at[crit_line_name, "bus0"] == pred_node:
            edisgo_obj.topology._lines_df.at[crit_line_name, "bus0"] = station_node
        elif grid.lines_df.at[crit_line_name, "bus1"] == pred_node:
            edisgo_obj.topology._lines_df.at[crit_line_name, "bus1"] = station_node
        else:
            raise ValueError("Bus not in line buses. Please check.")
        # change line length and type
        edisgo_obj.topology._lines_df.at[crit_line_name, "length"] = path_length_dict[
            disconnect_node
        ]
        edisgo_obj.topology.change_line_type([crit_line_name], standard_line)
        lines_changes[crit_line_name] = 1
        # TODO: Include switch disconnector
        logger.debug(
            f"Feeder {feeder_name} in grid {grid.id} was split at "
            f"line {crit_line_name}."
        )
    return lines_changes


def reinforce_lines_voltage_issues(edisgo_obj, grid, crit_nodes):
    """
    Reinforce lines in MV and LV topology due to voltage issues.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_nodes : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with maximum deviations from allowed lower or upper voltage limits
        in p.u. for all buses in specified grid. For more information on dataframe see
        :attr:`~.flex_opt.check_tech_constraints.voltage_issues`.

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.

    Notes
    -----
    Reinforce measures:

    1. For LV only, exchange all cables in feeder by standard cable if smaller cable is
    currently used.
    2. Split feeder at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line, or if the line is
    already connected to the grid's station exchange by standard line or install
    parallel standard line. See function :attr:`split_feeder_at_given_length` for more
    information.

    """
    # load standard line data
    standard_line = get_standard_line(edisgo_obj, grid=grid)

    # get feeders with voltage issues
    grid.assign_grid_feeder()
    crit_buses_df = grid.buses_df.loc[crit_nodes.index, :]
    crit_feeders = crit_buses_df.grid_feeder.unique()

    # per default, measure to disconnect at two-thirds is set to True and only if cables
    # in grid are exchanged by standard lines it is set to False, to recheck voltage
    disconnect_2_3 = True

    lines_changes = {}
    for repr_node in crit_feeders:
        if isinstance(grid, LVGrid):
            lines_in_feeder = grid.lines_df[grid.lines_df.grid_feeder == repr_node]
            # check if line type is any of the following
            small_cables = ["NAYY 4x1x120", "NAYY 4x1x95", "NAYY 4x1x50", "NAYY 4x1x35"]
            small_lines_in_feeder = lines_in_feeder[
                lines_in_feeder.type_info.isin(small_cables)
            ]
            # filter cables connecting houses (their type is kept)
            # ToDo Currently new components can be connected to house connection via
            #  a new cable, wherefore it is checked, whether the house connecting cable
            #  is an end cable. Needs to be changed once grid connection is changed.
            for line in small_lines_in_feeder.index:
                lines_bus0 = edisgo_obj.topology.get_connected_lines_from_bus(
                    small_lines_in_feeder.at[line, "bus0"]
                )
                lines_bus1 = edisgo_obj.topology.get_connected_lines_from_bus(
                    small_lines_in_feeder.at[line, "bus1"]
                )
                if len(lines_bus0) == 1 or len(lines_bus1) == 1:
                    small_lines_in_feeder.drop(index=line, inplace=True)
            # if there are small lines, exchange them
            if len(small_lines_in_feeder) > 0:
                edisgo_obj.topology.change_line_type(
                    small_lines_in_feeder.index, standard_line
                )
                # check if s_nom before is larger than when using standard cable
                # and if so, install parallel cable
                lines_lower_snom = small_lines_in_feeder[
                    small_lines_in_feeder.s_nom
                    > grid.lines_df.loc[small_lines_in_feeder.index, "s_nom"]
                ]
                if len(lines_lower_snom) > 0:
                    number_parallel_lines = np.ceil(
                        lines_lower_snom.s_nom
                        / grid.lines_df.loc[lines_lower_snom.index, "s_nom"]
                    )
                    # update number of parallel lines
                    edisgo_obj.topology.update_number_of_parallel_lines(
                        number_parallel_lines
                    )
                # add to lines changes
                update_dict = {
                    _: grid.lines_df.at[_, "num_parallel"]
                    for _ in small_lines_in_feeder.index
                }
                lines_changes.update(update_dict)
                logger.debug(
                    f"When solving voltage issues in LV grid {grid.id} in feeder "
                    f"{repr_node}, {len(small_lines_in_feeder)} were exchanged by "
                    f"standard lines."
                )
                # if any cable was changed, set disconnect_2_3 to False
                disconnect_2_3 = False

        if disconnect_2_3 is True:
            lines_changes_tmp = split_feeder_at_given_length(
                edisgo_obj,
                grid,
                feeder_name=repr_node,
                crit_nodes_in_feeder=crit_buses_df[
                    crit_buses_df.grid_feeder == repr_node
                ].index,
                disconnect_length=2 / 3,
            )
            logger.debug(
                f"When solving voltage issues in grid {grid.id} in feeder "
                f"{repr_node}, disconnection at 2/3 was conducted "
                f"(line {list(lines_changes_tmp.keys())[0]})."
            )
            lines_changes.update(lines_changes_tmp)

    if not lines_changes:
        logger.debug(
            "==> {} line(s) was/were reinforced due to voltage "
            "issues.".format(len(lines_changes))
        )

    return lines_changes


def reinforce_lines_overloading(edisgo_obj, crit_lines):
    """
    Reinforce lines in MV and LV topology due to overloading.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    crit_lines : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.

    Notes
    -----
    Reinforce measures:

    1. Install parallel line of the same type as the existing line (Only if
       line is a cable, not an overhead line. Otherwise a standard equipment
       cable is installed right away.)
    2. Remove old line and install as many parallel standard lines as
       needed.

    """

    lines_changes = {}
    # reinforce mv lines
    lines_changes.update(
        _reinforce_lines_overloading_per_grid_level(edisgo_obj, "mv", crit_lines)
    )
    # reinforce lv lines
    lines_changes.update(
        _reinforce_lines_overloading_per_grid_level(edisgo_obj, "lv", crit_lines)
    )

    if not crit_lines.empty:
        logger.debug(
            "==> {} line(s) was/were reinforced due to over-loading "
            "issues.".format(crit_lines.shape[0])
        )

    return lines_changes


def _reinforce_lines_overloading_per_grid_level(edisgo_obj, voltage_level, crit_lines):
    """
    Reinforce lines in MV or LV topology due to overloading.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    voltage_level : str
        Voltage level, over-loading is handled for. Possible options are
        "mv" or "lv".
    crit_lines : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.

    """

    def _add_parallel_standard_lines(lines):
        """
        Adds as many parallel standard lines as needed so solve overloading.

        Adds number of added lines to `lines_changes` dictionary.

        Parameters
        ----------
        lines : list(str)
            List of line names to add parallel standard lines for.

        """
        # calculate necessary number of parallel lines
        number_parallel_lines = np.ceil(
            crit_lines.max_rel_overload[lines]
            * edisgo_obj.topology.lines_df.loc[lines, "num_parallel"]
        )

        # add number of added lines to lines_changes
        number_parallel_lines_pre = edisgo_obj.topology.lines_df.loc[
            lines, "num_parallel"
        ]
        lines_changes.update(
            (number_parallel_lines - number_parallel_lines_pre).to_dict()
        )

        # update number of parallel lines and line accordingly attributes
        edisgo_obj.topology.update_number_of_parallel_lines(number_parallel_lines)

    def _add_one_parallel_line_of_same_type(lines):
        """
        Adds one parallel line of same type.

        Adds number of added lines to `lines_changes` dictionary.

        Parameters
        ----------
        lines : list(str)
            List of line names to add parallel line of same type for.

        """
        # add number of added lines to lines_changes
        lines_changes.update(pd.Series(index=lines, data=[1] * len(lines)).to_dict())

        # update number of lines and accordingly line attributes
        edisgo_obj.topology.update_number_of_parallel_lines(
            pd.Series(index=lines, data=[2] * len(lines))
        )

    def _replace_by_parallel_standard_lines(lines):
        """
        Replaces existing line with as many parallel standard lines as needed.

        Adds number of added lines to `lines_changes` dictionary.

        Parameters
        ----------
        lines : list(str)
            List of line names to replace by parallel standard lines.

        """
        # save old nominal power to calculate number of parallel standard lines
        s_nom_old = edisgo_obj.topology.lines_df.loc[lines, "s_nom"]

        # change line type to standard line
        edisgo_obj.topology.change_line_type(lines, standard_line_type)

        # calculate and update number of parallel lines
        number_parallel_lines = np.ceil(
            s_nom_old
            * crit_lines.loc[lines, "max_rel_overload"]
            / edisgo_obj.topology.lines_df.loc[lines, "s_nom"]
        )
        edisgo_obj.topology.update_number_of_parallel_lines(number_parallel_lines)

        lines_changes.update(number_parallel_lines.to_dict())

    lines_changes = {}

    # chose lines of right grid level
    relevant_lines = edisgo_obj.topology.lines_df.loc[
        crit_lines[crit_lines.voltage_level == voltage_level].index
    ]
    if not relevant_lines.empty:
        nominal_voltage = edisgo_obj.topology.buses_df.loc[
            edisgo_obj.topology.lines_df.loc[relevant_lines.index[0], "bus0"], "v_nom"
        ]
        standard_line_type = get_standard_line(
            edisgo_obj, nominal_voltage=nominal_voltage
        )

        # handling of standard lines
        lines_standard = relevant_lines.loc[
            relevant_lines.type_info == standard_line_type
        ]
        if not lines_standard.empty:
            _add_parallel_standard_lines(lines_standard.index)

        # get lines that have not been updated yet (i.e. that are not standard
        # lines)
        relevant_lines = relevant_lines.loc[
            ~relevant_lines.index.isin(lines_standard.index)
        ]
        # handling of cables where adding one cable is sufficient
        lines_single = (
            relevant_lines.loc[relevant_lines.num_parallel == 1]
            .loc[relevant_lines.kind == "cable"]
            .loc[crit_lines.max_rel_overload < 2]
        )
        if not lines_single.empty:
            _add_one_parallel_line_of_same_type(lines_single.index)

        # handle rest of lines (replace by as many parallel standard lines as
        # needed)
        relevant_lines = relevant_lines.loc[
            ~relevant_lines.index.isin(lines_single.index)
        ]
        if not relevant_lines.empty:
            _replace_by_parallel_standard_lines(relevant_lines.index)

    return lines_changes


def separate_lv_grid(
    edisgo_obj: EDisGo,
    grid: LVGrid,
    use_standard_line_type: bool = True,
) -> tuple[dict[Any, Any], dict[str, int]]:
    """
    Separate LV grid by adding a new substation and connect half of each feeder.

    If a feeder cannot be split because it has too few nodes or too few nodes outside a
    building, each second inept feeder is connected to the new LV grid. The new LV grid
    is equipped with standard transformers until the nominal apparent power is at least
    the same as in the original LV grid. The new substation is at the same location as
    the originating substation. The workflow is as follows:

    * The point at half the length of the feeders is determined.
    * The first node following this point is chosen as the point where the new
      connection will be made.
    * New MV/LV station is connected to the existing MV/LV station.
    * The determined nodes are disconnected from the previous nodes and connected to the
      new MV/LV station.

    Notes:

    * The name of the new LV grid will be a combination of the originating existing grid
      ID. E.g. 40000 + X = 40000X
    * The name of the lines in the new LV grid are the same as in the grid where the
      nodes were removed
    * Except line names, all the data frames are named based on the new grid name

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.LVGrid`
    use_standard_line_type : bool
        If True, standard line type is used to connect bus, where feeder is split, to
        the station. If False, the same line type and number of parallel lines as
        the original line is used. Default: True.

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.
    dict
        Dictionary with added transformers in the form::

            {'added': {'Grid_1': ['transformer_reinforced_1',
                                  ...,
                                  'transformer_reinforced_x'],
                       'Grid_10': ['transformer_reinforced_10']
                       }
            }

    """

    def get_weight(u, v, data: dict) -> float:
        return data["length"]

    def create_bus_name(bus: str, lv_grid_id_new: int, voltage_level: str) -> str:
        """
        Create an LV and MV bus-bar name with the same grid_id but added '1001' which
        implies the separation.

        Parameters
        ----------
        bus : str
            Bus name. E.g. 'BusBar_mvgd_460_lvgd_131573_LV'
        voltage_level : str
            'mv' or 'lv'

        Returns
        ----------
        str
            New bus-bar name.

        """
        if bus in edisgo_obj.topology.buses_df.index:
            bus = bus.split("_")

            bus[-2] = lv_grid_id_new

            if voltage_level == "lv":
                bus = "_".join([str(_) for _ in bus])
            elif voltage_level == "mv":
                bus[-1] = "MV"
                bus = "_".join([str(_) for _ in bus])
            else:
                logger.error(
                    f"Voltage level can only be 'mv' or 'lv'. Voltage level used: "
                    f"{voltage_level}."
                )
        else:
            raise IndexError(f"Station bus {bus} is not within the buses DataFrame.")

        return bus

    def add_standard_transformer(
        edisgo_obj: EDisGo, grid: LVGrid, bus_lv: str, bus_mv: str, lv_grid_id_new: int
    ) -> dict:
        """
        Adds standard transformer to topology.

        Parameters
        ----------
        edisgo_obj : class:`~.EDisGo`
        grid : `~.network.grids.LVGrid`
        bus_lv : str
            Identifier of LV bus.
        bus_mv : str
            Identifier of MV bus.

        Returns
        ----------
        dict

        """
        if bus_lv not in edisgo_obj.topology.buses_df.index:
            raise ValueError(
                f"Specified bus {bus_lv} is not valid as it is not defined in "
                "buses_df."
            )
        if bus_mv not in edisgo_obj.topology.buses_df.index:
            raise ValueError(
                f"Specified bus {bus_mv} is not valid as it is not defined in "
                "buses_df."
            )

        try:
            standard_transformer = edisgo_obj.topology.equipment_data[
                "lv_transformers"
            ].loc[
                edisgo_obj.config["grid_expansion_standard_equipment"][
                    "mv_lv_transformer"
                ]
            ]
        except KeyError:
            raise KeyError("Standard MV/LV transformer is not in the equipment list.")

        transformer_changes = {"added": {}}

        new_transformer_df = grid.transformers_df.iloc[[0]]
        new_transformer_name = new_transformer_df.index[0].split("_")
        grid_id_ind = new_transformer_name.index(str(grid.id))
        new_transformer_name[grid_id_ind] = lv_grid_id_new

        new_transformer_df.s_nom = standard_transformer.S_nom
        new_transformer_df.type_info = None
        new_transformer_df.r_pu = standard_transformer.r_pu
        new_transformer_df.x_pu = standard_transformer.x_pu
        new_transformer_df.index = ["_".join([str(_) for _ in new_transformer_name])]
        new_transformer_df.bus0 = bus_mv
        new_transformer_df.bus1 = bus_lv

        old_s_nom = grid.transformers_df.s_nom.sum()

        max_iterations = 10
        n = 0

        while old_s_nom > new_transformer_df.s_nom.sum() and n < max_iterations:
            n += 1

            another_new_transformer = new_transformer_df.iloc[-1:, :]

            old_name = another_new_transformer.index[0]

            name = old_name.split("_")

            try:
                name[-1] = str(int(name[-1]) + 1)
            except ValueError:
                name.append("1")

            name = "_".join(name)

            another_new_transformer.rename(index={old_name: name}, inplace=True)

            new_transformer_df = pd.concat(
                [new_transformer_df, another_new_transformer]
            )

        edisgo_obj.topology.transformers_df = pd.concat(
            [edisgo_obj.topology.transformers_df, new_transformer_df]
        )
        transformer_changes["added"][
            f"LVGrid_{lv_grid_id_new}"
        ] = new_transformer_df.index.tolist()

        return transformer_changes

    G = grid.graph

    # main station
    station_node = grid.transformers_df.bus1.iat[0]

    relevant_lines = grid.lines_df.loc[
        (grid.lines_df.bus0 == station_node) | (grid.lines_df.bus1 == station_node)
    ]

    first_nodes = set(relevant_lines.bus0).union(set(relevant_lines.bus1)) - {
        station_node,
    }

    if len(relevant_lines) <= 1:
        logger.warning(
            f"{grid} has only {len(relevant_lines)} feeder and is therefore not "
            f"separated."
        )

        return {}, {}

    logger.debug(f"{grid} has {len(relevant_lines)} feeder.")

    paths = {}
    first_nodes_feeders = {}

    # determine ordered shortest path between each node and the station node and each
    # node per feeder
    for node in G.nodes:
        if node == station_node:
            continue

        path = nx.shortest_path(G, station_node, node)

        for first_node in first_nodes:
            if first_node in path:
                paths[node] = path

                first_nodes_feeders.setdefault(first_node, []).append(
                    node  # first nodes and paths
                )

    # note: The number of critical lines in the Lv grid can be more than 2. However,
    # if the node_1_2 of the first feeder in the for loop is not the first node of the
    # feeder, it will add data frames even though the following feeders only 1 node
    # (node_1_2=first node of feeder). In this type of case,the number of critical lines
    # should be evaluated for the feeders whose node_1_2 s are not the first node of the
    # feeder. The first check should be done on the feeders that have fewer nodes.

    first_nodes_feeders = dict(
        sorted(
            first_nodes_feeders.items(), key=lambda item: len(item[1]), reverse=False
        )
    )

    # make sure nodes are sorted correctly and node_1_2 is part of the main feeder
    for first_node, nodes_feeder in first_nodes_feeders.items():
        paths_first_node = {
            node: path for node, path in paths.items() if path[1] == first_node
        }

        # identify main feeder by maximum number of nodes in path
        first_nodes_feeders[first_node] = paths_first_node[
            max(paths_first_node, key=lambda x: len(paths_first_node[x]))
        ]

    lines_changes = {}
    transformers_changes = {}
    nodes_tb_relocated = {}  # nodes to be moved into the new grid

    count_inept = 0

    for first_node, nodes_feeder in first_nodes_feeders.items():
        # first line of the feeder
        first_line = relevant_lines[
            (relevant_lines.bus1 == first_node) | (relevant_lines.bus0 == first_node)
        ].index[0]

        # the last node of the feeder
        last_node = nodes_feeder[-1]

        # the length of each line (the shortest path)
        path_length_dict_tmp = dijkstra_shortest_path_length(
            G, station_node, get_weight, target=last_node
        )

        # path does not include the nodes branching from the node on the main path
        path = paths[last_node]

        # TODO: replace this to be weighted by the connected load per bus incl.
        #  branched of feeders
        node_1_2 = next(
            j
            for j in path
            if path_length_dict_tmp[j] >= path_length_dict_tmp[last_node] * 1 / 2
        )

        # if LVGrid: check if node_1_2 is outside a house
        # and if not find next BranchTee outside the house
        while (
            ~np.isnan(grid.buses_df.loc[node_1_2].in_building)
            and grid.buses_df.loc[node_1_2].in_building
        ):
            node_1_2 = path[path.index(node_1_2) - 1]
            # break if node is station
            if node_1_2 is path[0]:
                logger.warning(
                    f"{grid} ==> {first_line} and following lines could not be "
                    f"reinforced due to insufficient number of node in the feeder. "
                    f"A method to handle such cases is not yet implemented."
                )

                node_1_2 = path[path.index(node_1_2) + 1]

                break

        # NOTE: If node_1_2 is a representative (meaning it is already directly
        #  connected to the station) feeder cannot be split. Instead, every second
        #  inept feeder is assigned to the new grid
        if node_1_2 not in first_nodes_feeders or count_inept % 2 == 1:
            nodes_tb_relocated[node_1_2] = get_downstream_buses(edisgo_obj, node_1_2)

            if node_1_2 in first_nodes_feeders:
                count_inept += 1
        else:
            count_inept += 1

    if nodes_tb_relocated:
        # generate new lv grid id
        n = 0
        lv_grid_id_new = int(f"{grid.id}{n}")

        max_iterations = 10**4

        g_ids = [g.id for g in edisgo_obj.topology.mv_grid.lv_grids]

        while lv_grid_id_new in g_ids:
            n += 1
            lv_grid_id_new = int(f"{grid.id}{n}")

            if n >= max_iterations:
                raise ValueError(
                    f"No suitable name for the new LV grid originating from {grid} was "
                    f"found in {max_iterations=}."
                )

        # Create the bus-bar name of primary and secondary side of new MV/LV station
        lv_bus_new = create_bus_name(station_node, lv_grid_id_new, "lv")
        mv_bus = grid.transformers_df.bus0.iat[0]

        # Add MV and LV bus
        v_nom_lv = edisgo_obj.topology.buses_df.at[
            grid.transformers_df.bus1[0],
            "v_nom",
        ]

        x_bus = grid.buses_df.at[station_node, "x"]
        y_bus = grid.buses_df.at[station_node, "y"]

        building_bus = grid.buses_df.at[station_node, "in_building"]

        # add lv busbar
        edisgo_obj.topology.add_bus(
            lv_bus_new,
            v_nom_lv,
            x=x_bus,
            y=y_bus,
            lv_grid_id=lv_grid_id_new,
            in_building=building_bus,
        )

        # ADD TRANSFORMER
        transformer_changes = add_standard_transformer(
            edisgo_obj, grid, lv_bus_new, mv_bus, lv_grid_id_new
        )
        transformers_changes.update(transformer_changes)

        logger.info(f"New LV grid {lv_grid_id_new} added to topology.")

        lv_standard_line = get_standard_line(edisgo_obj, nominal_voltage=0.4)

        # changes on relocated lines to the new LV grid
        # grid_ids
        for node_1_2, nodes in nodes_tb_relocated.items():
            # the last node of the feeder
            last_node = nodes[-1]

            # path does not include the nodes branching from the node on the main path
            path = paths[last_node]

            nodes.append(node_1_2)

            edisgo_obj.topology.buses_df.loc[nodes, "lv_grid_id"] = lv_grid_id_new

            dist = dijkstra_shortest_path_length(
                G, station_node, get_weight, target=node_1_2
            )[node_1_2]

            # predecessor node of node_1_2
            pred_node = path[path.index(node_1_2) - 1]
            # the line
            line_removed = G.get_edge_data(node_1_2, pred_node)["branch_name"]
            if use_standard_line_type is True:
                line_type = lv_standard_line
                num_parallel = 1
            else:
                type_info = edisgo_obj.topology.lines_df.at[line_removed, "type_info"]
                line_type = type_info if type_info is not None else lv_standard_line
                num_parallel = edisgo_obj.topology.lines_df.at[
                    line_removed, "num_parallel"
                ]
            line_added_lv = edisgo_obj.add_component(
                comp_type="line",
                bus0=lv_bus_new,
                bus1=node_1_2,
                length=dist,
                type_info=line_type,
                num_parallel=num_parallel,
            )

            lines_changes[line_added_lv] = num_parallel

            edisgo_obj.remove_component(
                comp_type="line",
                comp_name=line_removed,
            )

        logger.info(
            f"{len(nodes_tb_relocated.keys())} feeders are removed from the grid "
            f"{grid} and located in new grid {lv_grid_id_new} by method: "
            f"add_station_at_half_length "
        )

        # check if new grids have isolated nodes
        grids = [
            g
            for g in edisgo_obj.topology.mv_grid.lv_grids
            if g.id in [grid.id, lv_grid_id_new]
        ]

        for g in grids:
            n = nx.number_of_isolates(g.graph)

            if n > 0 and len(g.buses_df) > 1:
                raise ValueError(
                    f"There are isolated nodes in {g}. The following nodes are "
                    f"isolated: {list(nx.isolates(g.graph))}"
                )

    else:
        logger.warning(f"{grid} was not split because it has too few suitable feeders.")

    return transformers_changes, lines_changes
