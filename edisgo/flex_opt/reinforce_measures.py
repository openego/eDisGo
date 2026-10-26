# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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

from edisgo.flex_opt.costs import line_expansion_costs
from edisgo.network.grids import LVGrid, MVGrid
from edisgo.tools.tools import calculate_apparent_power, get_downstream_buses

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
            f"{voltage_level} is not a valid option for input variable "
            "'voltage_level' in function _station_overloading. Try 'mv' or "
            "'lv'."
        )

    transformers_changes = {"added": {}, "removed": {}}
    # Collect the per-station frames of newly added transformers and the indices
    # of transformers to be removed, so the topology dataframe is rebuilt only
    # once after the loop instead of once per station (avoids O(stations x
    # total_transformers) behaviour). Output is identical: surviving original
    # rows keep their order and the new rows are concatenated in station order.
    new_transformers_collected: list[pd.DataFrame] = []
    removed_transformers_collected: list[np.ndarray] = []
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
            # collect previous transformers to be removed from topology (batched
            # after the loop)
            removed_transformers_collected.append(grid.transformers_df.index.values)

        # collect new transformers to be added to topology (batched after the loop)
        new_transformers_collected.append(new_transformers)

    # remove previous transformers from topology in a single drop. Each station
    # corresponds to a distinct grid, so dropping one station's transformers does
    # not affect the per-grid views read for other stations during the loop.
    if removed_transformers_collected:
        removed_index = np.concatenate(removed_transformers_collected)
        if voltage_level == "lv":
            edisgo_obj.topology.transformers_df.drop(removed_index, inplace=True)
        else:
            edisgo_obj.topology.transformers_hvmv_df.drop(removed_index, inplace=True)

    # add all new transformers to topology in a single concat. Surviving original
    # rows keep their order and new rows are appended in station order, matching
    # the per-iteration concat behaviour.
    if new_transformers_collected:
        if voltage_level == "lv":
            edisgo_obj.topology.transformers_df = pd.concat(
                [edisgo_obj.topology.transformers_df, *new_transformers_collected]
            )
        else:
            edisgo_obj.topology.transformers_hvmv_df = pd.concat(
                [
                    edisgo_obj.topology.transformers_hvmv_df,
                    *new_transformers_collected,
                ]
            )
    return transformers_changes


def reinforce_mv_lv_station_voltage_issues(
    edisgo_obj, critical_stations, return_node_mapping=False
):
    """
    Reinforce MV/LV substations due to voltage issues.

    A parallel standard transformer is installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    critical_stations : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with maximum deviations from allowed lower or upper voltage limits
        in p.u. for all MV-LV stations with voltage issues. For more information on
        dataframe see :func:`~.flex_opt.check_tech_constraints.voltage_issues`.
    return_node_mapping : bool
        If True, additionally returns a dictionary mapping each added
        transformer to the critical station (as in the index of
        `critical_stations`) that triggered its reinforcement. Default: False.

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
    dict
        Only returned if `return_node_mapping` is True. Dictionary with name
        of added transformers as keys and the name of the critical station
        that triggered the transformer's addition as values.

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
    node_mapping = {}
    # Collect the per-station new transformer frames so the topology dataframe is
    # rebuilt only once after the loop instead of once per station (avoids
    # O(stations x total_transformers) behaviour). Output is identical: the new
    # rows are concatenated in station order.
    new_transformers_collected: list[pd.DataFrame] = []
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
        # collect new transformer to be added to topology (batched after the loop)
        new_transformers_collected.append(duplicated_transformer)
        transformers_changes["added"][str(grid)] = duplicated_transformer.index.tolist()
        for new_transformer_name in duplicated_transformer.index:
            node_mapping[new_transformer_name] = station

    # add all new transformers to topology in a single concat, in station order,
    # matching the per-iteration concat behaviour.
    if new_transformers_collected:
        edisgo_obj.topology.transformers_df = pd.concat(
            [edisgo_obj.topology.transformers_df, *new_transformers_collected]
        )

    if transformers_changes["added"]:
        logger.debug(
            "==> {} LV station(s) has/have been reinforced due to voltage "
            "issues.".format(len(transformers_changes["added"]))
        )

    if return_node_mapping:
        return transformers_changes, node_mapping
    return transformers_changes


def reinforce_lines_voltage_issues(
    edisgo_obj,
    grid,
    crit_nodes,
    choose_cheaper_measure=False,
    return_node_mapping=False,
):
    """
    Reinforce lines in MV and LV topology due to voltage issues.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_nodes : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with maximum deviations from allowed lower or upper voltage limits
        in p.u. for all buses in specified grid. For more information on dataframe see
        :func:`~.flex_opt.check_tech_constraints.voltage_issues`.
    choose_cheaper_measure : bool
        If True, a line is only disconnected if this is cheaper than reinforcing
        the lines on the path between station and critical node. If False, a line
        is always disconnected if possible. Default: False.
    return_node_mapping : bool
        If True, additionally returns a dictionary mapping each reinforced line
        to the critical node that triggered its reinforcement. Default: False.

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.
    dict
        Only returned if `return_node_mapping` is True. Dictionary with name
        of reinforced lines as keys and the name of the critical node that
        triggered the line's reinforcement as values.

    Notes
    -----
    Reinforce measures:

    1. Disconnect line at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line
    2. If no line can be disconnected, reinforce lines on the path between
    station and critical node (see :func:`_reinforce_lines_on_path`)

    If `choose_cheaper_measure` is True, the costs of both measures are compared
    before a line is disconnected. The costs of disconnecting are the costs of the
    new standard line including earthwork. The costs of reinforcing the path are
    the costs of the lines that need to be reinforced to remove the full share of
    the voltage difference along the path (see :func:`_plan_lines_on_path`), not
    only the half that is reinforced per iteration, as disconnecting usually
    solves the voltage issue at once. If reinforcing the path is cheaper, the
    lines on the path are reinforced instead of disconnecting the line.

    In LV grids lines are only disconnected outside buildings; loads and
    generators in buildings cannot be directly connected to the MV/LV station.
    House connections, i.e. lines to buses in buildings, are not reinforced
    either.

    In MV grids lines can only be disconnected at LV stations because they
    have switch disconnectors needed to operate the lines as half rings (loads
    in MV would be suitable as well because they have a switch bay (Schaltfeld)
    but loads in dingo are only connected to MV busbar). If there is no
    suitable LV station the generator is directly connected to the MV busbar.
    There is no need for a switch disconnector in that case because generators
    don't need to be n-1 safe.

    """

    # load standard line data
    if isinstance(grid, LVGrid):
        standard_line = edisgo_obj.config["grid_expansion_standard_equipment"][
            "lv_line"
        ]
    elif isinstance(grid, MVGrid):
        standard_line = edisgo_obj.config["grid_expansion_standard_equipment"][
            f"mv_line_{int(grid.nominal_voltage)}kv"
        ]
    else:
        raise ValueError("Inserted grid is invalid.")

    # find path to each node in order to find node with voltage issues farthest
    # away from station in each feeder
    station_node = grid.transformers_df.bus1.iloc[0]
    graph = grid.graph
    paths = {}
    nodes_feeder = {}
    for node in crit_nodes.index:
        path = nx.shortest_path(graph, station_node, node)
        paths[node] = path
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logger.error(
                f"Voltage issues at busbar in LV network {grid} should have "
                "been solved in previous steps."
            )
        nodes_feeder.setdefault(path[1], []).append(node)

    lines_changes = {}
    node_mapping = {}
    for repr_node in nodes_feeder.keys():
        # find node farthest away
        get_weight = lambda u, v, data: data["length"]  # noqa: E731
        path_length = 0
        for n in nodes_feeder[repr_node]:
            path_length_dict_tmp = dijkstra_shortest_path_length(
                graph, station_node, get_weight, target=n
            )
            if path_length_dict_tmp[n] > path_length:
                node = n
                path_length = path_length_dict_tmp[n]
                path_length_dict = path_length_dict_tmp
        path = paths[node]

        # find first node in path that exceeds 2/3 of the line length
        # from station to critical node farthest away from the station
        node_2_3 = next(
            j for j in path if path_length_dict[j] >= path_length_dict[node] * 2 / 3
        )

        # if LVGrid: check if node_2_3 is outside of a house
        # and if not find next BranchTee outside the house
        if isinstance(grid, LVGrid):
            while (
                ~np.isnan(grid.buses_df.loc[node_2_3].in_building)
                and grid.buses_df.loc[node_2_3].in_building
            ):
                node_2_3 = path[path.index(node_2_3) - 1]
                # break if node is station
                if node_2_3 is path[0]:
                    logger.error("Could not reinforce voltage issue.")
                    break

        # if MVGrid: check if node_2_3 is LV station and if not find
        # next LV station
        else:
            while node_2_3 not in edisgo_obj.topology.transformers_df.bus0.values:
                try:
                    # try to find LVStation behind node_2_3
                    node_2_3 = path[path.index(node_2_3) + 1]
                except IndexError:
                    # if no LVStation between node_2_3 and node with
                    # voltage problem, connect node directly to
                    # MVStation
                    node_2_3 = node
                    break

        # if node_2_3 is a representative (meaning it is already
        # directly connected to the station), line cannot be
        # disconnected and must therefore be reinforced
        disconnect = node_2_3 not in nodes_feeder.keys()
        if disconnect:
            # get line between node_2_3 and predecessor node (that is
            # closer to the station)
            pred_node = path[path.index(node_2_3) - 1]
            crit_line_name = graph.get_edge_data(node_2_3, pred_node)["branch_name"]
            crit_line_length = grid.lines_df.at[crit_line_name, "length"]
        if not disconnect or choose_cheaper_measure:
            path_lines, share, current = _get_path_lines_share_and_current(
                edisgo_obj, grid, graph, crit_nodes, node, path
            )

        # costs of the new line are derived from the costs of the line it replaces,
        # as costs only depend on line length and voltage level
        if disconnect and choose_cheaper_measure and crit_line_length > 0:
            costs = line_expansion_costs(edisgo_obj, [crit_line_name])
            costs_disconnect = (
                (costs.costs_earthworks + costs.costs_cable).iat[0]
                / crit_line_length
                * path_length_dict[node_2_3]
            )
            _, _, costs_path = _plan_lines_on_path(
                edisgo_obj,
                grid,
                path_lines,
                standard_line,
                share,
                current,
                max_costs=costs_disconnect,
            )
            disconnect = costs_path >= costs_disconnect
            logger.debug(
                f"Voltage issue at {node}: disconnecting {crit_line_name} costs "
                f"{costs_disconnect:.1f} kEUR, reinforcing the path costs "
                f"{costs_path:.1f} kEUR."
            )

        if not disconnect:
            path_changes = _reinforce_lines_on_path(
                edisgo_obj, grid, path_lines, standard_line, share, current
            )
            lines_changes.update(path_changes)
            node_mapping.update(dict.fromkeys(path_changes, node))

        # disconnect line
        else:
            if grid.lines_df.at[crit_line_name, "bus0"] == pred_node:
                edisgo_obj.topology._lines_df.at[crit_line_name, "bus0"] = station_node
            elif grid.lines_df.at[crit_line_name, "bus1"] == pred_node:
                edisgo_obj.topology._lines_df.at[crit_line_name, "bus1"] = station_node
            else:
                raise ValueError("Bus not in line buses. Please check.")
            # change line length and type
            edisgo_obj.topology._lines_df.at[crit_line_name, "length"] = (
                path_length_dict[node_2_3]
            )
            edisgo_obj.topology.change_line_type([crit_line_name], standard_line)
            lines_changes[crit_line_name] = 1
            node_mapping[crit_line_name] = node
            # TODO: Include switch disconnector

    if not lines_changes:
        logger.debug(
            f"==> {len(lines_changes)} line(s) was/were reinforced due to voltage "
            "issues."
        )

    if return_node_mapping:
        return lines_changes, node_mapping
    return lines_changes


def _get_path_lines_share_and_current(edisgo_obj, grid, graph, crit_nodes, node, path):
    """
    Get lines on the path to a critical node and the voltage issue on the path.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    graph : :networkx:`networkx.Graph<>`
        Graph of the grid the path was determined in.
    crit_nodes : :pandas:`pandas.DataFrame<DataFrame>`
        See parameter `crit_nodes` in :func:`reinforce_lines_voltage_issues`.
    node : str
        Name of critical node.
    path : list(str)
        Buses on the path from the station to the critical node.

    Returns
    -------
    (list(str), float, :pandas:`pandas.Series<Series>`)
        Names of lines on the path that may be reinforced, share of the voltage
        difference along the path that needs to be removed and current through each
        of these lines. See :func:`_reinforce_lines_on_path` for more information.

    """
    # share of the voltage difference between station and critical node
    # that needs to be removed to solve the voltage issue, used to decide
    # how many lines on the path are reinforced
    v_res = edisgo_obj.results.v_res
    time_step = crit_nodes.at[node, "time_index"]
    v_diff_path = abs(v_res.at[time_step, node] - v_res.at[time_step, path[0]])
    if v_diff_path > 0:
        share = min(1.0, crit_nodes.at[node, "abs_max_voltage_dev"] / v_diff_path)
    else:
        share = 1.0
    # house connections (lines to buses in buildings) are not reinforced
    in_building = grid.buses_df.in_building
    path_lines = [
        graph.get_edge_data(bus_0, bus_1)["branch_name"]
        for bus_0, bus_1 in zip(path[:-1], path[1:])
        if not (in_building[bus_0] or in_building[bus_1])
    ]
    # current through each line at the time step of the voltage issue,
    # all lines are weighted equally if no current is available
    current = (
        edisgo_obj.results.i_res.reindex(columns=path_lines)
        .loc[time_step]
        .abs()
        .fillna(0.0)
    )
    if current.sum() == 0:
        current[:] = 1.0
    return path_lines, share, current


def _reinforce_lines_on_path(
    edisgo_obj, grid, path_lines, standard_line, share, current
):
    """
    Reinforce lines on the path from the station to a critical node.

    The voltage difference along the path is estimated as the sum of resistance times
    current of all lines on the path. Lines are reinforced one at a time until this
    voltage difference is reduced by `share` / 2, e.g. by 10 % for a share of 0.2.

    Each time, the measure that reduces the voltage difference the most per cost is
    chosen. The cost of a measure is the cost of the new cable plus, if the line was
    not reinforced before, the cost of the earthwork. Earthwork is only paid once per
    line, as in :func:`~.flex_opt.costs.grid_expansion_costs`, so adding another
    parallel line to an already reinforced line is cheaper than reinforcing a new
    one.

    Only half of the share is used because the share assumes that the voltage
    difference along the path decreases proportionally to the reduction of
    resistance times current. In the test grids, using the full share led to more
    reinforcement than needed. Voltage issues that remain are handled in the next
    iteration of the grid reinforcement.

    A line that is not yet a standard line is replaced by one standard line, unless
    this would increase its resistance, e.g. because the line has a larger cross
    section than the standard line. In that case, as for standard lines, one more
    parallel line of the same type is installed.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    path_lines : list(str)
        Names of lines on the path from the station to the critical node.
    standard_line : str
        Name of standard line type.
    share : float
        Voltage deviation at the critical node divided by the voltage difference
        between station and critical node, i.e. the share of the voltage difference
        that needs to be removed to solve the voltage issue. Between 0 and 1.
    current : :pandas:`pandas.Series<Series>`
        Current through each line on the path at the time step of the voltage
        issue. Index contains the names of the lines in `path_lines`.

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.

    """
    lines_changes, lines, _ = _plan_lines_on_path(
        edisgo_obj, grid, path_lines, standard_line, share / 2, current
    )
    lines_before = grid.lines_df.loc[path_lines]
    # a line is replaced at most once and before parallel lines are added to it
    replaced = lines.index[
        (lines.type_info == standard_line) & (lines_before.type_info != standard_line)
    ]
    if not replaced.empty:
        edisgo_obj.topology.change_line_type(list(replaced), standard_line)
    lines_before = grid.lines_df.loc[path_lines]
    num_parallel = lines.num_parallel[lines.num_parallel != lines_before.num_parallel]
    if not num_parallel.empty:
        edisgo_obj.topology.update_number_of_parallel_lines(num_parallel)

    return lines_changes


def _plan_lines_on_path(
    edisgo_obj, grid, path_lines, standard_line, reduction, current, max_costs=np.inf
):
    """
    Determine the lines on the path to reinforce, without changing the topology.

    Lines are chosen one at a time as described in :func:`_reinforce_lines_on_path`
    until the voltage difference along the path, estimated as the sum of resistance
    times current, is reduced by `reduction`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    path_lines : list(str)
        Names of lines on the path from the station to the critical node.
    standard_line : str
        Name of standard line type.
    reduction : float
        Share by which the voltage difference along the path is reduced. Between 0
        and 1.
    current : :pandas:`pandas.Series<Series>`
        Current through each line on the path at the time step of the voltage
        issue. Index contains the names of the lines in `path_lines`.
    max_costs : float
        Planning stops once the costs reach this value, in kEUR. Default: inf.

    Returns
    -------
    (dict, :pandas:`pandas.DataFrame<DataFrame>`, float)
        Dictionary with name of lines as keys and the corresponding number of lines
        added as values, lines on the path with their resistance, number of parallel
        lines and type after the reinforcement, and costs of the reinforcement in
        kEUR. The costs are inf if the reduction is not reached within `max_costs`.

    """
    cables = "lv_cables" if isinstance(grid, LVGrid) else "mv_cables"
    r_per_km_standard = edisgo_obj.topology.equipment_data[cables].at[
        standard_line, "R_per_km"
    ]
    costs = line_expansion_costs(edisgo_obj, path_lines)
    reinforced_before = edisgo_obj.results.equipment_changes.index
    lines = grid.lines_df.loc[path_lines, ["length", "r", "num_parallel", "type_info"]]
    v_target = (lines.r * current).sum() * (1 - reduction)

    lines_changes = {}
    total_costs = 0.0
    while (lines.r * current).sum() > v_target and total_costs < max_costs:
        r_standard = r_per_km_standard * lines.length
        replace = (lines.type_info != standard_line) & (r_standard < lines.r)
        r_reduction = (lines.r / (lines.num_parallel + 1)).where(
            ~replace, lines.r - r_standard
        )
        v_reduction = r_reduction * current
        if v_reduction.max() <= 0:
            break
        new_line = ~lines.index.isin(reinforced_before) & ~lines.index.isin(
            list(lines_changes)
        )
        measure_costs = costs.costs_cable + costs.costs_earthworks.where(new_line, 0.0)
        line_name = (v_reduction / measure_costs).fillna(0.0).idxmax()

        lines.at[line_name, "r"] -= r_reduction[line_name]
        if replace[line_name]:
            lines.at[line_name, "type_info"] = standard_line
            lines.at[line_name, "num_parallel"] = 1
        else:
            lines.at[line_name, "num_parallel"] += 1
        lines_changes[line_name] = lines_changes.get(line_name, 0) + 1
        total_costs += measure_costs[line_name]

    if (lines.r * current).sum() > v_target:
        total_costs = np.inf
    return lines_changes, lines, total_costs


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
            f"==> {crit_lines.shape[0]} line(s) was/were reinforced due to "
            "over-loading issues."
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
        if nominal_voltage == 0.4:
            standard_line_type = edisgo_obj.config["grid_expansion_standard_equipment"][
                "lv_line"
            ]
        else:
            standard_line_type = edisgo_obj.config["grid_expansion_standard_equipment"][
                f"mv_line_{int(nominal_voltage)}kv"
            ]

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
    edisgo_obj: EDisGo, grid: LVGrid
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
      new MV/LV station. The new line is of the standard type, with as many parallel
      lines as needed to be at least as strong (``s_nom``) as the line it replaces.

    Notes:

    * The name of the new LV grid will be a combination of the originating existing grid
      ID. E.g. 40000 + X = 40000X
    * The name of the lines in the new LV grid are the same as in the grid where the
      nodes were removed
    * Except line names, all the data frames are named based on the new grid name

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo object whose topology is modified in place by adding the new
        substation, buses and lines.
    grid : :class:`~.network.grids.LVGrid`
        The overloaded LV grid to be split. Its feeders are halved and the second
        half is reconnected to a newly added MV/LV station.

    Returns
    -------
    dict
        Dictionary with the names of changed lines as keys and the corresponding
        number of parallel lines added as values.
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
                f"Specified bus {bus_lv} is not valid as it is not defined in buses_df."
            )
        if bus_mv not in edisgo_obj.topology.buses_df.index:
            raise ValueError(
                f"Specified bus {bus_mv} is not valid as it is not defined in buses_df."
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
        new_transformer_df.type_info = standard_transformer.name
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
        transformer_changes["added"][f"LVGrid_{lv_grid_id_new}"] = (
            new_transformer_df.index.tolist()
        )

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
            grid.transformers_df.bus1.iloc[0],
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

        lv_standard_line = edisgo_obj.config["grid_expansion_standard_equipment"][
            "lv_line"
        ]
        lv_standard_line_data = edisgo_obj.topology.equipment_data["lv_cables"].loc[
            lv_standard_line
        ]
        s_nom_standard_line = calculate_apparent_power(
            lv_standard_line_data.U_n, lv_standard_line_data.I_max_th, 1
        )

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

            # the new line carries the load of the line it replaces, so it gets as
            # many parallel standard lines as needed to be at least as strong (the
            # small offset keeps e.g. exactly two standard lines from rounding to 3);
            # one standard line if the replaced line has no nominal power
            s_nom_removed = edisgo_obj.topology.lines_df.at[line_removed, "s_nom"]
            num_parallel = (
                1
                if pd.isna(s_nom_removed)
                else max(1, math.ceil(s_nom_removed / s_nom_standard_line - 1e-9))
            )

            line_added_lv = edisgo_obj.add_component(
                comp_type="line",
                bus0=lv_bus_new,
                bus1=node_1_2,
                length=dist,
                type_info=lv_standard_line,
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
