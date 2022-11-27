import logging
import math

import networkx as nx
import numpy as np
import pandas as pd

from networkx.algorithms.shortest_paths.weighted import (
    _dijkstra as dijkstra_shortest_path_length,
)

from edisgo.network.components import Switch
from edisgo.network.grids import LVGrid, MVGrid

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

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1': ['transformer_1']}
        }

    """
    transformers_changes = _station_overloading(
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

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1': ['transformer_1']}
        }

    """
    transformers_changes = _station_overloading(
        edisgo_obj, critical_stations, voltage_level="mv"
    )

    if transformers_changes["added"]:
        logger.debug("==> MV station has been reinforced due to overloading issues.")

    return transformers_changes


def _station_overloading(edisgo_obj, critical_stations, voltage_level):
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

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   },
         'removed': {'Grid_1': ['transformer_1']}
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
    for grid_name in critical_stations.index:
        if "MV" in grid_name:
            grid = edisgo_obj.topology.mv_grid
        else:
            grid = edisgo_obj.topology.get_lv_grid(grid_name)
        # list of maximum power of each transformer in the station
        s_max_per_trafo = grid.transformers_df.s_nom
        # missing capacity
        s_trafo_missing = critical_stations.at[grid_name, "s_missing"]

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
            transformers_changes["added"][grid_name] = [new_transformers.index[0]]
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
            transformers_changes["added"][grid_name] = new_transformers.index.values
            # add previous transformers to list of removed transformers
            transformers_changes["removed"][
                grid_name
            ] = grid.transformers_df.index.values
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
    critical_stations : :obj:`dict`
        Dictionary with representative of :class:`~.network.grids.LVGrid` as
        key and a :pandas:`pandas.DataFrame<DataFrame>` with station's voltage
        deviation from allowed lower or upper voltage limit as value.
        Index of the dataframe is the station with voltage issues.
        Columns are 'v_diff_max' containing the maximum voltage deviation as
        float and 'time_index' containing the corresponding time step the
        voltage issue occured in as :pandas:`pandas.Timestamp<Timestamp>`.

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
    for grid_name in critical_stations.keys():
        if "MV" in grid_name:
            grid = edisgo_obj.topology.mv_grid
        else:
            grid = edisgo_obj.topology.get_lv_grid(grid_name)
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
        transformers_changes["added"][grid_name] = duplicated_transformer.index.tolist()

    if transformers_changes["added"]:
        logger.debug(
            "==> {} LV station(s) has/have been reinforced due to voltage "
            "issues.".format(len(transformers_changes["added"]))
        )

    return transformers_changes


def reinforce_lines_voltage_issues(edisgo_obj, grid, crit_nodes):
    """
    Reinforce lines in MV and LV topology due to voltage issues.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    grid : :class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_nodes : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with all nodes with voltage issues in the grid and
        their maximal deviations from allowed lower or upper voltage limits
        sorted descending from highest to lowest voltage deviation
        (it is not distinguished between over- or undervoltage).
        Columns of the dataframe are 'v_diff_max' containing the maximum
        absolute voltage deviation as float and 'time_index' containing the
        corresponding time step the voltage issue occured in as
        :pandas:`pandas.Timestamp<Timestamp>`. Index of the dataframe are the
        names of all buses with voltage issues.

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.

    Notes
    -----
    Reinforce measures:

    1. Disconnect line at 2/3 of the length between station and critical node
    farthest away from the station and install new standard line
    2. Install parallel standard line

    In LV grids only lines outside buildings are reinforced; loads and
    generators in buildings cannot be directly connected to the MV/LV station.

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
            logging.error(
                "Voltage issues at busbar in LV network {} should have "
                "been solved in previous steps.".format(grid)
            )
        nodes_feeder.setdefault(path[1], []).append(node)

    lines_changes = {}
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
        if node_2_3 in nodes_feeder.keys():
            crit_line_name = graph.get_edge_data(station_node, node_2_3)["branch_name"]
            crit_line = grid.lines_df.loc[crit_line_name]

            # if critical line is already a standard line install one
            # more parallel line
            if crit_line.type_info == standard_line:
                edisgo_obj.topology.update_number_of_parallel_lines(
                    pd.Series(
                        index=[crit_line_name],
                        data=[
                            edisgo_obj.topology._lines_df.at[
                                crit_line_name, "num_parallel"
                            ]
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

        # if node_2_3 is not a representative, disconnect line
        else:
            # get line between node_2_3 and predecessor node (that is
            # closer to the station)
            pred_node = path[path.index(node_2_3) - 1]
            crit_line_name = graph.get_edge_data(node_2_3, pred_node)["branch_name"]
            if grid.lines_df.at[crit_line_name, "bus0"] == pred_node:
                edisgo_obj.topology._lines_df.at[crit_line_name, "bus0"] = station_node
            elif grid.lines_df.at[crit_line_name, "bus1"] == pred_node:
                edisgo_obj.topology._lines_df.at[crit_line_name, "bus1"] = station_node
            else:
                raise ValueError("Bus not in line buses. Please check.")
            # change line length and type
            edisgo_obj.topology._lines_df.at[
                crit_line_name, "length"
            ] = path_length_dict[node_2_3]
            edisgo_obj.topology.change_line_type([crit_line_name], standard_line)
            lines_changes[crit_line_name] = 1
            # ToDo: Include switch disconnector

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


def add_same_type_of_parallel_line(edisgo_obj, crit_lines):
    """
    Adds one parallel line of same type.
    Adds number of added lines to `lines_changes` dictionary.

    Parameters
    ----------
    crit_lines: pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing over-loaded lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the dataframe are the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time step the over-loading occured in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').
    edisgo_obj: class:`~.EDisGo`

    Returns
    -------
    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.

    Notes
    ------

    """

    lines_changes = {}
    # add number of added lines to lines_changes
    lines_changes.update(
        pd.Series(index=crit_lines.index, data=[1] * len(crit_lines.index)).to_dict()
    )

    # update number of lines and accordingly line attributes

    edisgo_obj.topology.update_number_of_parallel_lines(
        pd.Series(
            index=crit_lines.index,
            data=(
                edisgo_obj.topology.lines_df[
                    edisgo_obj.topology.lines_df.index.isin(crit_lines.index)
                ].num_parallel
                + 1
            ),
        )
    )

    return lines_changes


def split_feeder_at_half_length(edisgo_obj, grid, crit_lines):
    """
    The critical string load in MV and LV grid is remedied by splitting the feeder
    at the half-length

    1-The point at half the length of the feeders is found.
    2-The first node following this point is chosen as the point where the new
    connection will be made. This node can only be a station.
    3-This node is disconnected from the previous node and connected to the main station

    Notes:
    In LV grids, the node inside the building is not considered.
    The method is not applied if the node is the first node after the main station.


    Parameters
    ----------
    edisgo_obj: class:`~.EDisGo`
    grid: class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_lines:  Dataframe containing over-loaded lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the data frame is the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time-step the over-loading occurred in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').


    Returns
    -------
    dict

    Dictionary with the name of lines as keys and the corresponding number of
    lines added as values.

    Notes
    -----
    In this method, the separation is done according to the longest route
    (not the feeder has more load)
    """

    def get_weight(u, v, data):
        return data["length"]

    if isinstance(grid, LVGrid):

        voltage_level = "lv"
        relevant_lines = edisgo_obj.topology.lines_df.loc[
            crit_lines[crit_lines.voltage_level == voltage_level].index
        ]
        """
        # TODO:to be deleted after decision
        if not relevant_lines.empty:
            nominal_voltage = edisgo_obj.topology.buses_df.loc[
                edisgo_obj.topology.lines_df.loc[relevant_lines.index[0], "bus0"],
                "v_nom",
            ]
            standard_line_type = edisgo_obj.config["grid_expansion_standard_equipment"][
                "lv_line"
            ]
        """
    elif isinstance(grid, MVGrid):

        voltage_level = "mv"
        # find all the mv lines that have overloading issues in lines_df
        relevant_lines = edisgo_obj.topology.lines_df.loc[
            crit_lines[crit_lines.voltage_level == voltage_level].index
        ]
        # TODO:to be deleted after decision
        """
        if not relevant_lines.empty:
            nominal_voltage = edisgo_obj.topology.buses_df.loc[
                edisgo_obj.topology.lines_df.loc[relevant_lines.index[0], "bus0"],
                "v_nom",
            ]
            standard_line_type = edisgo_obj.config["grid_expansion_standard_equipment"][
                "lv_line"
            ]
        """

    else:
        raise ValueError(f"Grid Type {type(grid)} is not supported.")

    G = grid.graph
    station_node = list(G.nodes)[0]  # main station

    # The most overloaded lines, generally first lines connected to the main station
    crit_lines_feeder = relevant_lines[relevant_lines["bus0"] == station_node]

    # the last node of each feeder of the ring networks (switches are open)
    switch_df = edisgo_obj.topology.switches_df.loc[:, "bus_closed":"bus_open"].values
    switches = [node for last_nodes in switch_df for node in last_nodes]

    if isinstance(grid, LVGrid):
        nodes = G
    else:
        nodes = switches
        # for the radial feeders in MV grid
        for node in G.nodes:
            if node in crit_lines.index.values:
                nodes.append(node)

    paths = {}
    nodes_feeder = {}
    for node in nodes:
        # paths for the open and closed sides of CBs
        path = nx.shortest_path(G, station_node, node)
        for first_node in crit_lines_feeder.bus1.values:
            if first_node in path:
                paths[node] = path
                nodes_feeder.setdefault(path[1], []).append(
                    node
                )  # key:first_node values:nodes in the critical feeder

    lines_changes = {}

    for node_feeder, node_list in nodes_feeder.items():
        feeder_first_line = crit_lines_feeder[
            crit_lines_feeder.bus1 == node_feeder
        ].index[0]
        farthest_node = node_list[-1]

        path_length_dict_tmp = dijkstra_shortest_path_length(
            G, station_node, get_weight, target=farthest_node
        )
        path = paths[farthest_node]

        node_1_2 = next(
            j
            for j in path
            if path_length_dict_tmp[j] >= path_length_dict_tmp[farthest_node] * 1 / 2
        )

        # if LVGrid: check if node_1_2 is outside a house
        # and if not find next BranchTee outside the house
        if isinstance(grid, LVGrid):
            while (
                ~np.isnan(grid.buses_df.loc[node_1_2].in_building)
                and grid.buses_df.loc[node_1_2].in_building
            ):
                node_1_2 = path[path.index(node_1_2) - 1]
                # break if node is station
                if node_1_2 is path[0]:
                    logger.error(
                        f" {feeder_first_line} and following lines could not "
                        f"be reinforced due to insufficient number of node . "
                    )
                    break

        # if MVGrid: check if node_1_2 is LV station and if not find
        # next or preceding LV station
        else:
            while node_1_2 not in edisgo_obj.topology.transformers_df.bus0.values:
                try:
                    node_1_2 = path[path.index(node_1_2) + 1]
                except IndexError:
                    while (
                        node_1_2 not in edisgo_obj.topology.transformers_df.bus0.values
                    ):
                        if path.index(node_1_2) > 1:
                            node_1_2 = path[path.index(node_1_2) - 1]
                        else:
                            logger.error(
                                f" {feeder_first_line} and following lines could not "
                                f"be reinforced due to the lack of LV station . "
                            )
                            break

        # if node_1_2 is a representative (meaning it is already directly connected
        # to the station), line cannot be disconnected and reinforced
        if node_1_2 not in nodes_feeder.keys():
            # get line between node_1_2 and predecessor node
            pred_node = path[path.index(node_1_2) - 1]
            line_removed = G.get_edge_data(node_1_2, pred_node)["branch_name"]

            # note:line between node_1_2 and pred_node is not removed and the connection
            # points of line ,changed from the node to main station,  is changed.
            # Therefore, the line connected to the main station has the same name
            # with the line to be removed.
            # todo: the name of added line should be
            #  created and name of removed line should be deleted from the lines_df

            # change the connection of the node_1_2 from pred node to main station
            if grid.lines_df.at[line_removed, "bus0"] == pred_node:

                edisgo_obj.topology._lines_df.at[line_removed, "bus0"] = station_node
                logger.info(
                    f"==> {grid}--> the line {line_removed} disconnected from  "
                    f"{pred_node} and connected to the main station {station_node} "
                )
            elif grid.lines_df.at[line_removed, "bus1"] == pred_node:

                edisgo_obj.topology._lines_df.at[line_removed, "bus1"] = station_node
                logger.info(
                    f"==> {grid}-->the line {line_removed} disconnected from "
                    f"{pred_node} and connected to the main station  {station_node} "
                )
            else:
                raise ValueError("Bus not in line buses. " "Please check.")
            # change the line length
            # the properties of the added line are the same as the removed line
            edisgo_obj.topology._lines_df.at[
                line_removed, "length"
            ] = path_length_dict_tmp[node_1_2]
            line_added = line_removed
            lines_changes[line_added] = 1
    if lines_changes:
        logger.info(
            f"{len(lines_changes)} line/s are reinforced by split feeder "
            f"method in {grid}"
        )

    return lines_changes


def add_station_at_half_length(edisgo_obj, grid, crit_lines):
    """
    If the number of overloaded feeders in the LV grid is more than 2, the feeders are
    split at their half-length, and the disconnected points are connected to the
    new MV/LV station.


    1-The point at half the length of the feeders is found.
    2-The first node following this point is chosen as the point where the new
    connection will be made. This node can only be a station.
    3-This node is disconnected from the previous node and connected to a new station.
    4-New MV/LV is connected to the existing MV/LV station with a line of which length
    equals the line length between the node at the half-length (node_1_2) and its
    preceding node.

    Notes:
    -If the number of overloaded lines in the LV grid is less than 3 and the node_1_2
    is the first node after the main station, the method is not applied.
    -The name of the new grid will be the existing grid code
    (e.g. 40000) + 1001 = 400001001
    -The name of the lines in the new LV grid is the same as the grid where the nodes
    are removed
    -Except line names, all the data frames are named based on the new grid name

    Parameters
    ----------
    edisgo_obj: class:`~.EDisGo`
    grid: class:`~.network.grids.LVGrid`
    crit_lines: Dataframe containing over-loaded lines, their maximum relative
        over-loading (maximum calculated current over allowed current) and the
        corresponding time step.
        Index of the data frame is the names of the over-loaded lines.
        Columns are 'max_rel_overload' containing the maximum relative
        over-loading as float, 'time_index' containing the corresponding
        time-step the over-loading occurred in as
        :pandas:`pandas.Timestamp<Timestamp>`, and 'voltage_level' specifying
        the voltage level the line is in (either 'mv' or 'lv').

    Returns
    -------
    line_changes=    dict
        Dictionary with name of lines as keys and the corresponding number of
        lines added as values.
    transformer_changes=    dict
        Dictionary with added and removed transformers in the form::

        {'added': {'Grid_1': ['transformer_reinforced_1',
                              ...,
                              'transformer_reinforced_x'],
                   'Grid_10': ['transformer_reinforced_10']
                   }
        }
    """

    def get_weight(u, v, data):
        return data["length"]

    def create_bus_name(bus, voltage_level):

        """
        Create an LV and MV bus-bar name with the same grid_id but added "1001" that
        implies the separation

        Parameters
        ----------
        bus :eg 'BusBar_mvgd_460_lvgd_131573_LV'
        voltage_level : "mv" or "lv"

        Returns
        ----------
        bus: str New bus-bar name
        """
        if bus in edisgo_obj.topology.buses_df.index:
            bus = bus.split("_")
            grid_id_ind = bus.index(str(grid.id))
            bus[grid_id_ind] = str(grid.id) + "1001"
            if voltage_level == "lv":
                bus = "_".join([str(_) for _ in bus])
            elif voltage_level == "mv":
                bus[-1] = "MV"
                bus = "_".join([str(_) for _ in bus])
            else:
                logger.error("voltage level can only be " "mv" " or " "lv" "")
        else:
            raise IndexError("The bus is not in the dataframe")

        return bus

    def add_standard_transformer(edisgo_obj, grid, bus_lv, bus_mv):
        """
        Adds standard transformer to topology.

        Parameters
        ----------
        edisgo_obj: class:`~.EDisGo`
        grid: `~.network.grids.LVGrid`
        bus_lv: Identifier of lv bus
        bus_mv: Identifier of mv bus

        Returns
        ----------
        transformer_changes=    dict
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

        transformers_changes = {"added": {}}

        transformer_s = grid.transformers_df.iloc[0]
        new_transformer_name = transformer_s.name.split("_")
        grid_id_ind = new_transformer_name.index(str(grid.id))
        new_transformer_name[grid_id_ind] = str(grid.id) + "1001"

        transformer_s.s_nom = standard_transformer.S_nom
        transformer_s.type_info = standard_transformer.name
        transformer_s.r_pu = standard_transformer.r_pu
        transformer_s.x_pu = standard_transformer.x_pu
        transformer_s.name = "_".join([str(_) for _ in new_transformer_name])
        transformer_s.bus0 = bus_mv
        transformer_s.bus1 = bus_lv

        new_transformer_df = transformer_s.to_frame().T

        edisgo_obj.topology.transformers_df = pd.concat(
            [edisgo_obj.topology.transformers_df, new_transformer_df]
        )
        transformers_changes["added"][
            f"LVGrid_{str(grid.id)}1001"
        ] = new_transformer_df.index.tolist()
        return transformers_changes

    G = grid.graph
    station_node = list(G.nodes)[0]  # main station

    relevant_lines = edisgo_obj.topology.lines_df.loc[
        crit_lines[crit_lines.voltage_level == "lv"].index
    ]
    crit_lines_feeder = relevant_lines[relevant_lines["bus0"] == station_node]

    paths = {}
    first_nodes_feeders = {}

    for node in G:
        path = nx.shortest_path(G, station_node, node)

        for first_node in crit_lines_feeder.bus1.values:
            if first_node in path:
                paths[node] = path
                first_nodes_feeders.setdefault(path[1], []).append(
                    node  # first nodes and paths
                )

    lines_changes = {}
    transformers_changes = {}
    nodes_tb_relocated = {}  # nodes to be moved into the new grid

    # note: The number of critical lines in the Lv grid can be more than 2. However,
    # if the node_1_2 of the first feeder in the for loop is not the first node of the
    # feeder, it will add data frames even though the following feeders only 1 node
    # (node_1_2=first node of feeder). In this type of case,the number of critical lines
    # should be evaluated for the feeders whose node_1_2 s are not the first node of the
    # feeder. The first check should be done on the feeders that have fewer nodes.

    first_nodes_feeders = sorted(
        first_nodes_feeders.items(), key=lambda item: len(item[1]), reverse=False
    )
    first_nodes_feeders = dict(first_nodes_feeders)

    loop_counter = len(first_nodes_feeders)
    for first_node, nodes_feeder in first_nodes_feeders.items():
        first_line = crit_lines_feeder[crit_lines_feeder.bus1 == first_node].index[
            0
        ]  # first line of the feeder

        last_node = nodes_feeder[-1]  # the last node of the feeder
        path_length_dict_tmp = dijkstra_shortest_path_length(
            G, station_node, get_weight, target=last_node
        )  # the length of each line (the shortest path)
        path = paths[
            last_node
        ]  # path does not include the nodes branching from the node on the main path

        node_1_2 = next(
            j
            for j in path
            if path_length_dict_tmp[j] >= path_length_dict_tmp[last_node] * 1 / 2
        )
        # if LVGrid: check if node_1_2 is outside a house
        # and if not find next BranchTee outside the house
        if isinstance(grid, LVGrid):
            while (
                ~np.isnan(grid.buses_df.loc[node_1_2].in_building)
                and grid.buses_df.loc[node_1_2].in_building
            ):
                node_1_2 = path[path.index(node_1_2) - 1]
                # break if node is station
                if node_1_2 is path[0]:
                    grid.error(
                        f" {first_line} and following lines could not be reinforced "
                        f"due to insufficient number of node in the feeder . "
                    )
                    break
        loop_counter -= 1
        # if node_1_2 is a representative (meaning it is already directly connected
        # to the station), line cannot be disconnected and reinforced
        if node_1_2 not in first_nodes_feeders.keys():
            nodes_tb_relocated[node_1_2] = nodes_feeder[nodes_feeder.index(node_1_2) :]
            pred_node = path[path.index(node_1_2) - 1]  # predecessor node of node_1_2
            line_removed = G.get_edge_data(node_1_2, pred_node)[
                "branch_name"
            ]  # the line
            line_added_lv = line_removed
            lines_changes[line_added_lv] = 1
            # removed from exiting LV grid and converted to an MV line between new
            # and existing MV/LV station
        if len(nodes_tb_relocated) > 2 and loop_counter == 0:
            # Create the bus-bar name of primary and secondary side of new MV/LV station
            lv_bus_new = create_bus_name(station_node, "lv")
            mv_bus_new = create_bus_name(station_node, "mv")

            # ADD MV and LV bus
            v_nom_lv = edisgo_obj.topology.buses_df.loc[
                grid.transformers_df.bus1[0],
                "v_nom",
            ]
            v_nom_mv = edisgo_obj.topology.buses_df.loc[
                grid.transformers_df.bus0[0],
                "v_nom",
            ]

            x_bus = grid.buses_df.loc[station_node, "x"]
            y_bus = grid.buses_df.loc[station_node, "y"]

            # the new lv line id: e.g. 496021001
            lv_grid_id_new = int(str(grid.id) + "1001")
            building_bus = grid.buses_df.loc[station_node, "in_building"]

            # the distance between new and existing MV station in MV grid will be the
            # same with the distance between pred. node of node_1_2 of one of first
            # feeders to be split in LV grid

            length = (
                path_length_dict_tmp[node_1_2]
                - path_length_dict_tmp[path[path.index(node_1_2) - 1]]
            )

            # if the transformer already added, do not add bus and transformer once more
            if not transformers_changes:
                # the coordinates of new MV station (x2,y2)
                # the coordinates of existing LV station (x1,y1)
                # y1=y2, x2=x1+length/1000

                # add lv busbar
                edisgo_obj.topology.add_bus(
                    lv_bus_new,
                    v_nom_lv,
                    x=x_bus + length / 1000,
                    y=y_bus,
                    lv_grid_id=lv_grid_id_new,
                    in_building=building_bus,
                )
                # add  mv busbar
                edisgo_obj.topology.add_bus(
                    mv_bus_new,
                    v_nom_mv,
                    x=x_bus + length / 1000,
                    y=y_bus,
                    in_building=building_bus,
                )

                # ADD TRANSFORMER
                transformer_changes = add_standard_transformer(
                    edisgo_obj, grid, lv_bus_new, mv_bus_new
                )
                transformers_changes.update(transformer_changes)

                logger.debug(f"A new grid {lv_grid_id_new} added into topology")

                # ADD the MV LINE between existing and new MV station

                standard_line = edisgo_obj.config["grid_expansion_standard_equipment"][
                    f"mv_line_{int(edisgo_obj.topology.mv_grid.nominal_voltage)}kv"
                ]

                line_added_mv = edisgo_obj.topology.add_line(
                    bus0=grid.transformers_df.bus0[0],
                    bus1=mv_bus_new,
                    length=length,
                    type_info=standard_line,
                    kind="cable",
                )
                lines_changes[line_added_mv] = 1

                # changes on relocated lines to the new LV grid
                # grid_ids
                for node_1_2, nodes in nodes_tb_relocated.items():
                    edisgo_obj.topology.buses_df.loc[
                        node_1_2, "lv_grid_id"
                    ] = lv_grid_id_new
                    edisgo_obj.topology.buses_df.loc[
                        nodes, "lv_grid_id"
                    ] = lv_grid_id_new
                    # line connection of node_1_2 from the predecessor node in the
                    # existing grid to the lv side of new station
                    if edisgo_obj.topology.lines_df.bus1.isin([node_1_2]).any():
                        edisgo_obj.topology.lines_df.loc[
                            edisgo_obj.topology.lines_df.bus1 == node_1_2, "bus0"
                        ] = lv_bus_new
                    else:
                        raise LookupError(f"{node_1_2} is not in the lines dataframe")
                    logger.debug(
                        f"the node {node_1_2} is split from the line and connected to "
                        f"{lv_grid_id_new} "
                    )

            logger.info(
                f"{len(nodes_tb_relocated.keys())} feeders are removed from the grid "
                f"{grid} and located in new grid{repr(grid)+str(1001)} by split feeder+"
                f"add transformer method"
            )

    return transformers_changes, lines_changes


def optimize_cb_location(edisgo_obj, mv_grid, mode="loadgen"):
    """
    Locates the circuit breakers at the optimal position in the rings to
    reduce the difference in loading of feeders

    Parameters
    ----------
    edisgo_obj:
        class:`~.EDisGo`
    mv_grid :
        class:`~.network.grids.MVGrid`
    mode :obj:`str`
        Type of loading.
        1-'load'
        2-'loadgen'
        3-'gen'
        Default: 'loadgen'.


    Notes:According to planning principles of MV grids, a MV ring is run as two strings
    (half-rings) separated by a circuit breaker which is open at normal operation.
    Assuming a ring (route which is connected to the root node at either sides),
    the optimal position of a circuit breaker is defined as the position
    (virtual cable) between two nodes where the conveyed current is minimal on the
    route.Instead of the peak current,the peak load is used here (assuming a constant
    voltage.

    The circuit breaker will be installed to a node in the main route of the ring

    If a ring is dominated by loads (peak load > peak capacity of generators),
    only loads are used for determining the location of circuit breaker.
    If generators are prevailing (peak load < peak capacity of generators),
    only generator capacities are considered for relocation.

    Returns
    -------
    obj:`str`
    the node where the cb is located

    """
    logging.basicConfig(format=10)
    # power factor of loads and generators
    cos_phi_load = edisgo_obj.config["reactive_power_factor"]["mv_load"]
    cos_phi_feedin = edisgo_obj.config["reactive_power_factor"]["mv_gen"]

    buses_df = edisgo_obj.topology.buses_df
    lines_df = edisgo_obj.topology.lines_df
    loads_df = edisgo_obj.topology.loads_df
    generators_df = edisgo_obj.topology.generators_df
    switches_df = edisgo_obj.topology.switches_df
    transformers_df = edisgo_obj.topology.transformers_df

    station = mv_grid.station.index[0]
    graph = mv_grid.graph

    def id_mv_node(mv_node):
        """
        Returns id of mv node
        Parameters
        ----------
        mv_node:'str'
            name of node. E.g. 'BusBar_mvgd_2534_lvgd_450268_MV'

        Returns
        -------
        obj:`str`
        the id of the node. E.g '450268'
        """
        lv_bus_tranformer = transformers_df[transformers_df.bus0 == mv_node].bus1[0]
        lv_id = buses_df[buses_df.index == lv_bus_tranformer].lv_grid_id[0]
        return int(lv_id)

    def _sort_rings(remove_mv_station=True):
        """
        Sorts the nodes beginning from HV/MV station in the ring.

        Parameters
        ----------
        remove_mv_station :
            obj:`boolean`
            If True reinforcement HV/MV station is not included
            Default: True.

        Returns
        -------
            obj:'dict`
            Dictionary with name of sorted nodes in the ring
        """
        # close switches
        switches = [
            Switch(id=_, topology=edisgo_obj.topology)
            for _ in edisgo_obj.topology.switches_df.index
        ]
        switch_status = {}
        for switch in switches:
            switch_status[switch] = switch.state
            switch.close()
        # find rings in topology
        graph = edisgo_obj.topology.to_graph()
        rings = nx.cycle_basis(graph, root=station)
        if remove_mv_station:

            for r in rings:
                r.remove(station)

        # reopen switches
        for switch in switches:
            if switch_status[switch] == "open":
                switch.open()
        return rings

    def get_subtree_of_nodes(ring, graph):
        """
        Finds all nodes of a tree that is connected to main nodes in the ring and are
        (except main nodes) not part of the ring of main nodes (traversal of graph
        from main nodes excluding nodes along ring).
        Parameters
        ----------
        edisgo_obj:
            class:`~.EDisGo`
        ring:
            obj:'dict`
            Dictionary with name of sorted nodes in the ring
        graph
            networkx:`networkx.Graph<network.Graph>`

        Returns
        -------
            obj:'dict`
            index:main node
            columns: nodes of main node's tree
        """
        node_ring_d = {}
        for node in ring:

            if node == station:
                continue

            nodes_subtree = set()
            for path in nx.shortest_path(graph, node).values():
                if len(path) > 1:
                    if (path[1] not in ring) and (path[1] != station):
                        nodes_subtree.update(path[1 : len(path)])

            if len(nodes_subtree) == 0:
                node_ring_d.setdefault(node, []).append(None)
            else:
                for node_subtree in nodes_subtree:
                    node_ring_d.setdefault(node, []).append(node_subtree)

        return node_ring_d

    def _calculate_peak_load_gen(bus_node):
        """
        Cumulative peak load/generation of loads/generators connected to underlying
        MV or LV grid
        Parameters
        ----------
        bus_node:
            obj: bus_name of the node.

        Returns
        -------
            obj:'list'
            list of total generation and load of MV node
        """
        if (
            bus_node
            in buses_df[
                buses_df.index.str.contains("BusBar")
                & (~buses_df.index.str.contains("virtual"))
                & (buses_df.v_nom >= 10)
            ].index.values
        ):
            id_node = id_mv_node(bus_node)
            p_load = (
                loads_df[loads_df.index.str.contains(str(id_node))].p_set.sum()
                / cos_phi_load
            )
            p_gen = (
                generators_df[
                    generators_df.index.str.contains(str(id_node))
                ].p_nom.sum()
                / cos_phi_feedin
            )

        elif bus_node in buses_df[buses_df.index.str.contains("gen")].index.values:
            p_gen = (
                generators_df[generators_df.bus == bus_node].p_nom.sum()
                / cos_phi_feedin
            )
            p_load = loads_df[loads_df.bus == bus_node].p_set.sum() / cos_phi_feedin

        else:
            p_gen = 0
            p_load = 0

        return [p_gen, p_load]

    def _circuit_breaker(ring):
        """
        finds the circuit of the related ring
        Parameters
        ----------
        ring:
            obj:'dict`
            Dictionary with name of sorted nodes in the ring
        Returns
        -------
        obj: str
        the name of circuit breaker
        """
        circuit_breaker = []
        for node in ring:

            for switch in switches_df.bus_closed.values:
                if switch in node:
                    circuit_b = switches_df.loc[
                        switches_df.bus_closed == node, "bus_closed"
                    ].index[0]
                    circuit_breaker.append(circuit_b)
                else:
                    continue
        return circuit_breaker[0]

    def _change_dataframe(node_cb, ring):

        circuit_breaker = _circuit_breaker(ring)

        if node_cb != switches_df.loc[circuit_breaker, "bus_closed"]:

            node_existing = switches_df.loc[circuit_breaker, "bus_closed"]
            new_virtual_bus = f"virtual_{node_cb}"
            # if the adjacent node is previous circuit breaker
            if f"virtual_{node2}" in mv_grid.graph.adj[node_cb]:
                branch = mv_grid.graph.adj[node_cb][f"virtual_{node2}"]["branch_name"]
            else:
                branch = mv_grid.graph.adj[node_cb][node2]["branch_name"]
            # Switch
            # change bus0
            switches_df.loc[circuit_breaker, "bus_closed"] = node_cb
            # change bus1
            switches_df.loc[circuit_breaker, "bus_open"] = new_virtual_bus
            # change branch
            switches_df.loc[circuit_breaker, "branch"] = branch

            # Bus
            x_coord = buses_df.loc[node_cb, "x"]
            y_coord = buses_df.loc[node_cb, "y"]
            buses_df.rename(index={node_existing: new_virtual_bus}, inplace=True)
            buses_df.loc[new_virtual_bus, "x"] = x_coord
            buses_df.loc[new_virtual_bus, "y"] = y_coord

            buses_df.rename(
                index={f"virtual_{node_existing}": node_existing}, inplace=True
            )

            # Line
            lines_df.loc[
                lines_df.bus0 == f"virtual_{node_existing}", "bus0"
            ] = node_existing
            if lines_df.loc[branch, "bus0"] == node_cb:
                lines_df.loc[branch, "bus0"] = new_virtual_bus
            else:
                lines_df.loc[branch, "bus1"] = new_virtual_bus
        else:
            logging.info("The location of switch disconnector has not changed")

    rings = _sort_rings(remove_mv_station=True)
    for ring in rings:
        node_ring_dictionary = get_subtree_of_nodes(ring, graph)
        node_ring_df = pd.DataFrame.from_dict(node_ring_dictionary, orient="index")

        node_peak_d = {}
        for index, value in node_ring_df.iterrows():
            total_peak_gen = 0
            total_peak_load = 0
            if value[0] is not None:
                for v in value:
                    if v is None:
                        continue
                    # sum the load and generation of all subtree nodes
                    total_peak_gen += _calculate_peak_load_gen(v)[0]
                    total_peak_load += _calculate_peak_load_gen(v)[1]
                # sum the load and generation of nodes of subtree and tree itself
                total_peak_gen = total_peak_gen + _calculate_peak_load_gen(index)[0]
                total_peak_load = total_peak_load + _calculate_peak_load_gen(index)[1]
            else:
                total_peak_gen += _calculate_peak_load_gen(index)[0]
                total_peak_load += _calculate_peak_load_gen(index)[1]
            node_peak_d.setdefault(index, []).append(total_peak_gen)
            node_peak_d.setdefault(index, []).append(total_peak_load)
        node_peak_df = pd.DataFrame.from_dict(node_peak_d, orient="index")
        node_peak_df.rename(
            columns={0: "total_peak_gen", 1: "total_peak_load"}, inplace=True
        )

        diff_min = 10e9
        if mode == "load":
            node_peak_data = node_peak_df.total_peak_load
        elif mode == "generation":
            node_peak_data = node_peak_df.total_peak_gen
        elif mode == "loadgen":
            # is ring dominated by load or generation?
            # (check if there's more load than generation in ring or vice versa)
            if sum(node_peak_df.total_peak_load) > sum(node_peak_df.total_peak_gen):
                node_peak_data = node_peak_df.total_peak_load
            else:
                node_peak_data = node_peak_df.total_peak_gen
        else:
            raise ValueError("parameter 'mode' is invalid!")

        for ctr in range(len(node_peak_df.index)):

            # split route and calc demand difference
            route_data_part1 = sum(node_peak_data[0:ctr])
            route_data_part2 = sum(node_peak_data[ctr : len(node_peak_df.index)])

            diff = abs(route_data_part1 - route_data_part2)
            if diff <= diff_min:
                diff_min = diff
                position = ctr
            else:
                break

        # new cb location
        node_cb = node_peak_df.index[position]

        # check if node is last node of ring
        if position < len(node_peak_df.index):
            # check which branch to disconnect by determining load difference
            # of neighboring nodes
            diff2 = abs(
                sum(node_peak_data[0 : position + 1])
                - sum(node_peak_data[position + 1 : len(node_peak_data)])
            )

            if diff2 < diff_min:

                node2 = node_peak_df.index[position + 1]
            else:
                node2 = node_peak_df.index[position - 1]
        _change_dataframe(node_cb, ring)
    return node_cb
