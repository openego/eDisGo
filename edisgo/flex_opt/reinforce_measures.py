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
            crit_line = grid.lines_df.loc[crit_line_name].to_frame().T

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


def split_feeder_at_half_length(edisgo_obj, grid, crit_lines, split_mode="back"):
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
    split_mode: it determines the pathway to be searched for MV/LV station when the
    node_1_2 comes after the half-length of feeder is not a MV/LV station.
        *None: search for MV/LV station in all the nodes in the path (first back then
        forward)
        *back: search for MV/LV station in preceding nodes of node_1_2 in the path
        *forward: search for MV/LV station in latter nodes of node_1_2 in the path

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

    if isinstance(grid, LVGrid):

        voltage_level = "lv"
        relevant_lines = edisgo_obj.topology.lines_df.loc[
            crit_lines[crit_lines.voltage_level == voltage_level].index
        ]

    elif isinstance(grid, MVGrid):

        voltage_level = "mv"
        # find all the mv lines that have overloading issues in lines_df
        relevant_lines = edisgo_obj.topology.lines_df.loc[
            crit_lines[crit_lines.voltage_level == voltage_level].index
        ]
        # TODO:to be deleted after decision

    else:
        raise ValueError(f"Grid Type {type(grid)} is not supported.")

    G = grid.graph
    station_node = list(G.nodes)[0]  # main station

    # The most overloaded lines, generally first lines connected to the main station
    crit_lines_feeder = relevant_lines[relevant_lines["bus0"] == station_node]

    if isinstance(grid, LVGrid):
        nodes = G.nodes
    else:
        switches = np.concatenate(
            (
                edisgo_obj.topology.switches_df.bus_open.values,
                edisgo_obj.topology.switches_df.bus_closed.values,
            )
        )
        nodes = switches
        # todo:add radial feeders

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
        get_weight = lambda u, v, data: data["length"]  # noqa: E731
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
        # next or preceding LV station. If there is no LV station, do not split the
        # feeder
        else:
            nodes_tb_selected = [
                path[path.index(node_1_2) - ctr] for ctr in range(len(path))
            ]
            if split_mode is None:
                # the nodes in the entire path will be evaluated for has_mv/lv_station
                # first the nodes before node_1_2
                nodes_tb_selected.remove(station_node)
            elif split_mode == "back":
                # the preceding nodes of node_1_2 will be evaluated
                nodes_tb_selected = nodes_tb_selected[
                    : nodes_tb_selected.index(station_node)
                ]
            elif split_mode == "forward":
                # the latter nodes of node_1_2 will be evaluated.(node_1_2-switch)
                nodes_tb_selected = list(
                    reversed(
                        nodes_tb_selected[nodes_tb_selected.index(station_node) + 1 :]
                    )
                )
                nodes_tb_selected.insert(0, node_1_2)
            else:
                logger.error(f"{split_mode} is not a valid mode")

            while (
                node_1_2 not in nodes_feeder.keys()
                and node_1_2 not in edisgo_obj.topology.transformers_df.bus0.values
                and not len(node_1_2) == 0
            ):
                try:
                    node_1_2 = nodes_tb_selected[nodes_tb_selected.index(node_1_2) + 1]
                except IndexError:
                    logger.error(
                        f" {feeder_first_line} and following lines could not "
                        f"be reinforced due to the lack of LV station . "
                    )
                    node_1_2 = str()
                    break

        # if node_1_2 is a representative (meaning it is already directly connected
        # to the station), line cannot be disconnected and reinforced
        if node_1_2 not in nodes_feeder.keys() and not len(node_1_2) == 0:
            logger.info(
                f"==>method:split_feeder_at_half_length is running for " f"{grid}: "
            )
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
            f"{len(lines_changes)} line/s are reinforced by method: split feeder "
            f"at half-length method in {grid}"
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
        transformer_s.type_info = standard_transformer.type_info
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

    for node in G.nodes:
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
            logger.info(f"==>method:add_station_at_half_length is running for {grid}: ")
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

            length_lv = (
                path_length_dict_tmp[node_1_2]
                - path_length_dict_tmp[path[path.index(node_1_2) - 1]]
            )
            length_mv = path_length_dict_tmp[node_1_2]

            # if the transformer already added, do not add bus and transformer once more
            if not transformers_changes:
                # the coordinates of new MV station (x2,y2)
                # the coordinates of existing LV station (x1,y1)
                # y1=y2, x2=x1+length/1000

                # add lv busbar
                edisgo_obj.topology.add_bus(
                    lv_bus_new,
                    v_nom_lv,
                    x=x_bus + length_lv / 1000,
                    y=y_bus,
                    lv_grid_id=lv_grid_id_new,
                    in_building=building_bus,
                )
                # add  mv busbar
                edisgo_obj.topology.add_bus(
                    mv_bus_new,
                    v_nom_mv,
                    x=x_bus + length_mv / 1000,
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
                    length=length_mv,
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
                f"{grid} and located in new grid{repr(grid) + str(1001)} by method: "
                f"add_station_at_half_length "
            )
    if len(lines_changes) < 3:
        lines_changes = {}

    return transformers_changes, lines_changes


def relocate_circuit_breaker(edisgo_obj, mode="loadgen"):
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


    Notes: According to planning principles of MV grids, an MV ring is run as two
    strings (half-rings) separated by a circuit breaker which is open at normal
    operation. Assuming a ring (a route which is connected to the root node on either
    side),the optimal position of a circuit breaker is defined as the position
    (virtual cable) between two nodes where the conveyed current is minimal on the
    route. Instead of the peak current, the peak load is used here (assuming a constant
    voltage.

    The circuit breaker will be installed on a node in the main route of the ring.

    If a ring is dominated by loads (peak load > peak capacity of generators),
    only loads are used for determining the location of the circuit breaker.
    If generators are prevailing (peak load < peak capacity of generators),
    only generator capacities are considered for relocation.

    Returns
    -------
    obj:`str`
    the node where the cb is located

    """

    def _sort_nodes(remove_mv_station=True):
        """
        Sorts the nodes beginning from HV/MV station in the ring.

        Parameters
        ----------
        remove_mv_station :
            obj:`boolean`
            If True, reinforcement HV/MV station is not included
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
            for ring in rings:
                ring.remove(station)

        # reopen switches
        for switch in switches:
            if switch_status[switch] == "open":
                switch.open()
        return rings

    def _get_subtree_of_nodes(ring, graph):
        """
        Finds all nodes of a subtree connected to main nodes in the ring
        (except main nodes)

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
            columns: nodes of subtree
        """
        subtree_dict = {}
        for node in ring:
            # exclude main node
            if node != station:
                nodes_subtree = set()
                for path in nx.shortest_path(graph, node).values():
                    if len(path) > 1:
                        # Virtul_Busbars should not be included as it has the same
                        # characteristics as its main node. e.g. virtual_BusBar_
                        # mvgd_1056_lvgd_97722_MV =BusBar_mvgd_1056_lvgd_97722_MV
                        if (
                            (path[1] not in ring)
                            and (path[1] != station)
                            and ("virtual" not in path[1])
                        ):
                            nodes_subtree.update(path[1 : len(path)])

                if len(nodes_subtree) == 0:
                    subtree_dict.setdefault(node, []).append(None)
                else:
                    for node_subtree in nodes_subtree:
                        subtree_dict.setdefault(node, []).append(node_subtree)

        return subtree_dict

    def _get_circuit_breaker_df(ring):
        """
        Returns the circuit breaker df of the related ring

        Parameters
        ----------
        ring:
            obj:'dict`
            Dictionary with name of sorted nodes in the ring
        Returns
        -------
        obj: dict
        circuit breaker df
        """
        for node in ring:
            for cb in edisgo_obj.topology.switches_df.bus_closed.values:
                if cb in node:
                    circuit_breaker_df = edisgo_obj.topology.switches_df[
                        edisgo_obj.topology.switches_df.bus_closed == cb
                    ]

        return circuit_breaker_df

    def _change_dataframe(cb_new_closed, cb_old_df):

        # if the new cb location is not same as before
        if (
            cb_new_closed
            != edisgo_obj.topology.switches_df.loc[cb_old_df.index[0], "bus_closed"]
        ):
            # closed: the closed side of cb e.g. BusBar_mvgd_1056_lvgd_97722_MV
            # open: the open side of cb  e.g. virtual_BusBar_mvgd_1056_lvgd_97722_MV
            cb_old_closed = cb_old_df.bus_closed[0]
            cb_old_open = f"virtual_{cb_old_closed}"
            # open side of new cb
            cb_new_open = f"virtual_{cb_new_closed}"

            # create the branch
            # if the adjacent node is previous circuit breaker
            if f"virtual_{node2}" in G.adj[cb_new_closed]:
                branch = G.adj[cb_new_closed][f"virtual_{node2}"]["branch_name"]
            else:
                branch = G.adj[cb_new_closed][node2]["branch_name"]

            # Update switches_df
            # change bus0
            edisgo_obj.topology.switches_df.loc[
                cb_old_df.index[0], "bus_closed"
            ] = cb_new_closed
            # change bus1
            edisgo_obj.topology.switches_df.loc[
                cb_old_df.index[0], "bus_open"
            ] = cb_new_open
            # change branch
            edisgo_obj.topology.switches_df.loc[cb_old_df.index[0], "branch"] = branch

            # Update Buses_df
            x_coord = grid.buses_df.loc[cb_new_closed, "x"]
            y_coord = grid.buses_df.loc[cb_new_closed, "y"]
            edisgo_obj.topology.buses_df.rename(
                index={cb_old_closed: cb_new_open}, inplace=True
            )
            edisgo_obj.topology.buses_df.loc[cb_new_open, "x"] = x_coord
            edisgo_obj.topology.buses_df.loc[cb_new_open, "y"] = y_coord
            edisgo_obj.topology.buses_df.rename(
                index={cb_old_open: cb_old_closed}, inplace=True
            )

            # Update lines_df
            # convert old virtual busbar to real busbars
            if not edisgo_obj.topology.lines_df.loc[
                edisgo_obj.topology.lines_df.bus0 == cb_old_open, "bus0"
            ].empty:
                edisgo_obj.topology.lines_df.loc[
                    edisgo_obj.topology.lines_df.bus0 == cb_old_open,
                    "bus0",
                ] = cb_old_closed
            else:
                edisgo_obj.topology.lines_df.loc[
                    edisgo_obj.topology.lines_df.bus1 == cb_old_open,
                    "bus1",
                ] = cb_old_closed
            # convert the node where cb will be located from real bus-bar to virtual
            if edisgo_obj.topology.lines_df.loc[branch, "bus0"] == cb_new_closed:
                edisgo_obj.topology.lines_df.loc[branch, "bus0"] = cb_new_open
            else:
                edisgo_obj.topology.lines_df.loc[branch, "bus1"] = cb_new_open
            logging.info(f"The new location of circuit breaker is {cb_new_closed}")
        else:
            logging.info(
                f"The location of circuit breaker {cb_old_df.bus_closed[0]} "
                f"has not changed"
            )

    cos_phi_load = edisgo_obj.config["reactive_power_factor"]["mv_load"]
    cos_phi_feedin = edisgo_obj.config["reactive_power_factor"]["mv_gen"]

    grid = edisgo_obj.topology.mv_grid
    G = grid.graph
    station = list(G.nodes)[0]

    circuit_breaker_changes = {}
    node_peak_gen_dict = {}  # dictionary of peak generations of all nodes in the graph
    node_peak_load_dict = {}  # dictionary of peak loads of all nodes in the graph
    # add all the loads and gens to the dicts
    for node in G.nodes:
        # for Bus-bars
        if "BusBar" in node:
            # the lv_side of node
            if "virtual" in node:
                bus_node_lv = edisgo_obj.topology.transformers_df[
                    edisgo_obj.topology.transformers_df.bus0
                    == node.replace("virtual_", "")
                ].bus1[0]
            else:
                bus_node_lv = edisgo_obj.topology.transformers_df[
                    edisgo_obj.topology.transformers_df.bus0 == node
                ].bus1[0]
            # grid_id
            grid_id = edisgo_obj.topology.buses_df[
                edisgo_obj.topology.buses_df.index.values == bus_node_lv
            ].lv_grid_id[0]
            # get lv_grid
            lv_grid = edisgo_obj.topology.get_lv_grid(int(grid_id))

            node_peak_gen_dict[node] = (
                lv_grid.generators_df.p_nom.sum() / cos_phi_feedin
            )
            node_peak_load_dict[node] = lv_grid.loads_df.p_set.sum() / cos_phi_load

            # Generators
        elif "gen" in node:
            node_peak_gen_dict[node] = (
                edisgo_obj.topology.mv_grid.generators_df[
                    edisgo_obj.topology.mv_grid.generators_df.bus == node
                ].p_nom.sum()
                / cos_phi_feedin
            )
            node_peak_load_dict[node] = 0

        # branchTees do not have any load and generation
        else:
            node_peak_gen_dict[node] = 0
            node_peak_load_dict[node] = 0

    rings = _sort_nodes(remove_mv_station=True)
    for ring in rings:
        # nodes and subtree of these nodes
        subtree_dict = _get_subtree_of_nodes(ring, G)
        # find the peak generations and loads of nodes in the specified ring
        for node, subtree_list in subtree_dict.items():
            total_peak_gen = 0
            total_peak_load = 0
            for subtree_node in subtree_list:
                if subtree_node is not None:
                    total_peak_gen = total_peak_gen + node_peak_gen_dict[subtree_node]
                    total_peak_load = (
                        total_peak_load + node_peak_load_dict[subtree_node]
                    )

            node_peak_gen_dict[node] = total_peak_gen + node_peak_gen_dict[node]
            node_peak_load_dict[node] = total_peak_load + node_peak_load_dict[node]

        nodes_peak_load = []
        nodes_peak_generation = []

        for node in ring:
            nodes_peak_load.append(node_peak_load_dict[node])
            nodes_peak_generation.append(node_peak_gen_dict[node])

        if mode == "load":
            node_peak_data = nodes_peak_load
        elif mode == "generation":
            node_peak_data = nodes_peak_generation
        elif mode == "loadgen":
            # is ring dominated by load or generation?
            # (check if there's more load than generation in ring or vice versa)
            if sum(nodes_peak_load) > sum(nodes_peak_generation):
                node_peak_data = nodes_peak_load
            else:
                node_peak_data = nodes_peak_generation
        else:
            raise ValueError("parameter 'mode' is invalid!")

        # if none of the nodes is of the type LVStation, a switch
        # disconnecter will be installed anyways.
        if any([node for node in ring if "BusBar" in node]):
            has_lv_station = True
        else:
            has_lv_station = False
            logging.debug(
                f"Ring {ring} does not have a LV station."
                f"Switch disconnecter is installed at arbitrary "
                "node."
            )

        # calc optimal circuit breaker position
        # Set start value for difference in ring halfs
        diff_min = 10e9
        position = 0
        for ctr in range(len(node_peak_data)):
            # check if node that owns the switch disconnector is of type
            # LVStation

            if "BusBar" in ring[ctr] or not has_lv_station:
                # split route and calc demand difference
                route_data_part1 = sum(node_peak_data[0:ctr])
                route_data_part2 = sum(node_peak_data[ctr : len(node_peak_data)])
                # equality has to be respected, otherwise comparison stops when
                # demand/generation=0
                diff = abs(route_data_part1 - route_data_part2)
                if diff <= diff_min:
                    diff_min = diff
                    position = ctr
                else:
                    break

        # new cb location
        cb_new_closed = ring[position]

        # check if node is last node of ring
        if position < len(node_peak_data):
            # check which branch to disconnect by determining load difference
            # of neighboring nodes

            diff2 = abs(
                sum(node_peak_data[0 : position + 1])
                - sum(node_peak_data[position + 1 : len(node_peak_data)])
            )

            if diff2 < diff_min:
                node2 = ring[position + 1]
            else:
                node2 = ring[position - 1]
        else:
            node2 = ring[position - 1]

        cb_df_old = _get_circuit_breaker_df(ring)  # old circuit breaker df

        # update buses_df, lines_df and switches_df
        _change_dataframe(cb_new_closed, cb_df_old)

        # add number of changed circuit breakers to circuit_breaker_changes
        if cb_new_closed != cb_df_old.bus_closed[0]:
            circuit_breaker_changes[cb_df_old.index[0]] = 1

    if len(circuit_breaker_changes):
        logger.info(
            f"{len(circuit_breaker_changes)} circuit breakers are relocated in {grid}"
        )
    else:
        logger.info(f"no circuit breaker is relocated in {grid}")
    return circuit_breaker_changes


def split_feeder_at_2_3_length(edisgo_obj, grid, crit_nodes, split_mode="forward"):
    """
    The voltage issue of the lines in MV and LV grid is remedied by splitting the feeder
    at the 2/3-length

    1-The point at 2/3-length of the feeders is found.
    2-The first node following this point is chosen as the point where the new
    connection will be made. This node can only be a station.
    3-This node is disconnected from the previous node and connected to the main station

    Notes:
    In LV grids, the node inside the building is not considered.
    The method is not applied if the node is the first node after the main station.

    Parameters
    ----------
    edisgo_obj:class:`~.EDisGo`
    grid:class:`~.network.grids.MVGrid` or :class:`~.network.grids.LVGrid`
    crit_nodes:pandas:`pandas.DataFrame<DataFrame>`
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

    Dictionary with the name of lines as keys and the corresponding number of
    lines added as values.

    Notes
    -----
    In this method, the separation is done according to the farthest node of feeder

    """

    G = grid.graph
    station_node = list(G.nodes)[0]  # main station

    paths = {}
    crit_nodes_feeder = {}
    for node in crit_nodes.index:
        path = nx.shortest_path(G, station_node, node)
        paths[node] = path
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logging.error(
                f"Voltage issues at busbar in LV network {grid} "
                f"should have been solved in previous steps."
            )
        crit_nodes_feeder.setdefault(path[1], []).append(node)

    lines_changes = {}
    for repr_node in crit_nodes_feeder.keys():

        # find node farthest away
        get_weight = lambda u, v, data: data["length"]  # noqa: E731
        path_length = 0
        for c_node in crit_nodes_feeder[repr_node]:
            path_length_dict_tmp = dijkstra_shortest_path_length(
                G, station_node, get_weight, target=c_node
            )
            if path_length_dict_tmp[c_node] > path_length:
                node = c_node
                path_length = path_length_dict_tmp[c_node]
                path_length_dict = path_length_dict_tmp
        path = paths[node]

        # find first node in path that exceeds 2/3 of the line length
        # from station to critical node the farthest away from the station
        node_2_3 = next(
            j for j in path if path_length_dict[j] >= path_length_dict[node] * 2 / 3
        )
        # store the first found node_2_3
        st_node_2_3 = node_2_3
        # if LVGrid: check if node_2_3 is outside a house
        # and if not find next BranchTee outside the house
        if isinstance(grid, LVGrid):
            while (
                ~np.isnan(grid.buses_df.loc[node_2_3].in_building)
                and grid.buses_df.loc[node_2_3].in_building
            ):
                node_2_3 = path[path.index(node_2_3) - 1]
                # break if node is station
                if node_2_3 is path[0]:
                    logger.error(
                        f" line of {node_2_3} could not be reinforced due to "
                        f"insufficient number of node . "
                    )
                    break

        # if MVGrid: check if node_2_3 is LV station and if not find
        # next or preceding LV station
        else:
            nodes_tb_selected = [
                path[path.index(node_2_3) - ctr] for ctr in range(len(path))
            ]
            if split_mode is None:
                # the nodes in the entire path will be evaluated for has_mv/lv_station
                # first the latter nodes of node_2_3
                nodes_tb_selected = (
                    list(
                        reversed(
                            nodes_tb_selected[
                                nodes_tb_selected.index(station_node) + 1 :
                            ]
                        )
                    )
                    + nodes_tb_selected[: nodes_tb_selected.index(station_node)]
                )
            elif split_mode == "back":
                # the preceding nodes of node_2_3 will be evaluated
                nodes_tb_selected = nodes_tb_selected[
                    : nodes_tb_selected.index(station_node)
                ]
            elif split_mode == "forward":
                # the latter nodes of node_2_3 will be evaluated.(node_2_3-switch)
                nodes_tb_selected = list(
                    reversed(
                        nodes_tb_selected[nodes_tb_selected.index(station_node) + 1 :]
                    )
                )
                nodes_tb_selected.insert(0, node_2_3)
            else:
                logger.error(f"{split_mode} is not a valid mode")

            while (
                node_2_3 not in edisgo_obj.topology.transformers_df.bus0.values
                and not len(node_2_3) == 0
            ):
                try:
                    node_2_3 = nodes_tb_selected[nodes_tb_selected.index(node_2_3) + 1]
                except IndexError:
                    logger.error(
                        f" A lv station could not be found in the line of {node_2_3}. "
                        f"Therefore the node {st_node_2_3} will be separated from the "
                        f"feeder "
                    )
                    # instead of connecting last nodes of the feeders and reducing n-1
                    # security, install a disconnector in its current location
                    node_2_3 = st_node_2_3
                    break

        # if node_2_3 is a representative (meaning it is already
        # directly connected to the station), line cannot be
        # disconnected and must therefore be reinforced

        if node_2_3 in crit_nodes_feeder.keys():
            crit_line_name = G.get_edge_data(station_node, node_2_3)["branch_name"]
            crit_line = grid.lines_df.loc[crit_line_name:]
            # add same type of parallel line
            lines_changes = add_same_type_of_parallel_line(edisgo_obj, crit_line)

        else:
            # get line between node_2_3 and predecessor node
            pred_node = path[path.index(node_2_3) - 1]
            line_removed = G.get_edge_data(node_2_3, pred_node)["branch_name"]

            # note:line between node_2_3 and pred_node is not removed and the connection
            # points of line ,changed from the node to main station,  is changed.
            # Therefore, the line connected to the main station has the same name
            # with the line to be removed.
            # todo: the name of added line should be
            #  created and name of removed line should be deleted from the lines_df

            # change the connection of the node_2_3 from pred node to main station
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
            edisgo_obj.topology._lines_df.at[line_removed, "length"] = path_length_dict[
                node_2_3
            ]
            line_added = line_removed
            lines_changes[line_added] = 1

    if lines_changes:
        logger.info(
            f"{len(lines_changes)} line/s are reinforced by split feeder at 2/3-length "
            f"method in {grid}"
        )
    return lines_changes


def add_substation_at_2_3_length(edisgo_obj, grid, crit_nodes):
    """
    todo: docstring to be updated
    If the number of overloaded feeders in the LV grid is more than 2, the feeders are
    split at their 2/3-length, and the disconnected points are connected to the
    new MV/LV station.


    1-The point at 2/3 the length of the feeders is found.
    2-The first node following this point is chosen as the point where the new
    connection will be made. This node can only be a station.
    3-This node is disconnected from the previous node and connected to a new station.
    4-New MV/LV is connected to the existing MV/LV station with a line of which length
    equals the line length between the node at the half-length (node_2_3) and its
    preceding node.

    Notes:
    -If the number of overloaded lines in the LV grid is less than 3 and the node_2_3
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

    def _get_subtree_of_node(node, main_path):

        if node != station_node:
            nodes_subtree = set()
            for path in nx.shortest_path(G, node).values():
                if len(path) > 1:
                    if (path[1] not in main_path) and (path[1] != station_node):
                        nodes_subtree.update(path[1 : len(path)])

            return nodes_subtree

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

    paths = {}
    crit_nodes_feeder = {}
    for node in crit_nodes.index:
        path = nx.shortest_path(G, station_node, node)
        paths[node] = path
        # raise exception if voltage issue occurs at station's secondary side
        # because voltage issues should have been solved during extension of
        # distribution substations due to overvoltage issues.
        if len(path) == 1:
            logging.error(
                f"Voltage issues at busbar in LV network {grid} should have "
                "been solved in previous steps."
            )
        crit_nodes_feeder.setdefault(path[1], []).append(node)
    lines_changes = {}
    transformers_changes = {}
    nodes_tb_relocated = {}  # nodes to be moved into the new grid

    first_nodes_feeders = sorted(
        crit_nodes_feeder.items(), key=lambda item: len(item[1]), reverse=False
    )
    first_nodes_feeders = dict(first_nodes_feeders)

    loop_counter = len(first_nodes_feeders)

    for first_node, nodes_feeder in first_nodes_feeders.items():

        # find the farthest node in the feeder
        get_weight = lambda u, v, data: data["length"]  # noqa: E731

        path_length = 0
        for c_node in first_nodes_feeders[first_node]:
            path_length_dict_tmp = dijkstra_shortest_path_length(
                G, station_node, get_weight, target=c_node
            )
            if path_length_dict_tmp[c_node] > path_length:
                node = c_node
                path_length = path_length_dict_tmp[c_node]
                path_length_dict = path_length_dict_tmp
        path = paths[node]

        node_2_3 = next(
            j for j in path if path_length_dict[j] >= path_length_dict[node] * 2 / 3
        )
        # if LVGrid: check if node_2_3 is outside a house
        # and if not find next BranchTee outside the house
        if isinstance(grid, LVGrid):
            while (
                ~np.isnan(grid.buses_df.loc[node_2_3].in_building)
                and grid.buses_df.loc[node_2_3].in_building
            ):
                node_2_3 = path[path.index(node_2_3) - 1]
                # break if node is station
                if node_2_3 is path[0]:
                    grid.error(
                        f" line of {node_2_3} could not be reinforced "
                        f"due to insufficient number of node in the feeder . "
                    )
                    break

        loop_counter -= 1
        # if node_2_3 is a representative (meaning it is already directly connected
        # to the station), line cannot be disconnected and reinforced

        if node_2_3 not in first_nodes_feeders.keys():
            nodes_path = path.copy()
            for main_node in nodes_path:
                sub_nodes = _get_subtree_of_node(main_node, main_path=nodes_path)
                if sub_nodes is not None:
                    nodes_path[
                        nodes_path.index(main_node)
                        + 1 : nodes_path.index(main_node)
                        + 1
                    ] = [n for n in sub_nodes]
            nodes_tb_relocated[node_2_3] = nodes_path[nodes_path.index(node_2_3) :]
        pred_node = path[path.index(node_2_3) - 1]  # predecessor node of node_2_3

        line_removed = G.get_edge_data(node_2_3, pred_node)["branch_name"]  # the line
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
            # same with the distance between pred. node of node_2_3 of one of first
            # feeders to be split in LV grid

            length = (
                path_length_dict[node_2_3]
                - path_length_dict[path[path.index(node_2_3) - 1]]
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
                for node_2_3, nodes in nodes_tb_relocated.items():
                    edisgo_obj.topology.buses_df.loc[
                        node_2_3, "lv_grid_id"
                    ] = lv_grid_id_new
                    edisgo_obj.topology.buses_df.loc[
                        nodes, "lv_grid_id"
                    ] = lv_grid_id_new
                    # line connection of node_2_3 from the predecessor node in the
                    # existing grid to the lv side of new station
                    if edisgo_obj.topology.lines_df.bus1.isin([node_2_3]).any():
                        edisgo_obj.topology.lines_df.loc[
                            edisgo_obj.topology.lines_df.bus1 == node_2_3, "bus0"
                        ] = lv_bus_new
                    else:
                        raise LookupError(f"{node_2_3} is not in the lines dataframe")
                    logger.debug(
                        f"the node {node_2_3} is split from the line and connected to "
                        f"{lv_grid_id_new} "
                    )
            logger.info(
                f"{len(nodes_tb_relocated.keys())} feeders are removed from the grid "
                f"{grid} and located in new grid{repr(grid) + str(1001)} by split "
                f"feeder+add transformer method"
            )
    if len(lines_changes) < 3:
        lines_changes = {}

    return transformers_changes, lines_changes
