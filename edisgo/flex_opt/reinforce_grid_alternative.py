from __future__ import annotations

import copy
import datetime
import logging

import pandas as pd

from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import exceptions, reinforce_measures
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.tools import tools

logger = logging.getLogger(__name__)


def reinforce_line_overloading_alternative(
    edisgo,
    add_method=None,
    timesteps_pfa=None,
    copy_grid=False,
    voltage_level=None,
    loading_mode="load",
    split_mode="back",
    max_while_iterations=20,
    without_generator_import=False,
):
    """
    todo: docstring is to be updated
    Evaluates network reinforcement needs and performs measures.

    This function is the parent function for all network reinforcements.

    MV Grid Reinforcement:
        After circuit breakers are relocated based on the least load/gen difference
        between the feeders, the feeder is split at the half-length and connected
        to the new mv/lv station

    LV Grid Reinforcement
        If the number of overloaded lines are more than 2 in the grid, the feeder is
    split at the half-length and connected to the new mv/lv station. Otherwise,
    the feeder is split at the half-length and connected to the HV/MV station.

    MV_LV Grid Reinforcement
        The remaining overloaded lines are reinforced by add same type of parallel line
    method

    Parameters
    ----------

    edisgo: class:`~.EDisGo`
        The eDisGo API object
    add_method: The following methods can be used:
        [
            "relocate_circuit_breaker",
            "add_station_at_half_length",
            "split_feeder_at_half_length",
            "add_same_type_of_parallel_line",
        ]
    timesteps_pfa: str or \
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        timesteps_pfa specifies for which time steps power flow analysis is
        conducted and therefore which time steps to consider when checking
        for over-loading and over-voltage issues.
        It defaults to None in which case all timesteps in
        timeseries.timeindex (see :class:`~.network.network.TimeSeries`) are
        used.
        Possible options are:

        * None
          Time steps in timeseries.timeindex (see
          :class:`~.network.network.TimeSeries`) are used.
        * 'snapshot_analysis'
          Reinforcement is conducted for two worst-case snapshots. See
          :meth:`edisgo.tools.tools.select_worstcase_snapshots()` for further
          explanation on how worst-case snapshots are chosen.
          Note: If you have large time series choosing this option will save
          calculation time since power flow analysis is only conducted for two
          time steps. If your time series already represents the worst-case
          keep the default value of None because finding the worst-case
          snapshots takes some time.
        * :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
          :pandas:`pandas.Timestamp<Timestamp>`
          Use this option to explicitly choose which time steps to consider.
    copy_grid:If True reinforcement is conducted on a copied grid and discarded.
        Default: False.
    voltage_level : str
        Determines network levels reinforcement is conducted for. Specify
        * None to reinforce MV and LV network levels. None is the default.
        * 'mv' to reinforce MV network level only, neglecting MV/LV stations,
          and LV network topology. LV load and generation is aggregated per
          LV network and directly connected to the primary side of the
          respective MV/LV station.
        * 'lv' to reinforce LV networks.
    max_while_iterations : int
        Maximum number of times each while loop is conducted.
    without_generator_import: bool
        If True excludes lines that were added in the generator import to
        connect new generators to the topology from calculation of topology expansion
        costs. Default: False.
    loading_mode:
        Type of loading.
        1-'load'
        2-'loadgen'
        3-'gen'
        Default: 'load'.
    split_mode: it determines the pathway to be searched for MV/LV station when the
        node_1_2 comes after the half-length of feeder is not a MV/LV station.
        Default: back
        *None: search for MV/LV station in all the nodes in the path (first back then
        forward)
        *back: search for MV/LV station in preceding nodes of node_1_2 in the path
        *forward: search for MV/LV station in latter nodes of node_1_2 in the path

    Returns
    -------
    :class:`~.network.network.Results`
        Returns the Results object holding network expansion costs, equipment
        changes, etc.

    Assumptions
    ------
    1-The removing cost of cables are not incorporated.
    2-Line Reinforcements are done with the same type of lines as lines reinforced


    """

    def _add_lines_changes_to_equipment_changes():
        edisgo_reinforce.results.equipment_changes = pd.concat(
            [
                edisgo_reinforce.results.equipment_changes,
                pd.DataFrame(
                    {
                        "iteration_step": [iteration_step] * len(lines_changes),
                        "change": ["changed"] * len(lines_changes),
                        "equipment": edisgo_reinforce.topology.lines_df.loc[
                            lines_changes.keys(), "type_info"
                        ].values,
                        "quantity": [_ for _ in lines_changes.values()],
                    },
                    index=lines_changes.keys(),
                ),
            ],
        )

    def _add_transformer_changes_to_equipment_changes(mode: str | None):
        df_list = [edisgo_reinforce.results.equipment_changes]
        df_list.extend(
            pd.DataFrame(
                {
                    "iteration_step": [iteration_step] * len(transformer_list),
                    "change": [mode] * len(transformer_list),
                    "equipment": transformer_list,
                    "quantity": [1] * len(transformer_list),
                },
                index=[station] * len(transformer_list),
            )
            for station, transformer_list in transformer_changes[mode].items()
        )

        edisgo_reinforce.results.equipment_changes = pd.concat(df_list)

    def _add_circuit_breaker_changes_to_equipment_changes():
        edisgo_reinforce.results.equipment_changes = pd.concat(
            [
                edisgo_reinforce.results.equipment_changes,
                pd.DataFrame(
                    {
                        "iteration_step": [iteration_step]
                        * len(circuit_breaker_changes),
                        "change": ["changed"] * len(circuit_breaker_changes),
                        "equipment": edisgo_reinforce.topology.switches_df.loc[
                            circuit_breaker_changes.keys(), "type_info"
                        ].values,
                        "quantity": [_ for _ in circuit_breaker_changes.values()],
                    },
                    index=circuit_breaker_changes.keys(),
                ),
            ],
        )

    # check if provided mode is valid
    if voltage_level and voltage_level not in ["mv", "lv"]:
        raise ValueError(f"Provided mode {voltage_level} is not valid.")
    # in case reinforcement needs to be conducted on a copied graph the
    # edisgo object is deep copied
    if copy_grid is True:
        edisgo_reinforce = copy.deepcopy(edisgo)
    else:
        edisgo_reinforce = edisgo

    # edisgo_reinforce = remove_short_lines(remove_1m_end_lines(edisgo_reinforce))

    if timesteps_pfa is not None:
        if isinstance(timesteps_pfa, str) and timesteps_pfa == "snapshot_analysis":
            snapshots = tools.select_worstcase_snapshots(edisgo_reinforce)
            # drop None values in case any of the two snapshots does not exist
            timesteps_pfa = pd.DatetimeIndex(
                data=[
                    snapshots["max_residual_load"],
                    snapshots["min_residual_load"],
                ]
            ).dropna()
        # if timesteps_pfa is not of type datetime or does not contain
        # datetimes throw an error
        elif not isinstance(timesteps_pfa, datetime.datetime):
            if hasattr(timesteps_pfa, "__iter__"):
                if not all(isinstance(_, datetime.datetime) for _ in timesteps_pfa):
                    raise ValueError(
                        f"Input {timesteps_pfa} for timesteps_pfa is not valid."
                    )
            else:
                raise ValueError(
                    f"Input {timesteps_pfa} for timesteps_pfa is not valid."
                )

    methods = [
        "relocate_circuit_breaker",
        "add_station_at_half_length",
        "split_feeder_at_half_length",
        "add_same_type_of_parallel_line",
    ]

    if add_method is None:
        add_method = methods

    if isinstance(add_method, str):
        add_method = [add_method]

    if add_method and not any(method in methods for method in add_method):
        # check if provided method is valid
        raise ValueError(f"Provided method {add_method} is not valid.")

    iteration_step = 1
    # analyze_mode = None if mode == "lv" else mode

    edisgo_reinforce.analyze(timesteps=timesteps_pfa)

    # 1-REINFORCE OVERLOADED LINES
    logger.debug("==> Check line loadings.")
    crit_lines_mv = checks.mv_line_load(edisgo_reinforce)
    crit_lines_lv = checks.lv_line_load(edisgo_reinforce)
    if not any(crit_lines_mv):
        logger.info(
            f"{edisgo_reinforce.topology.mv_grid}==>there is no critical line in MV "
            f"grid "
        )
    if not any(crit_lines_lv):
        logger.info(
            f"{edisgo_reinforce.topology.mv_grid}==>there is no critical line in lv "
            f"grids "
        )
    # 1.1 Voltage level= MV
    # 1.1.1 Method:Split the feeder at the half-length of feeder (applied only once to
    # secure n-1).
    if (not voltage_level or voltage_level == "mv") and not crit_lines_mv.empty:
        if (
            add_method == ["add_station_at_half_length"]
            or "add_station_at_half_length" in add_method
        ):
            logger.error(
                f"{edisgo_reinforce.topology.mv_grid}==> method"
                f":add_station_at_half_length is only applicable for LV grids "
            )

        if "relocate_circuit_breaker" in add_method or add_method is None:
            # method-1: relocate_circuit_breaker
            logger.info(
                f"{edisgo_reinforce.topology.mv_grid}==> method:relocate circuit "
                f"breaker location is running "
            )
            circuit_breaker_changes = reinforce_measures.relocate_circuit_breaker(
                edisgo_reinforce, mode=loading_mode
            )
            # write the installation cost of CBs to results.equipment_changes
            _add_circuit_breaker_changes_to_equipment_changes()
            logger.debug("==> Run power flow analysis.")
            edisgo_reinforce.analyze(timesteps=timesteps_pfa)
            crit_lines_mv = checks.mv_line_load(edisgo_reinforce)

        if "split_feeder_at_half_length" in add_method or add_method is None:
            # method-2: split_feeder_at_half_length

            lines_changes = reinforce_measures.split_feeder_at_half_length(
                edisgo_reinforce,
                edisgo_reinforce.topology.mv_grid,
                crit_lines_mv,
                split_mode=split_mode,
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

            logger.debug("==> Run power flow analysis.")
            edisgo_reinforce.analyze(timesteps=timesteps_pfa)

    # 1.2- Voltage level= LV
    if (not voltage_level or voltage_level == "lv") and not crit_lines_lv.empty:
        if (
            add_method == ["relocate_circuit_breaker"]
            or "relocate_circuit_breaker" in add_method
        ):
            logger.error(
                " method:relocate_circuit_breaker is only applicable for MV grids"
            )

        for lv_grid in list(edisgo_reinforce.topology.mv_grid.lv_grids):

            transformer_changes = {}
            lines_changes = {}

            if "add_station_at_half_length" in add_method or add_method is None:
                # 1.2.1  Method: Split the feeder at the half-length of feeder and add
                # new station( applied only once )
                # if the number of overloaded lines is more than 2

                (
                    transformer_changes,
                    lines_changes,
                ) = reinforce_measures.add_station_at_half_length(
                    edisgo_reinforce, lv_grid, crit_lines_lv
                )

            if transformer_changes and lines_changes:
                # write changed lines and transformers to results.equipment_changes
                _add_transformer_changes_to_equipment_changes("added")
                _add_lines_changes_to_equipment_changes()
            else:
                if "split_feeder_at_half_length" in add_method or add_method is None:
                    # 1.2.2 Method:Split the feeder at the half-length of feeder
                    # (applied only once)

                    lines_changes = reinforce_measures.split_feeder_at_half_length(
                        edisgo_reinforce, lv_grid, crit_lines_lv
                    )
                    # write changed lines to results.equipment_changes
                    _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-voltage problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo_reinforce.analyze(timesteps=timesteps_pfa)

    logger.debug("==> Recheck line load.")
    crit_lines = (
        pd.DataFrame(dtype=float)
        if voltage_level == "lv"
        else checks.mv_line_load(edisgo_reinforce)
    )

    if not voltage_level or voltage_level == "lv":
        crit_lines = pd.concat(
            [
                crit_lines,
                checks.lv_line_load(edisgo_reinforce),
            ]
        )
    if "add_same_type_of_parallel_line" in add_method or add_method is None:
        # 2- Remaining crit_lines- Voltage level MV and LV
        # Method: Add same type of parallel line
        while_counter = 0
        while not crit_lines.empty and while_counter < max_while_iterations:
            if voltage_level is None:
                grid_level = "MV and LV "
            else:
                grid_level = voltage_level

            logger.info(
                f"{edisgo_reinforce.topology.mv_grid}==>method:add_same_type_of_"
                f"parallel_line is running for {grid_level} grid/s_Step{iteration_step}"
            )
            lines_changes = reinforce_measures.add_same_type_of_parallel_line(
                edisgo_reinforce, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

            # run power flow analysis again (after updating pypsa object) and check
            # if all over-voltage problems were solved
            logger.debug("==> Run power flow analysis.")
            edisgo_reinforce.analyze(timesteps=timesteps_pfa)

            logger.debug("==> Recheck line load.")
            crit_lines = (
                pd.DataFrame(dtype=float)
                if voltage_level == "lv"
                else checks.mv_line_load(edisgo_reinforce)
            )

            if not voltage_level or voltage_level == "lv":
                crit_lines = pd.concat(
                    [
                        crit_lines,
                        checks.lv_line_load(edisgo_reinforce),
                    ]
                )
            while_counter += 1
            iteration_step += +1
        # check if all load problems were solved after maximum number of
        # iterations allowed
        if while_counter == max_while_iterations and (not crit_lines.empty):
            edisgo_reinforce.results.unresolved_issues = pd.concat(
                [
                    edisgo_reinforce.results.unresolved_issues,
                    crit_lines,
                ]
            )
            raise exceptions.MaximumIterationError(
                f"{edisgo_reinforce.topology.mv_grid}==>Overloading issues could not "
                f"be solved after maximum allowed "
                "iterations."
            )
        else:
            logger.info(
                f"{edisgo_reinforce.topology.mv_grid}==> Load issues were solved in "
                f"{while_counter} iteration step(s)."
            )

    if not crit_lines.empty:
        logger.warning(
            f"{edisgo_reinforce.topology.mv_grid}==>Not all overloading issues could "
            "be solved. "
        )

    edisgo_reinforce.results.grid_expansion_costs = grid_expansion_costs(
        edisgo_reinforce, without_generator_import=without_generator_import
    )

    return edisgo_reinforce.results


def reinforce_lines_voltage_issues_alternative(
    edisgo,
    add_method=None,
    timesteps_pfa=None,
    split_mode="forward",
    copy_grid=False,
    voltage_level=None,
    max_while_iterations=20,
    combined_analysis=False,
    without_generator_import=False,
):
    """
    # Todo: To be updated
    Parameters
    ----------
    edisgo: class:`~.EDisGo`
        The eDisGo API object

    add_method: The following methods can be used:
        [
            "add_station_at_half_length",
            "split_feeder_at_half_length",
        ]
    timesteps_pfa: str or \
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        timesteps_pfa specifies for which time steps power flow analysis is
        conducted and therefore which time steps to consider when checking
        for over-loading and over-voltage issues.
        It defaults to None in which case all timesteps in
        timeseries.timeindex (see :class:`~.network.network.TimeSeries`) are
        used.
        Possible options are:

        * None
          Time steps in timeseries.timeindex (see
          :class:`~.network.network.TimeSeries`) are used.
        * 'snapshot_analysis'
          Reinforcement is conducted for two worst-case snapshots. See
          :meth:`edisgo.tools.tools.select_worstcase_snapshots()` for further
          explanation on how worst-case snapshots are chosen.
          Note: If you have large time series choosing this option will save
          calculation time since power flow analysis is only conducted for two
          time steps. If your time series already represents the worst-case
          keep the default value of None because finding the worst-case
          snapshots takes some time.
        * :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
          :pandas:`pandas.Timestamp<Timestamp>`
          Use this option to explicitly choose which time steps to consider.

    split_mode: it determines the pathway to be searched for MV/LV station when the
        node_2_3 comes after the half-length of feeder is not a MV/LV station.
        Default: Forward.
        *None: search for MV/LV station in all the nodes in the path (first back then
        forward)
        *back: search for MV/LV station in preceding nodes of node_2_3 in the path
        *forward: search for MV/LV station in latter nodes of node_2_3 in the path

    copy_grid:If True reinforcement is conducted on a copied grid and discarded.
        Default: False.
    voltage_level:
        Determines network levels reinforcement is conducted for. Specify
        * None to reinforce MV and LV network levels. None is the default.
        * 'mv' to reinforce MV network level only, neglecting MV/LV stations,
          and LV network topology. LV load and generation is aggregated per
          LV network and directly connected to the primary side of the
          respective MV/LV station.
        * 'lv' to reinforce LV networks.
    max_while_iterations : int
        Maximum number of times each while loop is conducted.
    without_generator_import: bool
        If True excludes lines that were added in the generator import to
        connect new generators to the topology from calculation of topology expansion
        costs. Default: False.


    Returns
    -------
    :class:`~.network.network.Results`
        Returns the Results object holding network expansion costs, equipment
        changes, etc.

    """

    def _add_lines_changes_to_equipment_changes():
        edisgo_reinforce.results.equipment_changes = pd.concat(
            [
                edisgo_reinforce.results.equipment_changes,
                pd.DataFrame(
                    {
                        "iteration_step": [iteration_step] * len(lines_changes),
                        "change": ["changed"] * len(lines_changes),
                        "equipment": edisgo_reinforce.topology.lines_df.loc[
                            lines_changes.keys(), "type_info"
                        ].values,
                        "quantity": [_ for _ in lines_changes.values()],
                    },
                    index=lines_changes.keys(),
                ),
            ],
        )

    def _add_transformer_changes_to_equipment_changes(mode: str | None):
        df_list = [edisgo_reinforce.results.equipment_changes]
        df_list.extend(
            pd.DataFrame(
                {
                    "iteration_step": [iteration_step] * len(transformer_list),
                    "change": [mode] * len(transformer_list),
                    "equipment": transformer_list,
                    "quantity": [1] * len(transformer_list),
                },
                index=[station] * len(transformer_list),
            )
            for station, transformer_list in transformer_changes[mode].items()
        )

        edisgo_reinforce.results.equipment_changes = pd.concat(df_list)

    # check if provided mode is valid
    if voltage_level and voltage_level not in ["mv", "lv"]:
        raise ValueError(f"Provided mode {voltage_level} is not valid.")
    # in case reinforcement needs to be conducted on a copied graph the
    # edisgo object is deep copied
    if copy_grid is True:
        edisgo_reinforce = copy.deepcopy(edisgo)
    else:
        edisgo_reinforce = edisgo

    if timesteps_pfa is not None:
        if isinstance(timesteps_pfa, str) and timesteps_pfa == "snapshot_analysis":
            snapshots = tools.select_worstcase_snapshots(edisgo_reinforce)
            # drop None values in case any of the two snapshots does not exist
            timesteps_pfa = pd.DatetimeIndex(
                data=[
                    snapshots["max_residual_load"],
                    snapshots["min_residual_load"],
                ]
            ).dropna()
        # if timesteps_pfa is not of type datetime or does not contain
        # datetimes throw an error
        elif not isinstance(timesteps_pfa, datetime.datetime):
            if hasattr(timesteps_pfa, "__iter__"):
                if not all(isinstance(_, datetime.datetime) for _ in timesteps_pfa):
                    raise ValueError(
                        f"Input {timesteps_pfa} for timesteps_pfa is not valid."
                    )
            else:
                raise ValueError(
                    f"Input {timesteps_pfa} for timesteps_pfa is not valid."
                )
    methods = [
        "split_feeder_at_2_3_length",
        "add_station_at_2_3_length",
        "add_same_type_parallel_line_voltage_issue",
    ]

    if add_method is None:
        add_method = methods

    if isinstance(add_method, str):
        add_method = [add_method]

    if add_method and not any(method in methods for method in add_method):
        # check if provided method is valid
        raise ValueError(f"Provided method {add_method} is not valid.")

    iteration_step = 1

    edisgo_reinforce.analyze(timesteps=timesteps_pfa)

    # REINFORCE BRANCHES DUE TO VOLTAGE ISSUES

    # 1.Voltage level= MV
    logger.debug(f"{edisgo_reinforce.topology.mv_grid}==>Check voltage in MV topology.")

    voltage_level_mv = "mv_lv" if combined_analysis else "mv"

    # The nodes that have voltage issue
    crit_nodes_mv = checks.mv_voltage_deviation(
        edisgo_reinforce, voltage_levels=voltage_level_mv
    )
    if not crit_nodes_mv:
        logger.info(
            f"{edisgo_reinforce.topology.mv_grid}==>there is no critical line in MV "
            f"grid"
        )

    if (not voltage_level or voltage_level == "mv") and crit_nodes_mv:
        # 1.1Method:Split the feeder at the 2_3-length of the feeder (applied several
        # times till all the voltage issues are remedied

        if add_method == ["add_station_at_2_3_length"] and voltage_level is not None:
            raise exceptions.Error(
                f"{edisgo_reinforce.topology.mv_grid}==>method"
                f":add_station_at_2_3_length is only applicable for LV "
                "grids"
            )
        elif add_method == ["add_station_at_2_3_length"] and voltage_level is None:
            logger.error(
                f"{edisgo_reinforce.topology.mv_grid}==>method"
                f":add_station_at_2_3_length is only applicable for LV grids "
            )
            while_counter = max_while_iterations
        elif "add_station_at_2_3_length" in add_method:
            logger.error(
                f"{edisgo_reinforce.topology.mv_grid}==>method"
                f":add_station_at_2_3_length is only applicable for LV grids "
            )
        if "split_feeder_at_2_3_length" in add_method:
            logger.info(
                f"{edisgo_reinforce.topology.mv_grid}==>method"
                f":split_feeder_at_2_3_length is running "
            )

            lines_changes = reinforce_measures.split_feeder_at_2_3_length(
                edisgo_reinforce,
                edisgo_reinforce.topology.mv_grid,
                crit_nodes_mv[repr(edisgo_reinforce.topology.mv_grid)],
                split_mode=split_mode,
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()
        # run power flow analysis again (after updating pypsa object) and check
        # if all over-voltage problems were solved
        logger.debug(f"{edisgo_reinforce.topology.mv_grid}==>Run power flow analysis.")
        edisgo_reinforce.analyze(timesteps=timesteps_pfa)

        logger.debug(
            f"{edisgo_reinforce.topology.mv_grid}==> Recheck voltage in MV grid."
        )
        crit_nodes_mv = checks.mv_voltage_deviation(
            edisgo_reinforce, voltage_levels=voltage_level_mv
        )

        if "add_same_type_parallel_line_voltage_issue" in add_method:
            logger.info(
                f"{edisgo_reinforce.topology.mv_grid}==>method"
                f":add_same_type_parallel_line_voltage_issue is running "
            )
            while_counter = 0
            while crit_nodes_mv and while_counter < max_while_iterations:
                lines_changes = (
                    reinforce_measures.add_same_type_parallel_line_voltage_issue(
                        edisgo_reinforce,
                        edisgo_reinforce.topology.mv_grid,
                        crit_nodes_mv[repr(edisgo_reinforce.topology.mv_grid)],
                    )
                )
                # write changed lines to results.equipment_changes
                _add_lines_changes_to_equipment_changes()

                # run power flow analysis again (after updating pypsa object)
                # and check if all over-voltage problems were solved
                logger.debug("==>Run power flow analysis.")
                edisgo_reinforce.analyze(timesteps=timesteps_pfa)

                logger.debug("==>Recheck voltage in LV grids.")
                crit_nodes_mv = checks.mv_voltage_deviation(
                    edisgo_reinforce, voltage_levels=voltage_level_mv
                )

                iteration_step += 1
                while_counter += 1

            # check if all voltage problems were solved after maximum number of
            # iterations allowed
            if while_counter == max_while_iterations and crit_nodes_mv:
                edisgo_reinforce.results.unresolved_issues = pd.concat(
                    [
                        edisgo_reinforce.results.unresolved_issues,
                        pd.concat([_ for _ in crit_nodes_mv.values()]),
                    ]
                )

                logger.info(
                    f"{edisgo_reinforce.topology.mv_grid}==>Voltage issues for the "
                    f"following nodes could not be solved in Mv grid since the the "
                    f"number of max. iteration is reached {crit_nodes_mv.keys()} "
                )
            elif not crit_nodes_mv:
                logger.info(
                    f"{edisgo_reinforce.topology.mv_grid}==>Voltage issues were solved "
                    f"in Mv grid in {iteration_step} iteration step(s). "
                )
        if any(crit_nodes_mv):
            logger.warning(
                f"{edisgo_reinforce.topology.mv_grid}==>Not all overloading issues in"
                f" MV grid could be solved. "
            )
    # 2 Voltage level= LV

    # todo: If new grid created by the method add
    #  station requires a voltage issue reinforcement, it will
    #  raise an error since the buses and lines name of the moved nodes to
    #  the new grid is not changed.

    voltage_level_lv = "mv_lv" if combined_analysis else "lv"
    logger.debug("==> Check voltage in LV grids.")

    crit_nodes_lv = checks.lv_voltage_deviation(
        edisgo_reinforce, voltage_levels=voltage_level_lv
    )
    if not crit_nodes_lv:
        logger.info(
            f"{edisgo_reinforce.topology.mv_grid}==>there is no critical line in lv "
            "grids "
        )
    if (not voltage_level or voltage_level == "lv") and crit_nodes_lv:

        for lv_grid in crit_nodes_lv:
            transformer_changes = {}
            lines_changes = {}
            # 2.1 add new station ( applied only once ) if the number of overloaded
            # lines is more than 2
            if "add_station_at_2_3_length" in add_method:

                logger.info(
                    f"{lv_grid}:==>method:add_station_at_2_3_length method is "
                    f"running "
                )
                (
                    transformer_changes,
                    lines_changes,
                ) = reinforce_measures.add_station_at_2_3_length(
                    edisgo_reinforce,
                    edisgo_reinforce.topology.get_lv_grid(lv_grid),
                    crit_nodes_lv[lv_grid],
                )
            if transformer_changes and lines_changes:
                # write changed lines and transformers to
                # results.equipment_changes
                _add_transformer_changes_to_equipment_changes("added")
                _add_lines_changes_to_equipment_changes()
            else:
                # 2.2 Method:split_feeder_at_2/3-length of feeder
                if "split_feeder_at_2_3_length" in add_method or add_method is None:
                    logger.info(
                        f"{lv_grid}:==>method:split_feeder_at_2_3_length is running"
                    )
                    lines_changes = reinforce_measures.split_feeder_at_2_3_length(
                        edisgo_reinforce,
                        edisgo_reinforce.topology.get_lv_grid(lv_grid),
                        crit_nodes_lv[lv_grid],
                        split_mode=split_mode,
                    )
                    # write changed lines to results.equipment_changes
                    _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object)
        # and check if all over-voltage problems were solved
        logger.debug("==>Run power flow analysis.")
        edisgo_reinforce.analyze(timesteps=timesteps_pfa)

        logger.debug("==>Recheck voltage in LV grids.")
        crit_nodes_lv = checks.lv_voltage_deviation(
            edisgo_reinforce, voltage_levels=voltage_level_lv
        )
        if "add_same_type_parallel_line_voltage_issue" in add_method:
            while_counter = 0
            while crit_nodes_lv and while_counter < max_while_iterations:
                for lv_grid in crit_nodes_lv:
                    logger.info(
                        f"{edisgo_reinforce.topology.mv_grid}==>method:add_same_type_of"
                        f"_parallel_line is running for LV grid/s_Step{iteration_step}"
                    )
                    lines_changes = (
                        reinforce_measures.add_same_type_parallel_line_voltage_issue(
                            edisgo_reinforce,
                            edisgo_reinforce.topology.get_lv_grid(lv_grid),
                            crit_nodes_lv[lv_grid],
                        )
                    )
                    # write changed lines to results.equipment_changes
                    _add_lines_changes_to_equipment_changes()

                # run power flow analysis again (after updating pypsa object)
                # and check if all over-voltage problems were solved
                logger.debug("==>Run power flow analysis.")
                edisgo_reinforce.analyze(timesteps=timesteps_pfa)

                logger.debug("==>Recheck voltage in LV grids.")
                crit_nodes_lv = checks.lv_voltage_deviation(
                    edisgo_reinforce, voltage_levels=voltage_level_lv
                )

                iteration_step += 1
                while_counter += 1

                # check if all load problems were solved after maximum number of
                # iterations allowed
                if while_counter == max_while_iterations and crit_nodes_lv:
                    edisgo_reinforce.results.unresolved_issues = pd.concat(
                        [
                            edisgo_reinforce.results.unresolved_issues,
                            pd.concat([_ for _ in crit_nodes_lv.values()]),
                        ]
                    )
                    raise exceptions.MaximumIterationError(
                        f"{edisgo_reinforce.topology.mv_grid}==>Overloading issues "
                        f"could not be solved after maximum allowed iterations."
                    )
                else:
                    logger.info(
                        f"{edisgo_reinforce.topology.mv_grid}==>Voltage issues in LV "
                        f"grids were solved in {while_counter} iteration step(s)."
                    )

        if any(crit_nodes_lv):
            logger.warning(
                f"{edisgo_reinforce.topology.mv_grid}==>Not all overloading issues in "
                f"LV could be solved. "
            )

    edisgo_reinforce.results.grid_expansion_costs = grid_expansion_costs(
        edisgo_reinforce, without_generator_import=without_generator_import
    )

    return edisgo_reinforce.results
