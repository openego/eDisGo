from __future__ import annotations

import datetime
import logging

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import exceptions, reinforce_measures
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.flex_opt.reinforce_measures import separate_lv_grid
from edisgo.tools import tools
from edisgo.tools.temporal_complexity_reduction import get_most_critical_time_steps

if TYPE_CHECKING:
    from edisgo import EDisGo
    from edisgo.network.results import Results

logger = logging.getLogger(__name__)


def reinforce_grid(
    edisgo: EDisGo,
    timesteps_pfa: str | pd.DatetimeIndex | pd.Timestamp | None = None,
    reduced_analysis: bool = False,
    max_while_iterations: int = 20,
    split_voltage_band: bool = True,
    mode: str | None = None,
    without_generator_import: bool = False,
    n_minus_one: bool = False,
    **kwargs,
) -> Results:
    """
    Evaluates network reinforcement needs and performs measures.

    This function is the parent function for all network reinforcements.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`
        The eDisGo object grid reinforcement is conducted on.
    timesteps_pfa : str or \
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        timesteps_pfa specifies for which time steps power flow analysis is
        conducted. See parameter `timesteps_pfa` in function :attr:`~.EDisGo.reinforce`
        for more information.
    reduced_analysis : bool
        Specifies, whether to run reinforcement on a subset of time steps that are most
        critical. See parameter `reduced_analysis` in function
        :attr:`~.EDisGo.reinforce` for more information.
    max_while_iterations : int
        Maximum number of times each while loop is conducted. Default: 20.
    split_voltage_band : bool
        If True the allowed voltage band of +/-10 percent is allocated to the different
        voltage levels MV, MV/LV and LV according to config values set in section
        `grid_expansion_allowed_voltage_deviations`. If False, the same voltage limits
        are used for all voltage levels. Be aware that this does currently not work
        correctly.
        Default: True.
    mode : str
        Determines network levels reinforcement is conducted for. See parameter
        `mode` in function :attr:`~.EDisGo.reinforce` for more information.
    without_generator_import : bool
        If True, excludes lines that were added in the generator import to connect
        new generators from calculation of network expansion costs. Default: False.
    n_minus_one : bool
        Determines whether n-1 security should be checked. Currently, n-1 security
        cannot be handled correctly, wherefore the case where this parameter is set to
        True will lead to an error being raised. Default: False.

    Other Parameters
    -----------------
    lv_grid_id : str or int or None
        LV grid id to specify the grid to check, if mode is "lv". See parameter
        `lv_grid_id` in function :attr:`~.EDisGo.reinforce` for more information.
    scale_timeseries : float or None
        If a value is given, the timeseries used in the power flow analysis are scaled
        with this factor (values between 0 and 1 will scale down the time series and
        values above 1 will scale the timeseries up). Downscaling of time series
        can be used to gradually reinforce the grid. If None, timeseries are not scaled.
        Default: None.
    skip_mv_reinforcement : bool
        If True, MV is not reinforced, even if `mode` is "mv", "mvlv" or None.
        This is used in case worst-case grid reinforcement is conducted in order to
        reinforce MV/LV stations for LV worst-cases.
        Default: False.
    num_steps_loading : int
        In case `reduced_analysis` is set to True, this parameter can be used
        to specify the number of most critical overloading events to consider.
        If None, `percentage` is used. Default: None.
    num_steps_voltage : int
        In case `reduced_analysis` is set to True, this parameter can be used
        to specify the number of most critical voltage issues to select. If None,
        `percentage` is used. Default: None.
    percentage : float
        In case `reduced_analysis` is set to True, this parameter can be used
        to specify the percentage of most critical time steps to select. The default
        is 1.0, in which case all most critical time steps are selected.
        Default: 1.0.
    use_troubleshooting_mode : bool
        In case `reduced_analysis` is set to True, this parameter can be used
        to specify how to handle non-convergence issues in the power flow analysis.
        See parameter `use_troubleshooting_mode` in function :attr:`~.EDisGo.reinforce`
        for more information. Default: True.
    run_initial_analyze : bool
        In case `reduced_analysis` is set to True, this parameter can be
        used to specify whether to run an initial analyze to determine most
        critical time steps or to use existing results. If set to False,
        `use_troubleshooting_mode` is ignored. Default: True.

    Returns
    -------
    :class:`~.network.results.Results`
        Returns the Results object holding network expansion costs, equipment
        changes, etc.

    Notes
    -----
    See :ref:`features-in-detail` for more information on how network
    reinforcement is conducted.

    """
    if n_minus_one is True:
        raise NotImplementedError("n-1 security can currently not be checked.")

    # check if provided mode is valid
    if mode and mode not in ["mv", "mvlv", "lv"]:
        raise ValueError(f"Provided mode {mode} is not a valid mode.")
    # give warning in case split_voltage_band is set to False
    if split_voltage_band is False:
        logger.warning(
            "You called the 'reinforce_grid' function with option "
            "'split_voltage_band' = False. Be aware that this does "
            "currently not work correctly and might lead to infeasible "
            "grid reinforcement."
        )

    if timesteps_pfa is not None:
        if isinstance(timesteps_pfa, str) and timesteps_pfa == "snapshot_analysis":
            snapshots = tools.select_worstcase_snapshots(edisgo)
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

    iteration_step = 1
    lv_grid_id = kwargs.get("lv_grid_id", None)
    scale_timeseries = kwargs.get("scale_timeseries", None)
    if mode == "lv" and lv_grid_id:
        analyze_mode = "lv"
    elif mode == "lv":
        analyze_mode = None
    else:
        analyze_mode = mode

    if reduced_analysis:
        timesteps_pfa = get_most_critical_time_steps(
            edisgo,
            mode=analyze_mode,
            timesteps=timesteps_pfa,
            lv_grid_id=lv_grid_id,
            scale_timeseries=scale_timeseries,
            num_steps_loading=kwargs.get("num_steps_loading", None),
            num_steps_voltage=kwargs.get("num_steps_voltage", None),
            percentage=kwargs.get("percentage", 1.0),
            use_troubleshooting_mode=kwargs.get("use_troubleshooting_mode", True),
            run_initial_analyze=kwargs.get("run_initial_analyze", True),
        )
    if timesteps_pfa is not None and len(timesteps_pfa) == 0:
        logger.debug("Zero time steps for grid reinforcement.")
        return edisgo.results

    edisgo.analyze(
        mode=analyze_mode,
        timesteps=timesteps_pfa,
        lv_grid_id=lv_grid_id,
        scale_timeseries=scale_timeseries,
    )

    # REINFORCE OVERLOADED TRANSFORMERS AND LINES
    logger.debug("==> Check station load.")
    overloaded_mv_station = (
        pd.DataFrame(dtype=float)
        if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
        else checks.hv_mv_station_max_overload(edisgo)
    )
    if lv_grid_id or (mode == "mv"):
        overloaded_lv_stations = pd.DataFrame(dtype=float)
    else:
        overloaded_lv_stations = checks.mv_lv_station_max_overload(edisgo)

    logger.debug("==> Check line load.")
    crit_lines = (
        pd.DataFrame(dtype=float)
        if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
        else checks.mv_line_max_relative_overload(edisgo)
    )
    if not mode or mode == "lv":
        crit_lines = pd.concat(
            [
                crit_lines,
                checks.lv_line_max_relative_overload(edisgo, lv_grid_id=lv_grid_id),
            ]
        )

    while_counter = 0
    while (
        not overloaded_mv_station.empty
        or not overloaded_lv_stations.empty
        or not crit_lines.empty
    ) and while_counter < max_while_iterations:
        if not overloaded_mv_station.empty:
            # reinforce substations
            transformer_changes = (
                reinforce_measures.reinforce_hv_mv_station_overloading(
                    edisgo, overloaded_mv_station
                )
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "added"
            )
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "removed"
            )

        if not overloaded_lv_stations.empty:
            # reinforce distribution substations
            transformer_changes = (
                reinforce_measures.reinforce_mv_lv_station_overloading(
                    edisgo, overloaded_lv_stations
                )
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "added"
            )
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "removed"
            )

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_lines_overloading(
                edisgo, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes(
                edisgo, lines_changes, iteration_step
            )

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-loading problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo.analyze(
            mode=analyze_mode,
            timesteps=timesteps_pfa,
            lv_grid_id=lv_grid_id,
            scale_timeseries=scale_timeseries,
        )

        logger.debug("==> Recheck station load.")
        overloaded_mv_station = (
            pd.DataFrame(dtype=float)
            if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
            else checks.hv_mv_station_max_overload(edisgo)
        )
        if mode != "mv" and (not lv_grid_id):
            overloaded_lv_stations = checks.mv_lv_station_max_overload(edisgo)

        logger.debug("==> Recheck line load.")
        crit_lines = (
            pd.DataFrame(dtype=float)
            if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
            else checks.mv_line_max_relative_overload(edisgo)
        )
        if not mode or mode == "lv":
            crit_lines = pd.concat(
                [
                    crit_lines,
                    checks.lv_line_max_relative_overload(edisgo, lv_grid_id=lv_grid_id),
                ]
            )

        iteration_step += 1
        while_counter += 1

    # check if all load problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and (
        not crit_lines.empty
        or not overloaded_mv_station.empty
        or not overloaded_lv_stations.empty
    ):
        edisgo.results.unresolved_issues = pd.concat(
            [
                edisgo.results.unresolved_issues,
                crit_lines,
                overloaded_lv_stations,
                overloaded_mv_station,
            ]
        )
        raise exceptions.MaximumIterationError(
            f"The following overloading issues could not be solved after maximum "
            f"allowed iterations: {edisgo.results.unresolved_issues}"
        )
    else:
        logger.info(
            f"==> Load issues were solved in {while_counter} iteration step(s)."
        )

    # REINFORCE BRANCHES DUE TO VOLTAGE ISSUES
    iteration_step += 1

    # solve voltage problems in MV topology
    logger.debug("==> Check voltage in MV topology.")

    crit_nodes = (
        pd.DataFrame()
        if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
        else checks.voltage_issues(
            edisgo, voltage_level="mv", split_voltage_band=split_voltage_band
        )
    )

    while_counter = 0
    while not crit_nodes.empty and while_counter < max_while_iterations:
        # reinforce lines
        lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
            edisgo,
            edisgo.topology.mv_grid,
            crit_nodes,
        )
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes(edisgo, lines_changes, iteration_step)

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-voltage problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo.analyze(
            mode=analyze_mode,
            timesteps=timesteps_pfa,
            lv_grid_id=lv_grid_id,
            scale_timeseries=scale_timeseries,
        )

        logger.debug("==> Recheck voltage in MV topology.")
        crit_nodes = checks.voltage_issues(
            edisgo, voltage_level="mv", split_voltage_band=split_voltage_band
        )

        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and not crit_nodes.empty:
        edisgo.results.unresolved_issues = pd.concat(
            [
                edisgo.results.unresolved_issues,
                crit_nodes,
            ]
        )
        raise exceptions.MaximumIterationError(
            "Over-voltage issues for the following nodes in MV topology could "
            f"not be solved: {crit_nodes}"
        )
    else:
        logger.info(
            f"==> Voltage issues in MV topology were solved in {while_counter} "
            "iteration step(s)."
        )

    # solve voltage problems at secondary side of LV stations
    if mode != "mv":
        logger.debug("==> Check voltage at secondary side of LV stations.")

        if lv_grid_id:
            crit_stations = pd.DataFrame()
        else:
            crit_stations = checks.voltage_issues(
                edisgo,
                voltage_level="mv_lv",
                split_voltage_band=split_voltage_band,
            )

        while_counter = 0
        while not crit_stations.empty and while_counter < max_while_iterations:
            # reinforce distribution substations
            transformer_changes = (
                reinforce_measures.reinforce_mv_lv_station_voltage_issues(
                    edisgo, crit_stations
                )
            )
            # write added transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "added"
            )

            # run power flow analysis again (after updating pypsa object) and
            # check if all over-voltage problems were solved
            logger.debug("==> Run power flow analysis.")
            edisgo.analyze(
                mode=analyze_mode,
                timesteps=timesteps_pfa,
                lv_grid_id=lv_grid_id,
                scale_timeseries=scale_timeseries,
            )

            logger.debug("==> Recheck voltage at secondary side of LV stations.")
            crit_stations = checks.voltage_issues(
                edisgo,
                voltage_level="mv_lv",
                split_voltage_band=split_voltage_band,
            )

            iteration_step += 1
            while_counter += 1

        # check if all voltage problems were solved after maximum number of
        # iterations allowed
        if while_counter == max_while_iterations and not crit_stations.empty:
            edisgo.results.unresolved_issues = pd.concat(
                [
                    edisgo.results.unresolved_issues,
                    crit_stations,
                ]
            )
            raise exceptions.MaximumIterationError(
                "Over-voltage issues at busbar could not be solved for the "
                f"following LV grids: {crit_stations.lv_grid_id.unique()}"
            )
        else:
            logger.info(
                "==> Voltage issues at busbars in LV grids were "
                f"solved in {while_counter} iteration step(s)."
            )

    # solve voltage problems in LV grids
    if not mode or mode == "lv":
        logger.debug("==> Check voltage in LV grids.")
        crit_nodes = checks.voltage_issues(
            edisgo,
            voltage_level="lv",
            split_voltage_band=split_voltage_band,
            lv_grid_id=lv_grid_id,
        )

        while_counter = 0
        while not crit_nodes.empty and while_counter < max_while_iterations:
            # for every topology in crit_nodes do reinforcement
            for grid_id in crit_nodes.lv_grid_id.unique():
                # reinforce lines
                lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
                    edisgo,
                    edisgo.topology.get_lv_grid(int(grid_id)),
                    crit_nodes[crit_nodes.lv_grid_id == grid_id],
                )
                # write changed lines to results.equipment_changes
                _add_lines_changes_to_equipment_changes(
                    edisgo, lines_changes, iteration_step
                )

            # run power flow analysis again (after updating pypsa object)
            # and check if all over-voltage problems were solved
            logger.debug("==> Run power flow analysis.")
            edisgo.analyze(
                mode=analyze_mode,
                timesteps=timesteps_pfa,
                lv_grid_id=lv_grid_id,
                scale_timeseries=scale_timeseries,
            )

            logger.debug("==> Recheck voltage in LV grids.")
            crit_nodes = checks.voltage_issues(
                edisgo,
                voltage_level="lv",
                split_voltage_band=split_voltage_band,
                lv_grid_id=lv_grid_id,
            )

            iteration_step += 1
            while_counter += 1

        # check if all voltage problems were solved after maximum number of
        # iterations allowed
        if while_counter == max_while_iterations and not crit_nodes.empty:
            edisgo.results.unresolved_issues = pd.concat(
                [
                    edisgo.results.unresolved_issues,
                    crit_nodes,
                ]
            )
            raise exceptions.MaximumIterationError(
                "Over-voltage issues for the following nodes in LV grids "
                f"could not be solved: {crit_nodes}"
            )
        else:
            logger.info(
                "==> Voltage issues in LV grids were solved "
                f"in {while_counter} iteration step(s)."
            )

    # RECHECK FOR OVERLOADED TRANSFORMERS AND LINES
    logger.debug("==> Recheck station load.")
    overloaded_mv_station = (
        pd.DataFrame(dtype=float)
        if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
        else checks.hv_mv_station_max_overload(edisgo)
    )
    if (lv_grid_id) or (mode == "mv"):
        overloaded_lv_stations = pd.DataFrame(dtype=float)
    else:
        overloaded_lv_stations = checks.mv_lv_station_max_overload(edisgo)

    logger.debug("==> Recheck line load.")
    crit_lines = (
        pd.DataFrame(dtype=float)
        if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
        else checks.mv_line_max_relative_overload(edisgo)
    )
    if not mode or mode == "lv":
        crit_lines = pd.concat(
            [
                crit_lines,
                checks.lv_line_max_relative_overload(edisgo, lv_grid_id=lv_grid_id),
            ]
        )

    while_counter = 0
    while (
        not overloaded_mv_station.empty
        or not overloaded_lv_stations.empty
        or not crit_lines.empty
    ) and while_counter < max_while_iterations:
        if not overloaded_mv_station.empty:
            # reinforce substations
            transformer_changes = (
                reinforce_measures.reinforce_hv_mv_station_overloading(
                    edisgo, overloaded_mv_station
                )
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "added"
            )
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "removed"
            )

        if not overloaded_lv_stations.empty:
            # reinforce substations
            transformer_changes = (
                reinforce_measures.reinforce_mv_lv_station_overloading(
                    edisgo, overloaded_lv_stations
                )
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "added"
            )
            _add_transformer_changes_to_equipment_changes(
                edisgo, transformer_changes, iteration_step, "removed"
            )

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_lines_overloading(
                edisgo, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes(
                edisgo, lines_changes, iteration_step
            )

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-loading problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo.analyze(
            mode=analyze_mode,
            timesteps=timesteps_pfa,
            lv_grid_id=lv_grid_id,
            scale_timeseries=scale_timeseries,
        )

        logger.debug("==> Recheck station load.")
        overloaded_mv_station = (
            pd.DataFrame(dtype=float)
            if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
            else checks.hv_mv_station_max_overload(edisgo)
        )
        if mode != "mv" and (not lv_grid_id):
            overloaded_lv_stations = checks.mv_lv_station_max_overload(edisgo)

        logger.debug("==> Recheck line load.")
        crit_lines = (
            pd.DataFrame(dtype=float)
            if mode == "lv" or kwargs.get("skip_mv_reinforcement", False)
            else checks.mv_line_max_relative_overload(edisgo)
        )
        if not mode or mode == "lv":
            crit_lines = pd.concat(
                [
                    crit_lines,
                    checks.lv_line_max_relative_overload(edisgo, lv_grid_id=lv_grid_id),
                ]
            )

        iteration_step += 1
        while_counter += 1

    # check if all load problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and (
        not crit_lines.empty
        or not overloaded_mv_station.empty
        or not overloaded_lv_stations.empty
    ):
        edisgo.results.unresolved_issues = pd.concat(
            [
                edisgo.results.unresolved_issues,
                crit_lines,
                overloaded_lv_stations,
                overloaded_mv_station,
            ]
        )
        raise exceptions.MaximumIterationError(
            f"The following overloading issues could not be solved after maximum "
            f"allowed iterations: {edisgo.results.unresolved_issues}"
        )
    else:
        logger.info(
            "==> Load issues were rechecked and solved "
            f"in {while_counter} iteration step(s)."
        )

    # final check 10% criteria
    voltage_dev = checks.voltage_deviation_from_allowed_voltage_limits(
        edisgo, split_voltage_band=False
    )
    voltage_dev = voltage_dev[voltage_dev != 0.0].dropna(how="all").dropna(how="all")
    if not voltage_dev.empty:
        message = "Maximum allowed voltage deviation of 10% exceeded."
        raise ValueError(message)

    # calculate topology expansion costs
    edisgo.results.grid_expansion_costs = grid_expansion_costs(
        edisgo, without_generator_import=without_generator_import
    )

    return edisgo.results


def catch_convergence_reinforce_grid(
    edisgo: EDisGo,
    **kwargs,
) -> Results:
    """
    Reinforcement strategy to reinforce grids with non-converging time steps.

    First, conducts a grid reinforcement with only converging time steps.
    Afterwards, tries to run reinforcement with all time steps that did not converge
    in the beginning. At last, if there are still time steps that do not converge,
    the feed-in and load time series are iteratively scaled and the grid reinforced,
    starting with a low grid load and scaling-up the time series until the original
    values are reached.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`
    kwargs : dict
        See parameters of function
        :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid`.

    Returns
    -------
    :class:`~.network.results.Results`
        Returns the Results object holding network expansion costs, equipment
        changes, etc.

    """

    def reinforce():
        try:
            reinforce_grid(
                edisgo,
                timesteps_pfa=selected_timesteps,
                scale_timeseries=set_scaling_factor,
                use_troubleshooting_mode=troubleshooting_mode,
                **kwargs,
            )
            converged = True
        except ValueError as e:
            if "Power flow analysis did not converge for the" in str(e):
                converged = False
            else:
                raise
        return converged

    logger.debug("Start 'catch-convergence' reinforcement.")

    # Get the timesteps from kwargs and then remove it to set it later manually
    timesteps_pfa = kwargs.pop("timesteps_pfa", None)
    selected_timesteps = timesteps_pfa
    troubleshooting_mode_set = kwargs.pop("troubleshooting_mode", True)

    # Initial try
    logger.info("Run initial reinforcement.")
    set_scaling_factor = 1.0
    iteration = 0
    troubleshooting_mode = False
    converged = reinforce()
    if converged is False:
        logger.info("Initial reinforcement did not succeed.")
    else:
        logger.info("Initial reinforcement succeeded.")
        return edisgo.results

    # Find non-converging time steps
    if kwargs.get("mode", None) == "lv" and kwargs.get("lv_grid_id", None):
        analyze_mode = "lv"
    elif kwargs.get("mode", None) == "lv":
        analyze_mode = None
    else:
        analyze_mode = kwargs.get("mode", None)
    kwargs_analyze = kwargs
    kwargs_analyze["mode"] = analyze_mode
    converging_timesteps, non_converging_timesteps = edisgo.analyze(
        timesteps=timesteps_pfa, raise_not_converged=False, **kwargs_analyze
    )
    logger.info(f"The following time steps converged: {converging_timesteps}.")
    logger.info(
        f"The following time steps did not converge: {non_converging_timesteps}."
    )

    # Run reinforcement for time steps that converged after initial reinforcement
    if not converging_timesteps.empty:
        logger.info(
            "Run reinforcement for time steps that converged after initial "
            "reinforcement."
        )
        selected_timesteps = converging_timesteps
        troubleshooting_mode = troubleshooting_mode_set
        reinforce()

    # Run reinforcement for time steps that did not converge after initial reinforcement
    if not non_converging_timesteps.empty:
        logger.info(
            "Run reinforcement for time steps that did not converge after initial "
            "reinforcement."
        )
        selected_timesteps = non_converging_timesteps
        troubleshooting_mode = False
        converged = reinforce()

    if converged:
        return edisgo.results

    # Run iterative grid reinforcement
    else:
        max_iterations = 10
        highest_converged_scaling_factor = 0
        minimal_scaling_factor = 0.05
        while iteration < max_iterations:
            iteration += 1
            if converged:
                if set_scaling_factor == 1:
                    # reinforcement for scaling factor of 1 worked - finished
                    break
                else:
                    # if converged, try again with scaling factor of 1
                    highest_converged_scaling_factor = set_scaling_factor
                    set_scaling_factor = 1
            else:
                if set_scaling_factor == minimal_scaling_factor:
                    raise ValueError(
                        f"Not reinforceable with {minimal_scaling_factor=}!"
                    )
                elif iteration == 1:
                    set_scaling_factor = minimal_scaling_factor
                else:
                    set_scaling_factor = (
                        (set_scaling_factor - highest_converged_scaling_factor) * 0.25
                    ) + highest_converged_scaling_factor

            logger.info(f"Try reinforcement with {set_scaling_factor=} at {iteration=}")
            troubleshooting_mode = False
            converged = reinforce()
            if converged:
                logger.info(
                    f"Reinforcement succeeded for {set_scaling_factor=} "
                    f"at {iteration=}."
                )
            else:
                logger.info(
                    f"Reinforcement failed for {set_scaling_factor=} at {iteration=}."
                )

        if converged is False:
            raise ValueError(
                f"Not reinforceable, max iterations ({max_iterations}) reached!"
            )

    # Final reinforcement
    if set_scaling_factor != 1:
        logger.info("Run final reinforcement.")
        selected_timesteps = timesteps_pfa
        troubleshooting_mode = False
        reinforce()

    return edisgo.results


def enhanced_reinforce_grid(
    edisgo_object: EDisGo,
    activate_cost_results_disturbing_mode: bool = False,
    separate_lv_grids: bool = True,
    separation_threshold: int | float = 2,
    use_standard_line_type: bool = True,
    **kwargs,
) -> EDisGo:
    """
    Reinforcement strategy to reinforce grids voltage level by voltage level in case
    grid reinforcement method
    :func:`edisgo.flex_opt.reinforce_grid.catch_convergence_reinforce_grid` is not
    sufficient.

    In a first step, if `separate_lv_grids` is set to True, LV grids with a large load,
    specified through parameter `separation_threshold`, are split, so that part of the
    load is served by a separate MV/LV station. See
    :func:`~.flex_opt.reinforce_grid.run_separate_lv_grids` for more information.
    In a second step, all LV grids are reinforced independently.
    Afterwards it is tried to run the grid reinforcement for all voltage levels at once.
    If this fails, reinforcement is first conducted for the MV level only, afterwards
    for the MV level including MV/LV stations and at last for each LV grid separately.
    For each LV grid is it checked, if all time steps converge in the power flow
    analysis. If this is not the case, the grid is split. Afterwards it is tried to
    be reinforced. If this fails and `activate_cost_results_disturbing_mode`
    parameter is set to True, further measures are taken. See parameter documentation
    for more information.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    activate_cost_results_disturbing_mode : bool
        If True, LV grids where normal grid reinforcement does not solve all issues,
        two additional approaches are used to obtain a grid where power flow can be
        conducted without non-convergence. These two approaches are currently not
        included in the calculation of grid reinforcement costs, wherefore grid
        reinforcement costs will be underestimated.
        In the first approach, all lines in the LV grid are replaced by the
        standard line type. Should this not be sufficient to solve non-convergence
        issues, all components in the LV grid are aggregated to the MV/LV station.
        Default: False.
    separate_lv_grids : bool
        If True, all highly overloaded LV grids are separated in a first step.
    separation_threshold : int or float
        Overloading threshold for LV grid separation. If the overloading is higher than
        the threshold times the total nominal apparent power of the MV/LV transformer(s)
        the grid is separated.
    kwargs : dict
        Keyword arguments can be all parameters of function
        :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid`, except
        `catch_convergence_problems` which will always be set to True, `mode` which
        is set to None, and `skip_mv_reinforcement` which will be ignored.

    Returns
    -------
    :class:`~.EDisGo`
        The reinforced eDisGo object.

    """
    kwargs.pop("skip_mv_reinforcement", False)
    # ToDo kwargs timesteps_pfa is currently ignored, should that be changed?

    num_lv_grids_standard_lines = 0
    num_lv_grids_aggregated = 0

    if separate_lv_grids:
        logger.info(
            "Separating lv grids. Set the parameter 'separate_lv_grids' to False if "
            "this is not desired."
        )
        run_separate_lv_grids(edisgo_object, threshold=separation_threshold)

    logger.info("Run initial grid reinforcement for single LV grids.")
    for lv_grid in list(edisgo_object.topology.mv_grid.lv_grids):
        logger.info(f"Check initial convergence for {lv_grid=}.")
        ts_converged, ts_not_converged = edisgo_object.analyze(
            mode="lv", raise_not_converged=False, lv_grid_id=lv_grid.id
        )
        if len(ts_not_converged) > 0:
            logger.info(
                f"Not all time steps converged in initial power flow analysis for "
                f"{lv_grid=}. It is therefore tried to be split."
            )
            transformers_changes, lines_changes = separate_lv_grid(
                edisgo_object, lv_grid
            )
            if len(lines_changes) > 0:
                _add_lines_changes_to_equipment_changes(edisgo_object, lines_changes, 1)
            if len(transformers_changes) > 0:
                _add_transformer_changes_to_equipment_changes(
                    edisgo_object, transformers_changes, 1, "added"
                )
        try:
            logger.info(f"Try initial mode 'lv' reinforcement for {lv_grid=}.")
            if len(ts_not_converged) > 0:
                # if there are time steps that did not converge, run reinforcement
                # first on converged time steps
                edisgo_object.reinforce(
                    mode="lv",
                    lv_grid_id=lv_grid.id,
                    catch_convergence_problems=False,
                    timesteps_pfa=ts_converged,
                    **kwargs,
                )
                # run reinforcement again in catch-convergence mode with all time steps
                edisgo_object.reinforce(
                    mode="lv",
                    lv_grid_id=lv_grid.id,
                    catch_convergence_problems=True,
                    **kwargs,
                )
            else:
                # if all time steps converged, run normal reinforcement
                edisgo_object.reinforce(
                    mode="lv",
                    lv_grid_id=lv_grid.id,
                    catch_convergence_problems=False,
                    run_initial_analyze=False,
                    **kwargs,
                )
            logger.info(f"Initial mode 'lv' reinforcement for {lv_grid} successful.")
        except (ValueError, RuntimeError, exceptions.MaximumIterationError):
            logger.warning(f"Initial mode 'lv' reinforcement for {lv_grid} failed.")

    try:
        logger.info("Try initial enhanced reinforcement.")
        edisgo_object.reinforce(mode=None, catch_convergence_problems=True, **kwargs)
        logger.info("Initial enhanced reinforcement succeeded.")
    except (ValueError, RuntimeError, exceptions.MaximumIterationError):
        logger.info("Initial enhanced reinforcement failed.")
        logger.info("Try mode 'mv' reinforcement.")

        try:
            edisgo_object.reinforce(
                mode="mv", catch_convergence_problems=True, **kwargs
            )
            logger.info("Mode 'mv' reinforcement succeeded.")
        except (ValueError, RuntimeError, exceptions.MaximumIterationError):
            logger.info("Mode 'mv' reinforcement failed.")

        logger.info("Try mode 'mvlv' reinforcement.")

        try:
            edisgo_object.reinforce(
                mode="mvlv", catch_convergence_problems=True, **kwargs
            )
            logger.info("Mode 'mvlv' reinforcement succeeded.")
        except (ValueError, RuntimeError, exceptions.MaximumIterationError):
            logger.info("Mode 'mvlv' reinforcement failed.")

        for lv_grid in list(edisgo_object.topology.mv_grid.lv_grids):
            logger.info(f"Check convergence for {lv_grid=}.")
            ts_converged, ts_not_converged = edisgo_object.analyze(
                mode="lv", raise_not_converged=False, lv_grid_id=lv_grid.id
            )
            if len(ts_not_converged) > 0:
                logger.info(
                    f"Not all time steps converged in power flow analysis for "
                    f"{lv_grid=}. It is therefore tried to be split."
                )
                transformers_changes, lines_changes = separate_lv_grid(
                    edisgo_object, lv_grid
                )
                if len(lines_changes) > 0:
                    _add_lines_changes_to_equipment_changes(
                        edisgo_object, lines_changes, 1
                    )
                if len(transformers_changes) > 0:
                    _add_transformer_changes_to_equipment_changes(
                        edisgo_object, transformers_changes, 1, "added"
                    )
            try:
                logger.info(f"Try mode 'lv' reinforcement for {lv_grid=}.")
                if len(ts_not_converged) > 0:
                    # if there are time steps that did not converge, run reinforcement
                    # first on converged time steps
                    edisgo_object.reinforce(
                        mode="lv",
                        lv_grid_id=lv_grid.id,
                        catch_convergence_problems=False,
                        timesteps_pfa=ts_converged,
                        **kwargs,
                    )
                    # run reinforcement again in catch-convergence mode with all time
                    # steps
                    edisgo_object.reinforce(
                        mode="lv",
                        lv_grid_id=lv_grid.id,
                        catch_convergence_problems=True,
                        **kwargs,
                    )
                else:
                    # if all time steps converged, run normal reinforcement
                    edisgo_object.reinforce(
                        mode="lv",
                        lv_grid_id=lv_grid.id,
                        catch_convergence_problems=False,
                        run_initial_analyze=False,
                        **kwargs,
                    )
                logger.info(f"Mode 'lv' reinforcement for {lv_grid} successful.")
            except (ValueError, RuntimeError, exceptions.MaximumIterationError):
                logger.info(f"Mode 'lv' reinforcement for {lv_grid} failed.")
                if activate_cost_results_disturbing_mode:
                    try:
                        logger.warning(
                            f"Change all lines to standard type in {lv_grid=}."
                        )
                        edisgo_object.results.measures = (
                            f"Standard lines in {lv_grid=}."
                        )
                        num_lv_grids_standard_lines += 1
                        lv_standard_line_type = edisgo_object.config[
                            "grid_expansion_standard_equipment"
                        ]["lv_line"]
                        lines = lv_grid.lines_df.index
                        edisgo_object.topology.change_line_type(
                            lines, lv_standard_line_type
                        )
                        lines_changes = {_: 1 for _ in lines}
                        _add_lines_changes_to_equipment_changes(
                            edisgo_object, lines_changes, 1
                        )
                        edisgo_object.reinforce(
                            mode="lv",
                            lv_grid_id=lv_grid.id,
                            catch_convergence_problems=True,
                            **kwargs,
                        )
                        logger.info(
                            f"Changed lines mode 'lv' for {lv_grid} successful."
                        )
                    except (ValueError, RuntimeError, exceptions.MaximumIterationError):
                        logger.info(f"Changed lines mode 'lv' for {lv_grid} failed.")
                        logger.warning(
                            f"Aggregate all nodes to station bus in {lv_grid=}."
                        )
                        edisgo_object.results.measures = f"Aggregation of {lv_grid=}."
                        num_lv_grids_aggregated += 1
                        try:
                            edisgo_object.topology.aggregate_lv_grid_at_station(
                                lv_grid_id=lv_grid.id
                            )
                            logger.info(
                                f"Aggregate to station for {lv_grid} successful."
                            )
                        except Exception as e:
                            logger.info(
                                f"Aggregate to station for {lv_grid} failed with "
                                f"exception:\n{e}"
                            )
                            raise e

        try:
            edisgo_object.reinforce(
                mode=None, catch_convergence_problems=True, **kwargs
            )
            logger.info("Enhanced reinforcement succeeded.")
        except Exception as e:
            logger.info("Enhanced reinforcement failed.")
            raise e

    if activate_cost_results_disturbing_mode is True:
        if num_lv_grids_standard_lines > 0:
            msg = (
                f"In {num_lv_grids_standard_lines} LV grid(s) all lines were "
                f"exchanged by standard lines."
            )
            logger.warning(msg)
            edisgo_object.results.measures = msg
        else:
            msg = (
                "Enhanced reinforcement: No exchange of lines with standard lines or "
                "aggregation at MV/LV station needed."
            )
            logger.info(msg)
            edisgo_object.results.measures = msg
        if num_lv_grids_aggregated > 0:
            msg = (
                f"Enhanced reinforcement: In {num_lv_grids_aggregated} LV grid(s) all "
                f"components were aggregated at the MV/LV station."
            )
            logger.warning(msg)
            edisgo_object.results.measures = msg

    return edisgo_object


def run_separate_lv_grids(edisgo_obj: EDisGo, threshold: int | float = 2) -> None:
    """
    Separate all highly overloaded LV grids within the MV grid.

    The loading is approximated by aggregation of all load and generator time series
    and comparison with the total nominal apparent power of the MV/LV transformer(s).
    This approach is chosen because this method aims at resolving highly overloaded
    grid situations in which cases the power flow often does not converge. This method
    ignores grid losses and voltage deviations. Original and new LV grids can be
    separated multiple times if the overloading is very high.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    threshold : int or float
        Overloading threshold. If the overloading is higher than the threshold times
        the total nominal apparent power of the MV/LV transformer(s), the grid is
        separated.

    Returns
    -------
    :class:`~.EDisGo`
        The reinforced eDisGo object.

    """
    lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)
    n_grids_init = len(lv_grids)

    first_run = True

    active_str = "{}_active_power"
    reactive_str = "{}_reactive_power"
    tech_str = "{}_df"
    techs = ["generators", "loads", "storage_units"]

    n = 0
    max_iterations = 100

    while (
        n_grids_init != len(list(edisgo_obj.topology.mv_grid.lv_grids)) or first_run
    ) and n < max_iterations:
        n += 1
        first_run = False

        lv_grids = list(edisgo_obj.topology.mv_grid.lv_grids)
        n_grids_init = len(lv_grids)

        for lv_grid in lv_grids:
            active_power_dict = {}
            reactive_power_dict = {}

            for tech in techs:
                units = getattr(lv_grid, tech_str.format(tech)).index
                active_power_dict[tech] = (
                    getattr(
                        edisgo_obj.timeseries,
                        active_str.format(tech),
                    )
                    .loc[:, units]
                    .astype(float)
                    .sum(axis=1)
                )

                reactive_power_dict[tech] = (
                    getattr(
                        edisgo_obj.timeseries,
                        reactive_str.format(tech),
                    )
                    .loc[:, units]
                    .astype(float)
                    .sum(axis=1)
                )

            active_power = (
                active_power_dict["loads"]
                - active_power_dict["generators"]
                - active_power_dict["storage_units"]
            )

            reactive_power = (
                reactive_power_dict["loads"]
                - reactive_power_dict["generators"]
                - reactive_power_dict["storage_units"]
            )

            worst_case = np.hypot(active_power, reactive_power).max()

            transformers_s_nom = lv_grid.transformers_df.s_nom.sum()

            if worst_case > threshold * transformers_s_nom:
                logger.info(f"Trying to separate {lv_grid}...")
                transformers_changes, lines_changes = separate_lv_grid(
                    edisgo_obj, lv_grid
                )
                if len(lines_changes) > 0:
                    _add_lines_changes_to_equipment_changes(
                        edisgo_obj, lines_changes, 1
                    )

                if len(transformers_changes) > 0:
                    _add_transformer_changes_to_equipment_changes(
                        edisgo_obj, transformers_changes, 1, "added"
                    )

            else:
                logger.debug(
                    f"The overloading in {lv_grid} does not surpass the set threshold "
                    f"of {threshold}. The grid is therefore not separated."
                )


def _add_lines_changes_to_equipment_changes(
    edisgo: EDisGo, lines_changes: dict, iteration_step: int
) -> None:
    edisgo.results.equipment_changes = pd.concat(
        [
            edisgo.results.equipment_changes,
            pd.DataFrame(
                {
                    "iteration_step": [iteration_step] * len(lines_changes),
                    "change": ["changed"] * len(lines_changes),
                    "equipment": edisgo.topology.lines_df.loc[
                        lines_changes.keys(), "type_info"
                    ].values,
                    "quantity": [_ for _ in lines_changes.values()],
                },
                index=list(lines_changes.keys()),
            ),
        ],
    )


def _add_transformer_changes_to_equipment_changes(
    edisgo: EDisGo, transformer_changes: dict, iteration_step: int, mode: str | None
) -> None:
    df_list = [edisgo.results.equipment_changes]
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

    edisgo.results.equipment_changes = pd.concat(df_list)
