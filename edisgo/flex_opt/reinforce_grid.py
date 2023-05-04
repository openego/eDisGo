from __future__ import annotations

import copy
import datetime
import logging

from typing import TYPE_CHECKING

import pandas as pd

from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import exceptions, reinforce_measures
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.tools import tools
from edisgo.tools.temporal_complexity_reduction import get_most_critical_time_steps

if TYPE_CHECKING:
    from edisgo import EDisGo
    from edisgo.network.results import Results

logger = logging.getLogger(__name__)


def reinforce_grid(
    edisgo: EDisGo,
    timesteps_pfa: str | pd.DatetimeIndex | pd.Timestamp | None = None,
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
        In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be used
        to specify the number of most critical overloading events to consider.
        If None, `percentage` is used. Default: None.
    num_steps_voltage : int
        In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be used
        to specify the number of most critical voltage issues to select. If None,
        `percentage` is used. Default: None.
    percentage : float
        In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be used
        to specify the percentage of most critical time steps to select. The default
        is 1.0, in which case all most critical time steps are selected.
        Default: 1.0.
    use_troubleshooting_mode : bool
        In case `timesteps_pfa` is set to 'reduced_analysis', this parameter can be used
        to specify how to handle non-convergence issues in the power flow analysis.
        See parameter `use_troubleshooting_mode` in function :attr:`~.EDisGo.reinforce`
        for more information. Default: True.

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

    def _add_lines_changes_to_equipment_changes():
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
                    index=lines_changes.keys(),
                ),
            ],
        )

    def _add_transformer_changes_to_equipment_changes(mode: str | None):
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
        elif isinstance(timesteps_pfa, str) and timesteps_pfa == "reduced_analysis":
            timesteps_pfa = get_most_critical_time_steps(
                edisgo,
                num_steps_loading=kwargs.get("num_steps_loading", None),
                num_steps_voltage=kwargs.get("num_steps_voltage", None),
                percentage=kwargs.get("percentage", 1.0),
                use_troubleshooting_mode=kwargs.get("use_troubleshooting_mode", True),
            )
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
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not overloaded_lv_stations.empty:
            # reinforce distribution substations
            transformer_changes = (
                reinforce_measures.reinforce_mv_lv_station_overloading(
                    edisgo, overloaded_lv_stations
                )
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_lines_overloading(
                edisgo, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

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
            "Overloading issues could not be solved after maximum allowed "
            "iterations."
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
        _add_lines_changes_to_equipment_changes()

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
    if while_counter == max_while_iterations and crit_nodes.empty:
        edisgo.results.unresolved_issues = pd.concat(
            [
                edisgo.results.unresolved_issues,
                pd.concat([_ for _ in crit_nodes.values()]),
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
            _add_transformer_changes_to_equipment_changes("added")

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
        if while_counter == max_while_iterations and crit_stations.empty:
            edisgo.results.unresolved_issues = pd.concat(
                [
                    edisgo.results.unresolved_issues,
                    pd.concat([_ for _ in crit_stations.values()]),
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
                _add_lines_changes_to_equipment_changes()

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
        if while_counter == max_while_iterations and crit_nodes.empty:
            edisgo.results.unresolved_issues = pd.concat(
                [
                    edisgo.results.unresolved_issues,
                    pd.concat([_ for _ in crit_nodes.values()]),
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
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not overloaded_lv_stations.empty:
            # reinforce substations
            transformer_changes = (
                reinforce_measures.reinforce_mv_lv_station_overloading(
                    edisgo, overloaded_lv_stations
                )
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_lines_overloading(
                edisgo, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

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
            "Overloading issues (after solving over-voltage issues) for the"
            f"following lines could not be solved: {crit_lines}"
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
    timesteps_pfa = kwargs.get("timesteps_pfa")
    kwargs.pop("timesteps_pfa")
    selected_timesteps = timesteps_pfa

    # Initial try
    logger.info("Run initial reinforcement.")
    set_scaling_factor = 1.0
    iteration = 0
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
        reinforce()

    # Run reinforcement for time steps that did not converge after initial reinforcement
    if not non_converging_timesteps.empty:
        logger.info(
            "Run reinforcement for time steps that did not converge after initial "
            "reinforcement."
        )
        selected_timesteps = non_converging_timesteps
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
        reinforce()

    return edisgo.results


def enhanced_reinforce_grid(
    edisgo_object: EDisGo, activate_cost_results_disturbing_mode: bool = False, **kwargs
) -> EDisGo:
    """
    Reinforcement strategy to reinforce grids voltage level by voltage level in case
    grid reinforcement method
    :func:`edisgo.flex_opt.reinforce_grid.catch_convergence_reinforce_grid` is not
    sufficient.

    After first grid reinforcement for all voltage levels at once fails, reinforcement
    is first conducted for the MV level only, afterwards for the MV level including
    MV/LV stations and at last each LV grid separately.

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
    if kwargs.get("copy_grid", True):
        edisgo_obj = copy.deepcopy(edisgo_object)
    else:
        edisgo_obj = edisgo_object
    kwargs["copy_grid"] = False
    kwargs.pop("skip_mv_reinforcement", False)

    num_lv_grids_standard_lines = 0
    num_lv_grids_aggregated = 0

    try:
        logger.info("Try initial enhanced reinforcement.")
        edisgo_obj.reinforce(mode=None, catch_convergence_problems=True, **kwargs)
        logger.info("Initial enhanced reinforcement succeeded.")
    except:  # noqa: E722
        logger.info("Initial enhanced reinforcement failed.")
        logger.info("Try mode 'mv' reinforcement.")
        try:
            edisgo_obj.reinforce(mode="mv", catch_convergence_problems=True, **kwargs)
            logger.info("Mode 'mv' reinforcement succeeded.")
        except:  # noqa: E722
            logger.info("Mode 'mv' reinforcement failed.")

        logger.info("Try mode 'mvlv' reinforcement.")
        try:
            edisgo_obj.reinforce(mode="mvlv", catch_convergence_problems=True, **kwargs)
            logger.info("Mode 'mvlv' reinforcement succeeded.")
        except:  # noqa: E722
            logger.info("Mode 'mvlv' reinforcement failed.")

        for lv_grid in list(edisgo_obj.topology.mv_grid.lv_grids):
            try:
                logger.info(f"Try mode 'lv' reinforcement for {lv_grid=}.")
                edisgo_obj.reinforce(
                    mode="lv",
                    lv_grid_id=lv_grid.id,
                    catch_convergence_problems=True,
                    **kwargs,
                )
                logger.info(f"Mode 'lv' reinforcement for {lv_grid} successful.")
            except:  # noqa: E722
                logger.info(f"Mode 'lv' reinforcement for {lv_grid} failed.")
                if activate_cost_results_disturbing_mode:
                    try:
                        logger.warning(
                            f"Change all lines to standard type in {lv_grid=}."
                        )
                        num_lv_grids_standard_lines += 1
                        lv_standard_line_type = edisgo_obj.config[
                            "grid_expansion_standard_equipment"
                        ]["lv_line"]
                        edisgo_obj.topology.change_line_type(
                            lv_grid.lines_df.index.to_list(), lv_standard_line_type
                        )
                        edisgo_obj.reinforce(
                            mode="lv",
                            lv_grid_id=lv_grid.id,
                            catch_convergence_problems=True,
                            **kwargs,
                        )
                        logger.info(
                            f"Changed lines mode 'lv' for {lv_grid} successful."
                        )
                    except:  # noqa: E722
                        logger.info(f"Changed lines mode 'lv' for {lv_grid} failed.")
                        logger.warning(
                            f"Aggregate all nodes to station bus in {lv_grid=}."
                        )
                        num_lv_grids_aggregated += 1
                        try:
                            edisgo_obj.topology.aggregate_lv_grid_at_station(
                                lv_grid_id=lv_grid.id
                            )
                            logger.info(
                                f"Aggregate to station for {lv_grid} successful."
                            )
                        except:  # noqa: E722
                            logger.info(f"Aggregate to station for {lv_grid} failed.")

        try:
            edisgo_obj.reinforce(mode=None, catch_convergence_problems=True, **kwargs)
            logger.info("Enhanced reinforcement succeeded.")
        except Exception as e:  # noqa: E722
            logger.info("Enhanced reinforcement failed.")
            raise e

    if activate_cost_results_disturbing_mode is True:
        if num_lv_grids_standard_lines > 0:
            msg = (
                f"In {num_lv_grids_standard_lines} LV grid(s) all lines were "
                f"exchanged by standard lines."
            )
            logger.warning(msg)
            edisgo_obj.results.measures = msg
        else:
            msg = (
                "Enhanced reinforcement: No exchange of lines with standard lines or "
                "aggregation at MV/LV station needed."
            )
            logger.info(msg)
            edisgo_obj.results.measures = msg
        if num_lv_grids_aggregated > 0:
            msg = (
                f"Enhanced reinforcement: In {num_lv_grids_aggregated} LV grid(s) all "
                f"components were aggregated at the MV/LV station."
            )
            logger.warning(msg)
            edisgo_obj.results.measures = msg

    return edisgo_obj
