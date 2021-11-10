import pandas as pd
import copy
import datetime
from edisgo.flex_opt import check_tech_constraints as checks
from edisgo.flex_opt import reinforce_measures, exceptions
from edisgo.flex_opt.costs import grid_expansion_costs
from edisgo.tools import tools
import logging

logger = logging.getLogger("edisgo")


def reinforce_grid(
    edisgo,
    timesteps_pfa=None,
    copy_grid=False,
    max_while_iterations=10,
    combined_analysis=False,
    mode=None,
):
    """
    Evaluates network reinforcement needs and performs measures.

    This function is the parent function for all network reinforcements.

    Parameters
    ----------
    edisgo : :class:`~.EDisGo`
        The eDisGo API object
    timesteps_pfa : :obj:`str` or \
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
        :pandas:`pandas.Timestamp<Timestamp>`
        timesteps_pfa specifies for which time steps power flow analysis is
        conducted and therefore which time steps to consider when checking
        for over-loading and over-voltage issues.
        It defaults to None in which case all timesteps in
        timeseries.timeindex (see :class:`~.network.network.TimeSeries`) are used.
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

    copy_grid : :obj:`Boolean`
        If True reinforcement is conducted on a copied grid and discarded.
        Default: False.
    max_while_iterations : :obj:`int`
        Maximum number of times each while loop is conducted.
    combined_analysis : :obj:`Boolean`
        If True allowed voltage deviations for combined analysis of MV and LV
        topology are used. If False different allowed voltage deviations for MV
        and LV are used. See also config section
        `grid_expansion_allowed_voltage_deviations`. If `mode` is set to 'mv'
        `combined_analysis` should be False. Default: False.
    mode : :obj:`str`
        Determines network levels reinforcement is conducted for. Specify

        * None to reinforce MV and LV network levels. None is the default.
        * 'mv' to reinforce MV network level only, neglecting MV/LV stations,
          and LV network topology. LV load and generation is aggregated per
          LV network and directly connected to the primary side of the
          respective MV/LV station.
        * 'mvlv' to reinforce MV network level only, including MV/LV stations,
          and neglecting LV network topology. LV load and generation is
          aggregated per LV network and directly connected to the secondary
          side of the respective MV/LV station.

    Returns
    -------
    :class:`~.network.network.Results`
        Returns the Results object holding network expansion costs, equipment
        changes, etc.

    Notes
    -----
    See :ref:`features-in-detail` for more information on how network
    reinforcement is conducted.

    """

    def _add_lines_changes_to_equipment_changes():
        edisgo_reinforce.results.equipment_changes = edisgo_reinforce.results.equipment_changes.append(
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
            )
        )

    def _add_transformer_changes_to_equipment_changes(mode):
        for station, transformer_list in transformer_changes[mode].items():
            edisgo_reinforce.results.equipment_changes = edisgo_reinforce.results.equipment_changes.append(
                pd.DataFrame(
                    {
                        "iteration_step": [iteration_step]
                        * len(transformer_list),
                        "change": [mode] * len(transformer_list),
                        "equipment": transformer_list,
                        "quantity": [1] * len(transformer_list),
                    },
                    index=[station] * len(transformer_list),
                )
            )

    # check if provided mode is valid
    if mode and mode not in ["mv", "mvlv"]:
        raise ValueError("Provided mode {} is not a valid mode.".format(mode))

    # in case reinforcement needs to be conducted on a copied graph the
    # edisgo object is deep copied
    if copy_grid is True:
        edisgo_reinforce = copy.deepcopy(edisgo)
    else:
        edisgo_reinforce = edisgo

    if timesteps_pfa is not None:
        # if timesteps_pfa = 'snapshot_analysis' get snapshots
        if (
            isinstance(timesteps_pfa, str)
            and timesteps_pfa == "snapshot_analysis"
        ):
            snapshots = tools.select_worstcase_snapshots(edisgo_reinforce)
            # drop None values in case any of the two snapshots does not exist
            timesteps_pfa = pd.DatetimeIndex(
                data=[snapshots["max_residual_load"],
                      snapshots["min_residual_load"]]
            ).dropna()
        # if timesteps_pfa is not of type datetime or does not contain
        # datetimes throw an error
        elif not isinstance(timesteps_pfa, datetime.datetime):
            if hasattr(timesteps_pfa, "__iter__"):
                if not all(
                    isinstance(_, datetime.datetime) for _ in timesteps_pfa
                ):
                    raise ValueError(
                        "Input {} for timesteps_pfa is not valid.".format(
                            timesteps_pfa
                        )
                    )
            else:
                raise ValueError(
                    "Input {} for timesteps_pfa is not valid.".format(
                        timesteps_pfa
                    )
                )

    iteration_step = 1
    edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)

    # REINFORCE OVERLOADED TRANSFORMERS AND LINES

    logger.debug("==> Check station load.")
    overloaded_mv_station = checks.hv_mv_station_load(edisgo_reinforce)
    if mode is "mv":
        overloaded_lv_stations = pd.DataFrame()
    else:
        overloaded_lv_stations = checks.mv_lv_station_load(edisgo_reinforce)
    logger.debug("==> Check line load.")
    crit_lines = checks.mv_line_load(edisgo_reinforce)
    if not mode:
        crit_lines = crit_lines.append(checks.lv_line_load(edisgo_reinforce))

    while_counter = 0
    while (
        not overloaded_mv_station.empty
        or not overloaded_lv_stations.empty
        or not crit_lines.empty
    ) and while_counter < max_while_iterations:

        if not overloaded_mv_station.empty:
            # reinforce substations
            transformer_changes = reinforce_measures.reinforce_hv_mv_station_overloading(
                edisgo_reinforce, overloaded_mv_station
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not overloaded_lv_stations.empty:
            # reinforce distribution substations
            transformer_changes = reinforce_measures.reinforce_mv_lv_station_overloading(
                edisgo_reinforce, overloaded_lv_stations
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_lines_overloading(
                edisgo_reinforce, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-loading problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        logger.debug("==> Recheck station load.")
        overloaded_mv_station = checks.hv_mv_station_load(edisgo_reinforce)
        if mode is not "mv":
            overloaded_lv_stations = checks.mv_lv_station_load(
                edisgo_reinforce
            )
        logger.debug("==> Recheck line load.")
        crit_lines = checks.mv_line_load(edisgo_reinforce)
        if not mode:
            crit_lines = crit_lines.append(
                checks.lv_line_load(edisgo_reinforce)
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
        edisgo_reinforce.results.unresolved_issues = \
            pd.concat(
                [edisgo_reinforce.results.unresolved_issues,
                 crit_lines,
                 overloaded_lv_stations,
                 overloaded_mv_station
                 ]
            )
        raise exceptions.MaximumIterationError(
            "Overloading issues could not be solved after maximum allowed "
            "iterations."
        )
    else:
        logger.info(
            "==> Load issues were solved in {} iteration "
            "step(s).".format(while_counter)
        )

    # REINFORCE BRANCHES DUE TO VOLTAGE ISSUES
    iteration_step += 1

    # solve voltage problems in MV topology
    logger.debug("==> Check voltage in MV topology.")
    if combined_analysis:
        voltage_levels = "mv_lv"
    else:
        voltage_levels = "mv"

    crit_nodes = checks.mv_voltage_deviation(
        edisgo_reinforce, voltage_levels=voltage_levels
    )

    while_counter = 0
    while crit_nodes and while_counter < max_while_iterations:

        # reinforce lines
        lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
            edisgo_reinforce,
            edisgo_reinforce.topology.mv_grid,
            crit_nodes[repr(edisgo_reinforce.topology.mv_grid)],
        )
        # write changed lines to results.equipment_changes
        _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-voltage problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        logger.debug("==> Recheck voltage in MV topology.")
        crit_nodes = checks.mv_voltage_deviation(
            edisgo_reinforce, voltage_levels=voltage_levels
        )
        iteration_step += 1
        while_counter += 1

    # check if all voltage problems were solved after maximum number of
    # iterations allowed
    if while_counter == max_while_iterations and crit_nodes:
        edisgo_reinforce.results.unresolved_issues = \
            pd.concat(
                [edisgo_reinforce.results.unresolved_issues,
                 pd.concat([_ for _ in crit_nodes.values()])
                 ]
            )
        raise exceptions.MaximumIterationError(
            "Over-voltage issues for the following nodes in MV topology could "
            "not be solved: {}".format(crit_nodes)
        )
    else:
        logger.info(
            "==> Voltage issues in MV topology were solved in {} "
            "iteration step(s).".format(while_counter)
        )

    # solve voltage problems at secondary side of LV stations
    if mode is not "mv":
        logger.debug("==> Check voltage at secondary side of LV stations.")
        if combined_analysis:
            voltage_levels = "mv_lv"
        else:
            voltage_levels = "lv"
        crit_stations = checks.lv_voltage_deviation(
            edisgo_reinforce, mode="stations", voltage_levels=voltage_levels
        )

        while_counter = 0
        while crit_stations and while_counter < max_while_iterations:
            # reinforce distribution substations
            transformer_changes = reinforce_measures.reinforce_mv_lv_station_voltage_issues(
                edisgo_reinforce, crit_stations
            )
            # write added transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")

            # run power flow analysis again (after updating pypsa object) and
            # check if all over-voltage problems were solved
            logger.debug("==> Run power flow analysis.")
            edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
            logger.debug(
                "==> Recheck voltage at secondary side of LV stations."
            )
            crit_stations = checks.lv_voltage_deviation(
                edisgo_reinforce,
                mode="stations",
                voltage_levels=voltage_levels,
            )

            iteration_step += 1
            while_counter += 1

        # check if all voltage problems were solved after maximum number of
        # iterations allowed
        if while_counter == max_while_iterations and crit_stations:
            edisgo_reinforce.results.unresolved_issues = \
                pd.concat(
                    [edisgo_reinforce.results.unresolved_issues,
                     pd.concat([_ for _ in crit_stations.values()])
                     ]
                )
            raise exceptions.MaximumIterationError(
                "Over-voltage issues at busbar could not be solved for the "
                "following LV grids: {}".format(crit_stations)
            )
        else:
            logger.info(
                "==> Voltage issues at busbars in LV grids were "
                "solved in {} iteration step(s).".format(while_counter)
            )

    # solve voltage problems in LV grids
    if not mode:
        logger.debug("==> Check voltage in LV grids.")
        crit_nodes = checks.lv_voltage_deviation(
            edisgo_reinforce, voltage_levels=voltage_levels
        )

        while_counter = 0
        while crit_nodes and while_counter < max_while_iterations:
            # for every topology in crit_nodes do reinforcement
            for grid in crit_nodes:
                # reinforce lines
                lines_changes = reinforce_measures.reinforce_lines_voltage_issues(
                    edisgo_reinforce,
                    edisgo_reinforce.topology._grids[grid],
                    crit_nodes[grid]
                )
                # write changed lines to results.equipment_changes
                _add_lines_changes_to_equipment_changes()

            # run power flow analysis again (after updating pypsa object)
            # and check if all over-voltage problems were solved
            logger.debug("==> Run power flow analysis.")
            edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
            logger.debug("==> Recheck voltage in LV grids.")
            crit_nodes = checks.lv_voltage_deviation(
                edisgo_reinforce, voltage_levels=voltage_levels
            )

            iteration_step += 1
            while_counter += 1

        # check if all voltage problems were solved after maximum number of
        # iterations allowed
        if while_counter == max_while_iterations and crit_nodes:
            edisgo_reinforce.results.unresolved_issues = \
                pd.concat(
                    [edisgo_reinforce.results.unresolved_issues,
                     pd.concat([_ for _ in crit_nodes.values()])
                     ]
                )
            raise exceptions.MaximumIterationError(
                "Over-voltage issues for the following nodes in LV grids "
                "could not be solved: {}".format(crit_nodes)
            )
        else:
            logger.info(
                "==> Voltage issues in LV grids were solved "
                "in {} iteration step(s).".format(while_counter)
            )

    # RECHECK FOR OVERLOADED TRANSFORMERS AND LINES
    logger.debug("==> Recheck station load.")
    overloaded_mv_station = checks.hv_mv_station_load(edisgo_reinforce)
    if mode is "mv":
        overloaded_lv_stations = pd.DataFrame()
    else:
        overloaded_lv_stations = checks.mv_lv_station_load(edisgo_reinforce)
    logger.debug("==> Recheck line load.")
    crit_lines = checks.mv_line_load(edisgo_reinforce)
    if not mode:
        crit_lines = crit_lines.append(checks.lv_line_load(edisgo_reinforce))

    while_counter = 0
    while (
        not overloaded_mv_station.empty
        or not overloaded_lv_stations.empty
        or not crit_lines.empty
    ) and while_counter < max_while_iterations:

        if not overloaded_mv_station.empty:
            # reinforce substations
            transformer_changes = reinforce_measures.reinforce_hv_mv_station_overloading(
                edisgo_reinforce, overloaded_mv_station
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not overloaded_lv_stations.empty:
            # reinforce substations
            transformer_changes = reinforce_measures.reinforce_mv_lv_station_overloading(
                edisgo_reinforce, overloaded_lv_stations
            )
            # write added and removed transformers to results.equipment_changes
            _add_transformer_changes_to_equipment_changes("added")
            _add_transformer_changes_to_equipment_changes("removed")

        if not crit_lines.empty:
            # reinforce lines
            lines_changes = reinforce_measures.reinforce_lines_overloading(
                edisgo_reinforce, crit_lines
            )
            # write changed lines to results.equipment_changes
            _add_lines_changes_to_equipment_changes()

        # run power flow analysis again (after updating pypsa object) and check
        # if all over-loading problems were solved
        logger.debug("==> Run power flow analysis.")
        edisgo_reinforce.analyze(mode=mode, timesteps=timesteps_pfa)
        logger.debug("==> Recheck station load.")
        overloaded_mv_station = checks.hv_mv_station_load(edisgo_reinforce)
        if mode is not "mv":
            overloaded_lv_stations = checks.mv_lv_station_load(
                edisgo_reinforce
            )
        logger.debug("==> Recheck line load.")
        crit_lines = checks.mv_line_load(edisgo_reinforce)
        if not mode:
            crit_lines = crit_lines.append(
                checks.lv_line_load(edisgo_reinforce)
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
        edisgo_reinforce.results.unresolved_issues = \
            pd.concat(
                [edisgo_reinforce.results.unresolved_issues,
                 crit_lines,
                 overloaded_lv_stations,
                 overloaded_mv_station
                 ]
            )
        raise exceptions.MaximumIterationError(
            "Overloading issues (after solving over-voltage issues) for the"
            "following lines could not be solved: {}".format(crit_lines)
        )
    else:
        logger.info(
            "==> Load issues were rechecked and solved "
            "in {} iteration step(s).".format(while_counter)
        )

    # final check 10% criteria
    checks.check_ten_percent_voltage_deviation(edisgo_reinforce)

    # calculate topology expansion costs
    edisgo_reinforce.results.grid_expansion_costs = grid_expansion_costs(
        edisgo_reinforce
    )

    return edisgo_reinforce.results
