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

import logging

import pandas as pd

logger = logging.getLogger("edisgo")


def operating_strategy(
    edisgo_obj,
    strategy="uncontrolled",
    heat_pump_names=None,
):
    """
    Applies operating strategy to set electrical load time series of heat pumps.

    See :attr:`~.edisgo.EDisGo.apply_heat_pump_operating_strategy` for more information.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    strategy : str
        Defines the operating strategy to apply. See `strategy` parameter in
        :attr:`~.edisgo.EDisGo.apply_heat_pump_operating_strategy` for more information.
        Default: 'uncontrolled'.
    heat_pump_names : list(str) or None
        Defines for which heat pumps to apply operating strategy. See `heat_pump_names`
        parameter in :attr:`~.edisgo.EDisGo.apply_heat_pump_operating_strategy` for
        more information. Default: None.

    Notes
    -----
    The written ``loads_active_power`` is scoped to
    ``edisgo_obj.timeseries.timeindex``, regardless of whether
    ``edisgo_obj.heat_pump.heat_demand_df``/``cop_df`` currently span a wider
    or different range (e.g. because the timeindex changed after
    :attr:`~.edisgo.EDisGo.import_heat_pumps` ran). Raises ``KeyError`` if
    either is missing data for a time step in ``timeindex`` - this is
    data-staleness the caller should fix (re-import or re-set the heat pump
    time series for the active timeindex), not something to silently paper
    over.

    """
    if heat_pump_names is None:
        heat_pump_names = edisgo_obj.heat_pump.cop_df.columns

    if strategy == "uncontrolled":
        # Scope to the active timeindex explicitly rather than relying on
        # heat_demand_df/cop_df already matching it - import_heat_pumps trims
        # both to the timeindex active at import time, but nothing re-trims
        # them if the timeindex changes afterward (e.g. a later
        # select_timesteps step), which would otherwise silently write rows
        # outside the current timeindex into loads_active_power.
        timeindex = edisgo_obj.timeseries.timeindex
        ts = (
            edisgo_obj.heat_pump.heat_demand_df.loc[timeindex, heat_pump_names]
            / edisgo_obj.heat_pump.cop_df.loc[timeindex, heat_pump_names]
        )
        edisgo_obj.timeseries.add_component_time_series(
            "loads_active_power",
            ts,
        )
    else:
        raise ValueError(
            f"Heat pump operating strategy {strategy} is not a valid option. "
            f"The only operating strategy currently implemented is 'uncontrolled'."
        )

    # set reactive power time series to 0 Mvar
    edisgo_obj.timeseries.add_component_time_series(
        "loads_reactive_power",
        pd.DataFrame(
            data=0.0,
            index=edisgo_obj.timeseries.timeindex,
            columns=heat_pump_names,
        ),
    )

    logger.debug(f"Heat pump operating strategy {strategy} completed.")
