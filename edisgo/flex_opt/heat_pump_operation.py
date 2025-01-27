import copy
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

    """
    if heat_pump_names is None:
        heat_pump_names = edisgo_obj.heat_pump.cop_df.columns

    missing = set(heat_pump_names) - set(edisgo_obj.topology.loads_df.index)
    if missing:
        logger.warning(
            f"The following heat pumps are are in the heat pump class but not yet "
            f"integrated into the topology class. Therefore, their maximum capacity "
            f"cannot be considered in the operating strategies. {missing=}"
        )

    if strategy == "uncontrolled":
        ts = (
            edisgo_obj.heat_pump.heat_demand_df.loc[:, heat_pump_names]
            / edisgo_obj.heat_pump.cop_df.loc[:, heat_pump_names]
        )

        ts_prev = copy.deepcopy(ts)

        # clips heat pump load at maximum level
        in_topology = list(set(heat_pump_names) - set(missing))

        ts_clipped = ts[in_topology].clip(
            upper=edisgo_obj.topology.loads_df.p_set[in_topology].values
        )
        ts[in_topology] = ts_clipped[in_topology]

        clipped = ts.eq(ts_prev).all()[lambda x: ~x].index
        logger.warning(
            f"Heat pump power at {clipped} was "
            f"clipped at its maximum capacity. The heat demand at this bus "
            f"should be covered by additional heat sources."
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
