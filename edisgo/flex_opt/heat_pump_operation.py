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

    if strategy == "uncontrolled":
        ts = (
            edisgo_obj.heat_pump.heat_demand_df.loc[:, heat_pump_names]
            / edisgo_obj.heat_pump.cop_df.loc[:, heat_pump_names]
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
