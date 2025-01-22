import logging
import copy
import warnings

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

    # if heat_pump is not in topology -> exception
    # heat_pumps_in_topology_check = heat_pump_names.isin(edisgo_obj.topology.loads_df.index)
    # for x in range(len(heat_pump_names)):
    #     if not heat_pumps_in_topology_check[x]:
    #         warnings.warn(f"Warning: Heat pump {heat_pump_names[x]} has not been inserted into the grid topology.")
    #         heat_pump_names = heat_pump_names.drop[heat_pump_names[x]]

    if strategy == "uncontrolled":
        ts = (
            edisgo_obj.heat_pump.heat_demand_df.loc[:, heat_pump_names]
            / edisgo_obj.heat_pump.cop_df.loc[:, heat_pump_names]
        )

        # clips heat pump load at maximum level
        ts_prev = copy.deepcopy(ts)
        for heat_pump_name in heat_pump_names:
          ts[heat_pump_name] = [min(x,edisgo_obj.topology.loads_df.p_set[heat_pump_name]) for x in ts[heat_pump_name]]
          if not ts[heat_pump_name].equals(ts_prev[heat_pump_name]):
                warnings.warn(
                    # Extension possible: print heat pumps that were clipped
                    f"Warning: Heat pump active power at {edisgo_obj.topology.loads_df.bus[heat_pump_name]} was limited to its maximum."
                    f"Heat demand at bus {edisgo_obj.topology.loads_df.bus[heat_pump_name]} should be covered by additional heat sources."
                )

        if not ts.equals(ts_prev):
            warnings.warn(
                # Extension possible: print heat pumps that were clipped
                          "Warning: Heat pump active power was clipped at maximum level." 
                          "Heat demand at bus should be covered by additional heat sources."
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
