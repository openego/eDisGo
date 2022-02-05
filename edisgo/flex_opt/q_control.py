import numpy as np
import pandas as pd

from edisgo.tools.tools import (
    assign_voltage_level_to_component,
    drop_duplicated_columns
)


def _get_q_sign_generator(reactive_power_mode):
    """
    Get the sign of reactive power in generator sign convention.

    In the generator sign convention the reactive power is negative in
    inductive operation (`reactive_power_mode` is 'inductive') and positive
    in capacitive operation (`reactive_power_mode` is 'capacitive').

    Parameters
    ----------
    reactive_power_mode : str
        Possible options are 'inductive' and 'capacitive'.

    Returns
    --------
    int
        Sign of reactive power in generator sign convention.

    """
    if reactive_power_mode.lower() == "inductive":
        return -1
    elif reactive_power_mode.lower() == "capacitive":
        return 1
    else:
        raise ValueError(
            "reactive_power_mode must either be 'capacitive' "
            "or 'inductive' but is {}.".format(reactive_power_mode)
        )


def _get_q_sign_load(reactive_power_mode):
    """
    Get the sign of reactive power in load sign convention.

    In the load sign convention the reactive power is positive in
    inductive operation (`reactive_power_mode` is 'inductive') and negative
    in capacitive operation (`reactive_power_mode` is 'capacitive').

    Parameters
    ----------
    reactive_power_mode : str
        Possible options are 'inductive' and 'capacitive'.

    Returns
    --------
    int
        Sign of reactive power in load sign convention.

    """
    if reactive_power_mode.lower() == "inductive":
        return 1
    elif reactive_power_mode.lower() == "capacitive":
        return -1
    else:
        raise ValueError(
            "reactive_power_mode must either be 'capacitive' "
            "or 'inductive' but is {}.".format(reactive_power_mode)
        )


def fixed_cosphi(active_power, q_sign, power_factor):
    """
    Calculates reactive power for a fixed cosphi operation.

    Parameters
    ----------
    active_power : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with active power time series. Columns of the dataframe are
        names of the components and index of the dataframe are the time steps
        reactive power is calculated for.
    q_sign : :pandas:`pandas.Series<Series>` or int
        `q_sign` defines whether the reactive power is positive or
        negative and must either be -1 or +1. In case `q_sign` is given as a
        series, the index must contain the same component names as given in
        columns of parameter `active_power`.
    power_factor : :pandas:`pandas.Series<Series>` or float
        Ratio of real to apparent power.
        In case `power_factor` is given as a series, the index must contain the
        same component names as given in columns of parameter `active_power`.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with the same format as the `active_power` dataframe,
        containing the reactive power.

    """
    return active_power * q_sign * np.tan(np.arccos(power_factor))


def _set_reactive_power_time_series_for_fixed_cosphi_using_config(
        edisgo_obj, df, component_type):
    """
    Calculates reactive power in Mvar for a fixed cosphi operation.

    This function adds the calculated reactive power time series to the
    :class:`~.network.timeseries.TimeSeries` object. For
    `component_type`='generators' time series is added to
    :attr:`~.network.timeseries.TimeSeries.generators_reactive_power`, for
    `component_type`='storage_units' time series is added to
    :attr:`~.network.timeseries.TimeSeries.storage_units_reactive_power` and
    for `component_type`='loads' time series is added to
    :attr:`~.network.timeseries.TimeSeries.loads_reactive_power`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names (in the index) of all components
        reactive power needs to be calculated for. Only required column is
        column 'bus', giving the name of the bus the component is connected to.
        All components must have the same `component_type`.
    component_type : str
        Specifies whether to calculate reactive power for generators, storage
        units or loads. The component type determines the power factor and
        power mode used. Possible options are 'generators', 'storage_units' and
        'loads'.

    Notes
    -----
    Reactive power is determined based on reactive power factors and reactive
    power modes defined in the config file 'config_timeseries' in sections
    'reactive_power_factor' and 'reactive_power_mode'. Both are distinguished
    between the voltage level the components are in (medium or low voltage).

    """
    if df.empty:
        return

    # assign voltage level to generators
    df = assign_voltage_level_to_component(edisgo_obj, df)

    # get default configurations
    reactive_power_mode = edisgo_obj.config["reactive_power_mode"]
    reactive_power_factor = edisgo_obj.config["reactive_power_factor"]
    voltage_levels = df.voltage_level.unique()

    # write series with sign of reactive power and power factor
    # for each component
    q_sign = pd.Series(index=df.index)
    power_factor = pd.Series(index=df.index)
    if component_type in ["generators", "storage_units"]:
        get_q_sign = _get_q_sign_generator
    elif component_type == "loads":
        get_q_sign = _get_q_sign_load
    else:
        raise ValueError(
            "Given 'component_type' is not valid. Valid options are "
            "'generators','storage_units' and 'loads'.")
    for voltage_level in voltage_levels:
        cols = df.index[df.voltage_level == voltage_level]
        if len(cols) > 0:
            q_sign[cols] = get_q_sign(
                reactive_power_mode[
                    "{}_gen".format(voltage_level)
                ]
            )
            power_factor[cols] = reactive_power_factor[
                "{}_gen".format(voltage_level)
            ]

    # calculate reactive power time series and append to TimeSeries object
    reactive_power_df = drop_duplicated_columns(
        pd.concat(
            [getattr(edisgo_obj.timeseries,
                     component_type + "_reactive_power"),
             fixed_cosphi(
                 getattr(edisgo_obj.timeseries,
                         component_type + "_active_power").loc[:, df.index],
                 q_sign,
                 power_factor
             )],
            axis=1
        ),
        keep="last"
    )

    setattr(
        edisgo_obj.timeseries,
        component_type + "_reactive_power",
        reactive_power_df
    )
