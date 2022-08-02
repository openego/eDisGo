import numpy as np
import pandas as pd


def get_q_sign_generator(reactive_power_mode):
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


def get_q_sign_load(reactive_power_mode):
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


def _get_component_dict():
    """
    Helper function to translate from component type term used in function to the one
    used in the config files.

    """
    comp_dict = {
        "generators": "gen",
        "storage_units": "storage",
        "loads": "load",
        "charging_points": "cp",
        "heat_pumps": "hp",
    }
    return comp_dict


def _fixed_cosphi_default_power_factor(comp_df, component_type, configs):
    """
    Gets fixed cosphi default reactive power factor for each given component.

    Parameters
    -----------
    comp_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names (in the index) of all components
        reactive power factor needs to be set. Only required column is
        column 'voltage_level', giving the voltage level the component is in (the
        voltage level can be set using the function
        :func:`~.tools.tools.assign_voltage_level_to_component`).
        All components must have the same `component_type`.
    component_type : str
        The component type determines the reactive power factor and mode used.
        Possible options are 'generators', 'storage_units', 'loads', 'charging_points',
        and 'heat_pumps'.
    configs : :class:`~.tools.config.Config`
        eDisGo configuration data.

    Returns
    --------
    :pandas:`pandas.Series<Series>`
        Series with default reactive power factor in case of fixed coshpi for each
        component in index of `comp_df`.

    """
    reactive_power_factor = configs["reactive_power_factor"]
    comp_dict = _get_component_dict()

    if component_type in comp_dict.keys():
        comp = comp_dict[component_type]
        # write series with power factor for each component
        power_factor = pd.Series(index=comp_df.index, dtype=float)
        for voltage_level in comp_df.voltage_level.unique():
            cols = comp_df.index[comp_df.voltage_level == voltage_level]
            if len(cols) > 0:
                power_factor[cols] = reactive_power_factor[f"{voltage_level}_{comp}"]
        return power_factor
    else:
        raise ValueError(
            "Given 'component_type' is not valid. Valid options are "
            "'generators','storage_units', 'loads', 'charging_points', and "
            "'heat_pumps'."
        )


def _fixed_cosphi_default_reactive_power_sign(comp_df, component_type, configs):
    """
    Gets fixed cosphi default value for sign of reactive power for each given component.

    Parameters
    -----------
    comp_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with component names (in the index) of all components sign of
        reactive power needs to be set. Only required column is
        column 'voltage_level', giving the voltage level the component is in (the
        voltage level can be set using the function
        :func:`~.tools.tools.assign_voltage_level_to_component`).
        All components must have the same `component_type`.
    component_type : str
        The component type determines the reactive power factor and mode used.
        Possible options are 'generators', 'storage_units', 'loads', 'charging_points',
        and 'heat_pumps'.
    configs : :class:`~.tools.config.Config`
        eDisGo configuration data.

    Returns
    --------
    :pandas:`pandas.Series<Series>`
        Series with default sign of reactive power in case of fixed cosphi for each
        component in index of `comp_df`.

    """
    reactive_power_mode = configs["reactive_power_mode"]
    comp_dict = _get_component_dict()
    q_sign_dict = {
        "generators": get_q_sign_generator,
        "storage_units": get_q_sign_generator,
        "loads": get_q_sign_load,
        "charging_points": get_q_sign_load,
        "heat_pumps": get_q_sign_load,
    }

    if component_type in comp_dict.keys():
        comp = comp_dict[component_type]
        get_q_sign = q_sign_dict[component_type]
        # write series with power factor for each component
        q_sign = pd.Series(index=comp_df.index, dtype=float)
        for voltage_level in comp_df.voltage_level.unique():
            cols = comp_df.index[comp_df.voltage_level == voltage_level]
            if len(cols) > 0:
                q_sign[cols] = get_q_sign(
                    reactive_power_mode[f"{voltage_level}_{comp}"]
                )
        return q_sign
    else:
        raise ValueError(
            "Given 'component_type' is not valid. Valid options are "
            "'generators','storage_units', 'loads', 'charging_points', and "
            "'heat_pumps'."
        )
