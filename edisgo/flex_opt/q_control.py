import numpy as np


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
