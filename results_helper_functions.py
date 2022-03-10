import pandas as pd
import numpy as np
from edisgo.flex_opt.check_tech_constraints import lines_allowed_load, \
    lines_relative_load, _mv_allowed_voltage_limits, _lv_allowed_voltage_limits


def relative_load(edisgo_obj):
    """
    Calculates relative load of all lines and stations power flow results
    are available for.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    pd.DataFrame
        Dataframe with relative load (actual load / allowed load) for all
        components and time steps power flow analysis was conducted for. In
        case of transformers, relative load of station is returned.

    """
    # lines
    # check if power flow was conducted for the MV
    mv_lines = edisgo_obj.topology.mv_grid.lines_df.index
    if any(mv_lines.isin(edisgo_obj.results.i_res.columns)):
        allowed_load_lines = lines_allowed_load(edisgo_obj, "mv")
    else:
        allowed_load_lines = pd.DataFrame()

    # check if power flow was conducted for the LV
    lv_lines = edisgo_obj.topology.lines_df[
        ~edisgo_obj.topology.lines_df.index.isin(mv_lines)
    ].index
    if any(lv_lines.isin(edisgo_obj.results.i_res.columns)):
        lv_lines_allowed_load = lines_allowed_load(edisgo_obj, "lv")
        allowed_load_lines = pd.concat(
            [allowed_load_lines,
             lv_lines_allowed_load.loc[:, edisgo_obj.results.i_res.columns[
                                              edisgo_obj.results.i_res.columns.isin(
                                                  lv_lines_allowed_load.columns)]]],
            axis=1)

    # calculated relative load for lines
    rel_load = lines_relative_load(edisgo_obj, allowed_load_lines)

    # MV-LV stations
    # check if power flow was conducted for stations
    if not edisgo_obj.results.s_res.empty:
        if not edisgo_obj.results.s_res.loc[
           :, edisgo_obj.results.s_res.columns.str.contains("Transformer")].empty:

            load_factor = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
                lambda _: edisgo_obj.config["grid_expansion_load_factors"][
                    "{}_{}_transformer".format("lv", _)
                ]
            )
            for grid in edisgo_obj.topology.mv_grid.lv_grids:
                transformers_df = grid.transformers_df

                # check if grid was included in power flow
                if transformers_df.index[0] in edisgo_obj.results.s_res.columns:
                    # get apparent power over station from power flow analysis
                    s_station_pfa = edisgo_obj.results.s_res.loc[
                        :, transformers_df.index
                    ].sum(axis=1)

                    # get maximum allowed apparent power of station in each time
                    # step
                    s_station_allowed = sum(transformers_df.s_nom) * load_factor
                    rel_load["mvlv_station_{}".format(grid)] = \
                        s_station_pfa / s_station_allowed

    # HV-MV station
    # check if power flow was conducted for MV
    if any(mv_lines.isin(edisgo_obj.results.i_res.columns)):
        transformers_df = edisgo_obj.topology.transformers_hvmv_df
        s_station_pfa = np.hypot(
            edisgo_obj.results.pfa_slack.p,
            edisgo_obj.results.pfa_slack.q,
        )
        # get maximum allowed apparent power of station in each time step
        s_station = sum(transformers_df.s_nom)
        load_factor = edisgo_obj.timeseries.timesteps_load_feedin_case.apply(
            lambda _: edisgo_obj.config["grid_expansion_load_factors"][
                "{}_{}_transformer".format("mv", _)
            ]
        )
        s_station_allowed = s_station * load_factor
        rel_load["hvmv_station_{}".format(edisgo_obj.topology.mv_grid)] = \
            s_station_pfa / s_station_allowed

    return rel_load


def voltage_diff_stations(edisgo_obj):

    # get all primary and secondary sides
    primary_sides = pd.Series()
    secondary_sides = pd.Series()
    for grid in edisgo_obj.topology.mv_grid.lv_grids:
        primary_sides[grid] = grid.transformers_df.iloc[0].bus0
        secondary_sides[grid] = grid.station.index[0]

    voltage_base = edisgo_obj.results.v_res.loc[:, primary_sides.values]

    v_allowed_per_case = {}
    v_allowed_per_case["feedin_case_upper"] = (
            voltage_base
            + edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "mv_lv_station_feedin_case_max_v_deviation"
            ]
    )
    v_allowed_per_case["load_case_lower"] = (
            voltage_base
            - edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
                "mv_lv_station_load_case_max_v_deviation"
            ]
    )

    timeindex = voltage_base.index
    v_allowed_per_case["feedin_case_lower"] = pd.DataFrame(
        edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "feedin_case_lower"
        ],
        columns=v_allowed_per_case["load_case_lower"].columns,
        index=timeindex,
    )
    v_allowed_per_case["load_case_upper"] = pd.DataFrame(
        edisgo_obj.config["grid_expansion_allowed_voltage_deviations"][
            "load_case_upper"
        ],
        columns=v_allowed_per_case["load_case_lower"].columns,
        index=timeindex,
    )

    v_dev_allowed_upper_all = pd.DataFrame()
    v_dev_allowed_lower_all = pd.DataFrame()
    load_feedin_case = edisgo_obj.timeseries.timesteps_load_feedin_case
    for t in timeindex:
        case = load_feedin_case.loc[t]
        v_dev_allowed_upper_all = v_dev_allowed_upper_all.append(
            v_allowed_per_case["{}_upper".format(case)].loc[[t], :]
        )
        v_dev_allowed_lower_all = v_dev_allowed_lower_all.append(
            v_allowed_per_case["{}_lower".format(case)].loc[[t], :]
        )

    # rename columns to secondary side
    rename_dict = {primary_sides[g]: secondary_sides[g] for g in
                   edisgo_obj.topology.mv_grid.lv_grids}
    v_dev_allowed_upper_all.rename(columns=rename_dict, inplace=True)
    v_dev_allowed_lower_all.rename(columns=rename_dict, inplace=True)

    v_mag_pu_pfa_all = edisgo_obj.results.v_res.loc[:,
                       v_dev_allowed_upper_all.columns]

    overvoltage = v_mag_pu_pfa_all[
        v_mag_pu_pfa_all > v_dev_allowed_upper_all
        ]
    undervoltage = v_mag_pu_pfa_all[
        v_mag_pu_pfa_all < v_dev_allowed_lower_all
        ]

    # overvoltage diff (positive)
    overvoltage_diff = overvoltage - v_dev_allowed_upper_all
    # undervoltage diff (negative)
    undervoltage_diff = undervoltage - v_dev_allowed_lower_all

    voltage_difference_all = overvoltage_diff.fillna(
        0) + undervoltage_diff.fillna(0)

    return voltage_difference_all


def voltage_diff(edisgo_obj):
    """
    Calculates deviation from allowed upper or lower voltage limit.

    Voltage deviation equals difference between allowed voltage limit and
    actual voltage at bus. Overvoltage is defined as positive, undervoltage
    as negative and voltage within allowed limits as zero.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with deviations from allowed lower and upper voltage limit.
        Index of the dataframe are all time steps power flow analysis was
        conducted for of type :pandas:`pandas.Timestamp<Timestamp>`; columns
        are all MV lines and, in case MV-LV stations were included in the power
        flow analysis, the MV-LV station's secondary side buses.

    """
    # MV buses
    # check if power flow was conducted for the MV
    mv_buses = edisgo_obj.topology.mv_grid.buses_df.index
    if any(mv_buses.isin(edisgo_obj.results.v_res.columns)):
        v_dev_allowed_upper, v_dev_allowed_lower = _mv_allowed_voltage_limits(
            edisgo_obj, voltage_levels="mv")

        v_mag_pu_pfa = edisgo_obj.results.v_res.loc[
                       :, edisgo_obj.topology.mv_grid.buses_df.index]

        v_dev_allowed_upper_format = np.tile(
            (v_dev_allowed_upper.loc[v_mag_pu_pfa.index]).values,
            (v_mag_pu_pfa.shape[1], 1),
        )
        v_dev_allowed_lower_format = np.tile(
            (v_dev_allowed_lower.loc[v_mag_pu_pfa.index]).values,
            (v_mag_pu_pfa.shape[1], 1),
        )
        overvoltage = v_mag_pu_pfa.T[
            v_mag_pu_pfa.T > v_dev_allowed_upper_format
        ]
        undervoltage = v_mag_pu_pfa.T[
            v_mag_pu_pfa.T < v_dev_allowed_lower_format
        ]

        # overvoltage diff (positive)
        overvoltage_diff = overvoltage - v_dev_allowed_upper_format
        # undervoltage diff (negative)
        undervoltage_diff = undervoltage - v_dev_allowed_lower_format
        voltage_difference = (overvoltage_diff.fillna(0) +
                              undervoltage_diff.fillna(0))
        voltage_difference = voltage_difference.T
    else:
        voltage_difference = pd.DataFrame()

    # MV-LV stations
    # check if power flow was conducted for stations
    if not edisgo_obj.results.s_res.empty:
        if not edisgo_obj.results.s_res.loc[
               :,
               edisgo_obj.results.s_res.columns.str.contains("Transformer")].empty:

            voltage_difference_stations = voltage_diff_stations(edisgo_obj)
            voltage_difference = pd.concat(
                [voltage_difference, voltage_difference_stations],
                sort=False, axis=1
            )

    # LV buses
    # check if power flow was as well conducted for LV
    lv_buses = edisgo_obj.topology.buses_df[
        ~edisgo_obj.topology.buses_df.index.isin(mv_buses)
    ].index
    if any(lv_buses.isin(edisgo_obj.results.v_res.columns)):

        for lv_grid in edisgo_obj.topology.mv_grid.lv_grids:

            # check if grid was included in power flow
            if any(lv_grid.lines_df.index.isin(
                    edisgo_obj.results.s_res.columns)):

                v_dev_allowed_upper, v_dev_allowed_lower = \
                    _lv_allowed_voltage_limits(
                        edisgo_obj, lv_grid, mode=None)

                v_mag_pu_pfa = edisgo_obj.results.v_res.loc[
                               :, lv_grid.buses_df.index]

                v_dev_allowed_upper_format = np.tile(
                    (v_dev_allowed_upper.loc[v_mag_pu_pfa.index]).values,
                    (v_mag_pu_pfa.shape[1], 1),
                )
                v_dev_allowed_lower_format = np.tile(
                    (v_dev_allowed_lower.loc[v_mag_pu_pfa.index]).values,
                    (v_mag_pu_pfa.shape[1], 1),
                )
                overvoltage = v_mag_pu_pfa.T[
                    v_mag_pu_pfa.T > v_dev_allowed_upper_format
                    ]
                undervoltage = v_mag_pu_pfa.T[
                    v_mag_pu_pfa.T < v_dev_allowed_lower_format
                    ]

                # overvoltage diff (positive)
                overvoltage_diff = overvoltage - v_dev_allowed_upper_format
                # undervoltage diff (negative)
                undervoltage_diff = undervoltage - v_dev_allowed_lower_format
                voltage_difference_lv_grid = (
                        overvoltage_diff.fillna(0) +
                        undervoltage_diff.fillna(0)
                )
                voltage_difference = pd.concat(
                    [voltage_difference,
                     voltage_difference_lv_grid[~voltage_difference_lv_grid.T.columns.isin(voltage_difference.columns)].T],
                    sort=False, axis=1
                )

    return voltage_difference