from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import saio

from sqlalchemy.engine.base import Engine

from edisgo.io.db import session_scope_egon_data
from edisgo.io.timeseries_import import _timeindex_helper_func
from edisgo.tools import tools

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def oedb(
    edisgo_obj: EDisGo,
    scenario: str,
    engine: Engine,
    timeindex=None,
):
    """
    Gets industrial and CTS DSM profiles from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Profiles comprise minimum and maximum load increase in MW as well as maximum energy
    pre- and postponing in MWh.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve DSM data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Specifies time steps for which to return data. Leap years can currently
        not be handled. In case the given timeindex contains a leap year, the data will
        be indexed using the default year (2035 in case of the 'eGon2035' and to 2045
        in case of the 'eGon100RE' scenario) and returned for the whole year.
        If no timeindex is provided, the timeindex set in
        :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
        If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
        is indexed using the default year and returned for the whole year.

    Returns
    --------
    dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
        Dictionary with DSM data with keys `p_min`, `p_max`, `e_min` and `e_max` (see
        :class:`~.network.dsm.DSM` for more information). Values contain dataframes with
        DSM profiles per load for one year in an hourly resolution in MW. Index of the
        dataframes are time indices. Columns contain the load name the DSM profile is
        associated with as in index of :attr:`~.network.topology.Topology.loads_df`.

    """
    # get CTS and industrial DSM profiles
    dsm_cts = get_profile_cts(edisgo_obj, scenario, engine)
    ind_loads = edisgo_obj.topology.loads_df[
        (edisgo_obj.topology.loads_df.type == "conventional_load")
        & (edisgo_obj.topology.loads_df.sector == "industry")
    ]
    dsm_ind = get_profiles_per_industrial_load(
        ind_loads.building_id.unique(), scenario, engine
    )

    # rename industrial DSM profiles, join with CTS profiles and set time index
    rename_series = (
        ind_loads.loc[:, ["building_id"]]
        .dropna()
        .reset_index()
        .set_index("building_id")
        .iloc[:, 0]
    )
    timeindex, timeindex_full = _timeindex_helper_func(
        edisgo_obj,
        timeindex,
        default_year=tools.get_year_based_on_scenario(scenario),
        allow_leap_year=False,
    )
    dsm_ind_cts = {}
    for dsm_profile in ["e_min", "e_max", "p_min", "p_max"]:
        dsm_ind[dsm_profile].rename(columns=rename_series, inplace=True)
        dsm_ind_cts_tmp = pd.concat(
            [dsm_cts[dsm_profile], dsm_ind[dsm_profile]], axis=1
        )
        dsm_ind_cts_tmp.index = timeindex_full
        dsm_ind_cts[dsm_profile] = dsm_ind_cts_tmp.loc[timeindex, :]

    return dsm_ind_cts


def get_profiles_per_industrial_load(
    load_ids,
    scenario: str,
    engine: Engine,
):
    """
    Gets industrial DSM profiles per site and OSM area.

    Parameters
    ----------
    load_ids : list(int)
        List of industrial site and OSM IDs to retrieve DSM profiles for.
    scenario : str
        Scenario for which to retrieve DSM data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
        Dictionary with DSM data with keys `p_min`, `p_max`, `e_min` and `e_max`. Values
        contain dataframes with DSM profiles per site and OSM area for one year in an
        hourly resolution in MW. Index contains hour of the year (from 0 to 8759) and
        column names are site ID as integer.

    """
    saio.register_schema("demand", engine)
    from saio.demand import (
        egon_demandregio_sites_ind_electricity_dsm_timeseries as sites_ind_dsm_ts,
    )
    from saio.demand import (
        egon_osm_ind_load_curves_individual_dsm_timeseries,
        egon_sites_ind_load_curves_individual_dsm_timeseries,
    )

    dsm_dict = {}

    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_sites_ind_load_curves_individual_dsm_timeseries.site_id,
            egon_sites_ind_load_curves_individual_dsm_timeseries.p_min,
            egon_sites_ind_load_curves_individual_dsm_timeseries.p_max,
            egon_sites_ind_load_curves_individual_dsm_timeseries.e_min,
            egon_sites_ind_load_curves_individual_dsm_timeseries.e_max,
        ).filter(
            egon_sites_ind_load_curves_individual_dsm_timeseries.scn_name == scenario,
            egon_sites_ind_load_curves_individual_dsm_timeseries.site_id.in_(load_ids),
        )

        df_sites_1 = pd.read_sql(sql=query.statement, con=engine)

    with session_scope_egon_data(engine) as session:
        query = session.query(
            sites_ind_dsm_ts.industrial_sites_id.label("site_id"),
            sites_ind_dsm_ts.p_min,
            sites_ind_dsm_ts.p_max,
            sites_ind_dsm_ts.e_min,
            sites_ind_dsm_ts.e_max,
        ).filter(
            sites_ind_dsm_ts.scn_name == scenario,
            sites_ind_dsm_ts.industrial_sites_id.in_(load_ids),
        )

        df_sites_2 = pd.read_sql(sql=query.statement, con=engine)

    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_osm_ind_load_curves_individual_dsm_timeseries.osm_id.label("site_id"),
            egon_osm_ind_load_curves_individual_dsm_timeseries.p_min,
            egon_osm_ind_load_curves_individual_dsm_timeseries.p_max,
            egon_osm_ind_load_curves_individual_dsm_timeseries.e_min,
            egon_osm_ind_load_curves_individual_dsm_timeseries.e_max,
        ).filter(
            egon_osm_ind_load_curves_individual_dsm_timeseries.scn_name == scenario,
            egon_osm_ind_load_curves_individual_dsm_timeseries.osm_id.in_(load_ids),
        )

        df_areas = pd.read_sql(sql=query.statement, con=engine)

    df = pd.concat([df_sites_1, df_sites_2, df_areas])
    # add time step column
    df["time_step"] = len(df) * [np.arange(0, 8760)]
    # un-nest time series data and pivot so that time_step becomes index and
    # site_id column names
    dsm_dict["p_min"] = _pivot_helper(df, "p_min")
    dsm_dict["p_max"] = _pivot_helper(df, "p_max")
    dsm_dict["e_min"] = _pivot_helper(df, "e_min")
    dsm_dict["e_max"] = _pivot_helper(df, "e_max")

    return dsm_dict


def get_profile_cts(
    edisgo_obj: EDisGo,
    scenario: str,
    engine: Engine,
):
    """
    Gets CTS DSM profiles for all CTS loads in the MV grid.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve DSM data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    dict(str, :pandas:`pandas.DataFrame<DataFrame>`)
        Dictionary with DSM data with keys `p_min`, `p_max`, `e_min` and `e_max`. Values
        contain dataframes with DSM profiles per CTS load for one year in an
        hourly resolution in MW. Index contains hour of the year (from 0 to 8759) and
        column names are site ID as integer.

    Notes
    ------
    Be aware, that in this function the DSM time series are disaggregated to all CTS
    loads in the grid. In some cases, this can lead to an over- or underestimation of
    the DSM potential, as in egon_data buildings are mapped to a grid based on the
    zensus cell they are in whereas in ding0 buildings are mapped to a grid based on
    the geolocation. As it can happen that buildings lie outside an MV grid but within
    a zensus cell that is assigned to that MV grid, they are mapped differently in
    egon_data and ding0.

    """
    saio.register_schema("demand", engine)
    from saio.demand import egon_etrago_electricity_cts_dsm_timeseries

    # get data
    dsm_dict = {}

    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_etrago_electricity_cts_dsm_timeseries.bus.label("site_id"),
            egon_etrago_electricity_cts_dsm_timeseries.p_min,
            egon_etrago_electricity_cts_dsm_timeseries.p_max,
            egon_etrago_electricity_cts_dsm_timeseries.e_min,
            egon_etrago_electricity_cts_dsm_timeseries.e_max,
        ).filter(
            egon_etrago_electricity_cts_dsm_timeseries.scn_name == scenario,
            egon_etrago_electricity_cts_dsm_timeseries.bus == edisgo_obj.topology.id,
        )
        df = pd.read_sql(sql=query.statement, con=engine)
    # add time step column
    df["time_step"] = len(df) * [np.arange(0, 8760)]
    # un-nest time series data and pivot so that time_step becomes index and
    # site_id column names
    dsm_dict["p_min"] = _pivot_helper(df, "p_min")
    dsm_dict["p_max"] = _pivot_helper(df, "p_max")
    dsm_dict["e_min"] = _pivot_helper(df, "e_min")
    dsm_dict["e_max"] = _pivot_helper(df, "e_max")

    # distribute over all CTS loads
    cts_loads = edisgo_obj.topology.loads_df[
        (edisgo_obj.topology.loads_df.type == "conventional_load")
        & (edisgo_obj.topology.loads_df.sector == "cts")
    ]
    if not dsm_dict["p_min"].empty:
        if len(cts_loads) == 0:
            raise ValueError("There is CTS DSM potential but no CTS loads.")
        for dsm_ts in ["p_min", "p_max", "e_min", "e_max"]:
            dsm_dict[dsm_ts] = pd.DataFrame(
                data=(
                    np.matmul(
                        dsm_dict[dsm_ts].values, np.matrix(cts_loads["p_set"].values)
                    )
                    / cts_loads["p_set"].sum()
                ),
                index=dsm_dict[dsm_ts].index,
                columns=cts_loads["p_set"].index,
            )

    return dsm_dict


def _pivot_helper(df_db, col):
    df = (
        df_db.loc[:, ["site_id", col, "time_step"]]
        .explode([col, "time_step"])
        .astype({col: "float"})
    )
    df = df.pivot(index="time_step", columns="site_id", values=col)
    return df
