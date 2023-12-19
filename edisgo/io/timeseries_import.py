from __future__ import annotations

import datetime
import logging
import os

import numpy as np
import pandas as pd
import saio

from demandlib import bdew as bdew
from demandlib import particular_profiles as profiles
from sqlalchemy.engine.base import Engine
from workalendar.europe import Germany

from edisgo.io.db import session_scope_egon_data
from edisgo.tools import session_scope, tools

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import model_draft, supply

logger = logging.getLogger(__name__)


def _timeindex_helper_func(
    edisgo_object, timeindex, default_year=2011, allow_leap_year=False
):
    """
    Helper function to set up a timeindex for an entire year to initially set an index
    on the imported data and timeindex to select certain time steps.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Timeindex that was provided by the user.
    default_year : int
        Default year to use in case no timeindex was provided by the user and no
        timeindex is set in :py:attr:`~.network.timeseries.TimeSeries.timeindex`.
    allow_leap_year : bool
        If False and a leap year is given, either in `timeindex` given by the user or
        set in :py:attr:`~.network.timeseries.TimeSeries.timeindex`, the default
        year is used instead.

    Returns
    -------
    (:pandas:`pandas.DatetimeIndex<DatetimeIndex>`,\
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`)
        Returns timeindex to select certain time steps and timeindex for entire year.

    """
    if timeindex is None:
        year = tools.get_year_based_on_timeindex(edisgo_object)
        if year is None:
            year = default_year
            timeindex = pd.date_range(f"1/1/{year}", periods=8760, freq="H")
        else:
            timeindex = edisgo_object.timeseries.timeindex
        timeindex_full = pd.date_range(f"1/1/{year}", periods=8760, freq="H")
    else:
        year = timeindex.year[0]
        if allow_leap_year is False and pd.Timestamp(year, 1, 1).is_leap_year:
            year = default_year
            logger.warning(
                f"A leap year was given. This is currently not valid. The year the "
                f"data is indexed by is therefore set to the default value of "
                f"{default_year}."
            )
            timeindex = pd.date_range(f"1/1/{year}", periods=8760, freq="H")
        timeindex_full = pd.date_range(f"1/1/{year}", periods=8760, freq="H")
    return timeindex, timeindex_full


def feedin_oedb_legacy(edisgo_object, timeindex=None):
    """
    Import feed-in time series data for wind and solar power plants from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Specifies time steps for which to return feed-in data. Leap years can currently
        not be handled. In case the given timeindex contains a leap year, the data will
        be indexed using the default year 2011 and returned for the whole year.
        If no timeindex is provided, the timeindex set in
        :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
        If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
        is indexed using the default year 2011 and returned for the whole year.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly time series for active power feed-in per
        generator type (wind or solar, in column level 0) and weather cell
        (in column level 1), normalized to a capacity of 1 MW.

    """

    def _retrieve_timeseries_from_oedb(session):
        """Retrieve time series from oedb"""
        feedin_sqla = session.query(
            orm_feedin.w_id.label("weather_cell_id"),
            orm_feedin.source.label("carrier"),
            orm_feedin.feedin,
        ).filter(
            orm_feedin.w_id.in_(weather_cell_ids),
            orm_feedin.power_class.in_([0, 4]),
            orm_feedin_version,
            orm_feedin.weather_year == 2011,
        )
        return pd.read_sql_query(
            feedin_sqla.statement,
            session.bind,
        )

    if edisgo_object.config["data_source"]["oedb_data_source"] == "model_draft":
        orm_feedin_name = edisgo_object.config["model_draft"]["res_feedin_data"]
        orm_feedin = model_draft.__getattribute__(orm_feedin_name)
        orm_feedin_version = 1 == 1
    else:
        orm_feedin_name = edisgo_object.config["versioned"]["res_feedin_data"]
        orm_feedin = supply.__getattribute__(orm_feedin_name)
        orm_feedin_version = (
            orm_feedin.version == edisgo_object.config["versioned"]["version"]
        )

    weather_cell_ids = tools.get_weather_cells_intersecting_with_grid_district(
        edisgo_object
    )

    with session_scope() as session:
        feedin_df = _retrieve_timeseries_from_oedb(session)

    # rename wind_onshore to wind
    feedin_df.carrier = feedin_df.carrier.str.replace("_onshore", "")
    # add time step column
    feedin_df["time_step"] = len(feedin_df) * [np.arange(0, 8760)]
    # un-nest feedin and pivot so that time_step becomes index and carrier and
    # weather_cell_id column names
    feedin_df = feedin_df.explode(["feedin", "time_step"]).pivot(
        index="time_step", columns=["carrier", "weather_cell_id"], values="feedin"
    )

    # set time index
    timeindex, timeindex_full = _timeindex_helper_func(
        edisgo_object, timeindex, default_year=2011, allow_leap_year=False
    )
    feedin_df.index = timeindex_full

    return feedin_df.loc[timeindex, :].astype("float")


def feedin_oedb(
    edisgo_object,
    engine: Engine,
    timeindex=None,
):
    """
    Import feed-in time series data for wind and solar power plants from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Specifies time steps for which to return feed-in data. Leap years can currently
        not be handled. In case the given timeindex contains a leap year, the data will
        be indexed using the default year 2011 and returned for the whole year.
        If no timeindex is provided, the timeindex set in
        :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
        If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
        is indexed using the default year 2011 and returned for the whole year.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly feed-in time series per generator type (wind or solar,
        in column level 0) and weather cell (in column level 1), normalized to a
        capacity of 1 MW. Index of the dataframe depends on parameter `timeindex`.

    """
    # get weather cell IDs in grid
    weather_cell_ids = tools.get_weather_cells_intersecting_with_grid_district(
        edisgo_object, engine=engine
    )

    saio.register_schema("supply", engine)
    from saio.supply import egon_era5_renewable_feedin

    with session_scope_egon_data(engine) as session:
        query = (
            session.query(
                egon_era5_renewable_feedin.w_id.label("weather_cell_id"),
                egon_era5_renewable_feedin.carrier,
                egon_era5_renewable_feedin.feedin,
            )
            .filter(
                egon_era5_renewable_feedin.w_id.in_(weather_cell_ids),
                egon_era5_renewable_feedin.carrier.in_(["pv", "wind_onshore"]),
            )
            .order_by(
                egon_era5_renewable_feedin.w_id, egon_era5_renewable_feedin.carrier
            )
        )
        feedin_df = pd.read_sql(sql=query.statement, con=engine)

    # rename pv to solar and wind_onshore to wind
    feedin_df.carrier = feedin_df.carrier.str.replace("pv", "solar").str.replace(
        "_onshore", ""
    )
    # add time step column
    feedin_df["time_step"] = len(feedin_df) * [np.arange(0, 8760)]
    # un-nest feedin and pivot so that time_step becomes index and carrier and
    # weather_cell_id column names
    feedin_df = feedin_df.explode(["feedin", "time_step"]).pivot(
        index="time_step", columns=["carrier", "weather_cell_id"], values="feedin"
    )

    # set time index
    timeindex, timeindex_full = _timeindex_helper_func(
        edisgo_object, timeindex, default_year=2011, allow_leap_year=False
    )
    feedin_df.index = timeindex_full

    return feedin_df.loc[timeindex, :].astype("float")


def load_time_series_demandlib(edisgo_obj, timeindex=None):
    """
    Get normalized sectoral electricity load time series using the
    `demandlib <https://github.com/oemof/demandlib/>`_.

    Resulting electricity load profiles hold time series of hourly conventional
    electricity demand for the sectors residential, cts, agricultural
    and industrial. Time series are normalized to a consumption of 1 MWh per year.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Specifies time steps for which to return data. If no timeindex is provided, the
        timeindex set in :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
        If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
        is indexed using the default year 2011 and returned for the whole year.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with conventional electricity load time series for sectors
        residential, cts, agricultural and industrial.
        Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns
        hold the sector type.

    """
    timeindex, _ = _timeindex_helper_func(
        edisgo_obj, timeindex, default_year=2011, allow_leap_year=True
    )

    year = timeindex[0].year

    sectoral_consumption = {"h0": 1, "g0": 1, "i0": 1, "l0": 1}

    cal = Germany()
    holidays = dict(cal.holidays(year))

    e_slp = bdew.ElecSlp(year, holidays=holidays)

    # multiply given annual demand with timeseries
    elec_demand = e_slp.get_profile(sectoral_consumption)

    # Add the slp for the industrial group
    ilp = profiles.IndustrialLoadProfile(e_slp.date_time_index, holidays=holidays)

    # Beginning and end of workday, weekdays and weekend days, and scaling
    # factors by default
    elec_demand["i0"] = ilp.simple_profile(
        sectoral_consumption["i0"],
        am=datetime.time(
            edisgo_obj.config["demandlib"]["day_start"].hour,
            edisgo_obj.config["demandlib"]["day_start"].minute,
            0,
        ),
        pm=datetime.time(
            edisgo_obj.config["demandlib"]["day_end"].hour,
            edisgo_obj.config["demandlib"]["day_end"].minute,
            0,
        ),
        profile_factors={
            "week": {
                "day": edisgo_obj.config["demandlib"]["week_day"],
                "night": edisgo_obj.config["demandlib"]["week_night"],
            },
            "weekend": {
                "day": edisgo_obj.config["demandlib"]["weekend_day"],
                "night": edisgo_obj.config["demandlib"]["weekend_night"],
            },
        },
    )

    # Resample 15-minute values to hourly values and sum across sectors
    elec_demand = elec_demand.resample("H").mean()

    elec_demand.rename(
        columns={
            "g0": "cts",
            "h0": "residential",
            "l0": "agricultural",
            "i0": "industrial",
        },
        inplace=True,
    )

    return elec_demand.loc[timeindex]


def cop_oedb(edisgo_object, engine, weather_cell_ids, timeindex=None):
    """
    Get COP (coefficient of performance) time series data from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    edisgo_object : :class:`~.EDisGo`
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    weather_cell_ids : list(int) or list(float)
        List (or array) of weather cell IDs to obtain COP data for.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or None
        Specifies time steps for which to return data. Leap years can currently
        not be handled. In case the given timeindex contains a leap year, the data will
        be indexed using the default year 2011 and returned for the whole year.
        If no timeindex is provided, the timeindex set in
        :py:attr:`~.network.timeseries.TimeSeries.timeindex` is used.
        If :py:attr:`~.network.timeseries.TimeSeries.timeindex` is not set, the data
        is indexed using the default year 2011 and returned for the whole year.


    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly COP time series in p.u. per weather cell. Index of the
        dataframe is a time index. Columns contain the weather cell ID as integer.

    """
    # set up time index to index COP data by
    timeindex, timeindex_full = _timeindex_helper_func(
        edisgo_object, timeindex, default_year=2011, allow_leap_year=False
    )

    saio.register_schema("supply", engine)
    from saio.supply import egon_era5_renewable_feedin

    # get cop from database
    with session_scope_egon_data(engine) as session:
        query = (
            session.query(
                egon_era5_renewable_feedin.w_id,
                egon_era5_renewable_feedin.feedin.label("cop"),
            )
            .filter(egon_era5_renewable_feedin.carrier == "heat_pump_cop")
            .filter(egon_era5_renewable_feedin.w_id.in_(weather_cell_ids))
        )

        cop = pd.read_sql(query.statement, engine, index_col="w_id")

    # convert dataframe to have weather cell ID as column name and time index
    cop = pd.DataFrame(
        {w_id: ts.cop for w_id, ts in cop.iterrows()}, index=timeindex_full
    )

    return cop.loc[timeindex, :]


def heat_demand_oedb(edisgo_obj, scenario, engine, timeindex=None):
    """
    Get heat demand profiles for heat pumps from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Heat demand data is returned for all heat pumps in the grid.
    For more information on how individual heat demand profiles are obtained see
    functions :func:`~.io.timeseries_import.get_residential_heat_profiles_per_building`
    and :func:`~.io.timeseries_import.get_cts_profiles_per_building`.
    For more information on how district heating heat demand profiles are obtained see
    function :func:`~.io.timeseries_import.get_district_heating_heat_demand_profiles`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve demand data. Possible options
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
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly heat demand for one year in MW per heat pump. Index of the
        dataframe is a time index. Columns contain the heat pump name as in index of
        :attr:`~.network.topology.Topology.loads_df`.

    """
    if scenario not in ["eGon2035", "eGon100RE"]:
        raise ValueError(
            "Invalid input for parameter 'scenario'. Possible options are "
            "'eGon2035' and 'eGon100RE'."
        )

    # set up time index to index data by
    timeindex, timeindex_full = _timeindex_helper_func(
        edisgo_obj,
        timeindex,
        default_year=tools.get_year_based_on_scenario(scenario),
        allow_leap_year=False,
    )

    pth_df = edisgo_obj.topology.loads_df[
        edisgo_obj.topology.loads_df.type == "heat_pump"
    ]

    # get individual heating profiles from oedb
    pth_ind_df = pth_df[
        pth_df.sector.isin(
            ["individual_heating", "individual_heating_resistive_heater"]
        )
    ]
    building_ids = pth_ind_df.building_id.dropna().unique()
    if len(building_ids) > 0:
        residential_profiles_df = get_residential_heat_profiles_per_building(
            building_ids, scenario, engine
        )
        cts_profiles_df = get_cts_profiles_per_building(
            edisgo_obj, scenario, "heat", engine
        )
        # drop CTS profiles for buildings without a heat pump
        buildings_no_hp = [_ for _ in cts_profiles_df.columns if _ not in building_ids]
        cts_profiles_df = cts_profiles_df.drop(columns=buildings_no_hp)
        # add residential and CTS profiles
        individual_heating_df = pd.concat(
            [residential_profiles_df, cts_profiles_df], axis=1
        )
        individual_heating_df = individual_heating_df.groupby(axis=1, level=0).sum()
        # set column names to be heat pump names instead of building IDs
        individual_heating_df = pd.DataFrame(
            {
                hp_name: individual_heating_df.loc[
                    :, pth_ind_df.at[hp_name, "building_id"]
                ]
                for hp_name in pth_ind_df.index
            }
        )
        # set index
        individual_heating_df.index = timeindex_full
    else:
        individual_heating_df = pd.DataFrame(index=timeindex_full)

    # get district heating profiles from oedb
    pth_dh_df = pth_df[
        pth_df.sector.isin(["district_heating", "district_heating_resistive_heater"])
    ]
    if "area_id" in pth_dh_df.columns:
        dh_ids = pth_dh_df.area_id.dropna().unique()
    else:
        dh_ids = []
    if len(dh_ids) > 0:
        dh_profile_df = get_district_heating_heat_demand_profiles(
            dh_ids, scenario, engine
        )
        # set column names to be heat pump names instead of district heating IDs
        dh_profile_df = pd.DataFrame(
            {
                hp_name: dh_profile_df.loc[:, pth_dh_df.at[hp_name, "area_id"]]
                for hp_name in pth_dh_df.index
            }
        )
        # set index
        dh_profile_df.index = timeindex_full
    else:
        dh_profile_df = pd.DataFrame(index=timeindex_full)

    return pd.concat([individual_heating_df, dh_profile_df], axis=1).loc[timeindex, :]


def electricity_demand_oedb(
    edisgo_obj, scenario, engine, timeindex=None, load_names=None
):
    """
    Get electricity demand profiles for all conventional loads from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Conventional loads comprise conventional electricity applications in the
    residential, CTS and industrial sector.
    For more information on how the demand profiles are obtained see functions
    :func:`~.io.timeseries_import.get_residential_electricity_profiles_per_building`,
    :func:`~.io.timeseries_import.get_cts_profiles_per_building` and
    :func:`~.io.timeseries_import.get_industrial_electricity_profiles_per_site`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve demand data. Possible options
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
    load_names : list(str) or None
        Conventional loads (as in index of :attr:`~.network.topology.Topology.loads_df`)
        for which to retrieve electricity demand time series. If none are provided,
        profiles for all conventional loads are returned.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly electricity demand for one year in MW per conventional
        load. Index of the dataframe is a time index. Columns contain the load name as
        in index of :attr:`~.network.topology.Topology.loads_df`.

    """
    if scenario not in ["eGon2035", "eGon100RE"]:
        raise ValueError(
            "Invalid input for parameter 'scenario'. Possible options are "
            "'eGon2035' and 'eGon100RE'."
        )

    # set up time index to index data by
    timeindex, timeindex_full = _timeindex_helper_func(
        edisgo_obj,
        timeindex,
        default_year=tools.get_year_based_on_scenario(scenario),
        allow_leap_year=False,
    )

    # set loads for which to retrieve electricity profiles
    if load_names is None:
        conventional_loads = edisgo_obj.topology.loads_df[
            edisgo_obj.topology.loads_df.type == "conventional_load"
        ]
    else:
        loads_df = edisgo_obj.topology.loads_df.loc[load_names, :]
        conventional_loads = loads_df[loads_df.type == "conventional_load"]

    # get residential electricity profiles from oedb
    residential_loads = conventional_loads[conventional_loads.sector == "residential"]
    res_building_ids = residential_loads.building_id.dropna().unique()
    if len(res_building_ids) > 0:
        residential_profiles_df = get_residential_electricity_profiles_per_building(
            res_building_ids, scenario, engine
        )
        rename_series = (
            residential_loads.loc[:, ["building_id"]]
            .dropna()
            .reset_index()
            .set_index("building_id")
            .iloc[:, 0]
        )
        residential_profiles_df.rename(columns=rename_series, inplace=True)
        residential_profiles_df.index = timeindex_full
    else:
        residential_profiles_df = pd.DataFrame()

    # get CTS electricity profiles from oedb
    cts_loads = conventional_loads[conventional_loads.sector == "cts"]
    cts_building_ids = cts_loads.building_id.dropna().unique()
    if len(cts_building_ids) > 0:
        cts_profiles_df = get_cts_profiles_per_building(
            edisgo_obj, scenario, "electricity", engine
        )
        drop_buildings = [
            _ for _ in cts_profiles_df.columns if _ not in cts_building_ids
        ]
        cts_profiles_df = cts_profiles_df.drop(columns=drop_buildings)
        # set column names to be load names instead of building IDs
        rename_series = (
            cts_loads.loc[:, ["building_id"]]
            .dropna()
            .reset_index()
            .set_index("building_id")
            .iloc[:, 0]
        )
        cts_profiles_df.rename(columns=rename_series, inplace=True)
        cts_profiles_df.index = timeindex_full
    else:
        cts_profiles_df = pd.DataFrame()

    # get industrial electricity profiles from oedb
    ind_loads = conventional_loads[conventional_loads.sector == "industrial"]
    ind_building_ids = ind_loads.building_id.dropna().unique()
    if len(ind_building_ids) > 0:
        ind_profiles_df = get_industrial_electricity_profiles_per_site(
            ind_building_ids, scenario, engine
        )
        # set column names to be load names instead of building IDs
        rename_series = (
            ind_loads.loc[:, ["building_id"]]
            .dropna()
            .reset_index()
            .set_index("building_id")
            .iloc[:, 0]
        )
        ind_profiles_df.rename(columns=rename_series, inplace=True)
        ind_profiles_df.index = timeindex_full
    else:
        ind_profiles_df = pd.DataFrame()

    return pd.concat(
        [residential_profiles_df, cts_profiles_df, ind_profiles_df], axis=1
    ).loc[timeindex, :]


def _get_zensus_cells_of_buildings(building_ids, engine):
    """
    Gets zensus cell ID each building is in from oedb.

    Parameters
    ----------
    building_ids : list(int)
        List of building IDs to get zensus cell ID for.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe matching building ID in column 'building_id' (as integer) and
        zensus cell ID in column 'zensus_id' (as integer).

    """
    saio.register_schema("boundaries", engine)
    from saio.boundaries import egon_map_zensus_mvgd_buildings

    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_map_zensus_mvgd_buildings.building_id,
            egon_map_zensus_mvgd_buildings.zensus_population_id.label("zensus_id"),
        ).filter(egon_map_zensus_mvgd_buildings.building_id.in_(building_ids))
        df = pd.read_sql(query.statement, engine, index_col=None)

    # drop duplicated building IDs that exist because
    # egon_map_zensus_mvgd_buildings can contain several entries per building,
    # e.g. for CTS and residential
    return df.drop_duplicates(subset=["building_id"])


def get_residential_heat_profiles_per_building(building_ids, scenario, engine):
    """
    Gets residential heat demand profiles per building.

    Parameters
    ----------
    building_ids : list(int)
        List of building IDs to retrieve heat demand profiles for.
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with residential heat demand profiles per building for one year in an
        hourly resolution in MW. Index contains hour of the year (from 0 to 8759) and
        column names are building ID as integer.

    """

    def _get_peta_demand(zensus_ids, scenario):
        """
        Retrieve annual peta heat demand for residential buildings for either
        'eGon2035' or 'eGon100RE' scenario.

        Parameters
        ----------
        zensus_ids : list(int)
            List of zensus cell IDs to obtain peta demand data for.
        scenario : str
            Scenario for which to retrieve demand data. Possible options
            are 'eGon2035' and 'eGon100RE'.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Annual residential heat demand per zensus cell. Columns of
            the dataframe are zensus_id and demand.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_peta_heat.zensus_population_id.label("zensus_id"),
                egon_peta_heat.demand,
            ).filter(
                egon_peta_heat.sector == "residential",
                egon_peta_heat.scenario == scenario,
                egon_peta_heat.zensus_population_id.in_(zensus_ids),
            )
            df = pd.read_sql(query.statement, query.session.bind, index_col=None)
        return df

    def _get_residential_heat_profile_ids(zensus_ids):
        """
        Retrieve 365 daily heat profiles IDs for all residential buildings in given
        zensus cells.

        Parameters
        ----------
        zensus_ids : list(int)
            List of zensus cell IDs to get profiles for.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Residential daily heat profile ID's per building. Columns of the
            dataframe are zensus_id, building_id, selected_idp_profiles, buildings
            and day_of_year.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_heat_timeseries_selected_profiles.zensus_population_id.label(
                    "zensus_id"
                ),
                egon_heat_timeseries_selected_profiles.building_id,
                egon_heat_timeseries_selected_profiles.selected_idp_profiles,
            ).filter(
                egon_heat_timeseries_selected_profiles.zensus_population_id.in_(
                    zensus_ids
                )
            )
            df = pd.read_sql(query.statement, query.session.bind, index_col=None)

        # add building count per cell
        df = pd.merge(
            left=df,
            right=df.groupby("zensus_id")["building_id"].count().rename("buildings"),
            left_on="zensus_id",
            right_index=True,
        )

        # unnest array of IDs per building
        df = df.explode("selected_idp_profiles")
        # add day of year column by order of list
        df["day_of_year"] = df.groupby("building_id").cumcount() + 1
        return df

    def _get_daily_profiles(profile_ids):
        """
        Get hourly profiles corresponding to given daily profiles IDs.

        Parameters
        ----------
        profile_ids : list(int)
            Daily heat profile ID's.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Residential hourly heat profiles for each given daily profile ID.
            Index of the dataframe contains the profile ID. Columns of the dataframe
            are idp containing the demand value and hour containing the corresponding
            hour of the day.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_heat_idp_pool.index,
                egon_heat_idp_pool.idp,
            ).filter(egon_heat_idp_pool.index.in_(profile_ids))
            df_profiles = pd.read_sql(query.statement, engine, index_col="index")

        # unnest array of profile values per ID
        df_profiles = df_profiles.explode("idp")
        # add column for hour of day
        df_profiles["hour"] = df_profiles.groupby(axis=0, level=0).cumcount() + 1

        return df_profiles

    def _get_daily_demand_share(zensus_ids):
        """
        Get daily annual demand share per zensus cell.

        Parameters
        ----------
        zensus_ids : list(int)
            List of zensus cell IDs to daily demand share for.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Daily annual demand share per zensus cell. Columns of the dataframe
            are zensus_id, day_of_year and daily_demand_share.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_map_zensus_climate_zones.zensus_population_id.label("zensus_id"),
                daily_heat_demand.day_of_year,
                daily_heat_demand.daily_demand_share,
            ).filter(
                egon_map_zensus_climate_zones.climate_zone
                == daily_heat_demand.climate_zone,
                egon_map_zensus_climate_zones.zensus_population_id.in_(zensus_ids),
            )
            df = pd.read_sql(query.statement, query.session.bind, index_col=None)
        return df

    saio.register_schema("demand", engine)
    from saio.demand import egon_daily_heat_demand_per_climate_zone as daily_heat_demand
    from saio.demand import (
        egon_heat_idp_pool,
        egon_heat_timeseries_selected_profiles,
        egon_peta_heat,
    )

    saio.register_schema("boundaries", engine)
    from saio.boundaries import egon_map_zensus_climate_zones

    # get zensus cells
    zensus_cells_df = _get_zensus_cells_of_buildings(building_ids, engine)
    zensus_cells = zensus_cells_df.zensus_id.unique()

    # get peta demand of each zensus cell
    df_peta_demand = _get_peta_demand(zensus_cells, scenario)
    if df_peta_demand.empty:
        logger.info(f"No residential heat demand for buildings: {building_ids}")
        return pd.DataFrame(columns=building_ids)

    # get daily heat profile IDs per building
    df_profiles_ids = _get_residential_heat_profile_ids(zensus_cells)
    # get daily profiles
    df_profiles = _get_daily_profiles(df_profiles_ids["selected_idp_profiles"].unique())
    # get daily demand share of annual demand
    df_daily_demand_share = _get_daily_demand_share(zensus_cells)

    # merge profile IDs to peta demand by zensus ID
    df_profile_merge = pd.merge(
        left=df_peta_demand, right=df_profiles_ids, on="zensus_id"
    )
    # divide demand by number of buildings in zensus cell
    df_profile_merge.demand = df_profile_merge.demand.div(df_profile_merge.buildings)
    df_profile_merge.drop("buildings", axis="columns", inplace=True)

    # merge daily demand to daily profile IDs by zensus ID and day
    df_profile_merge = pd.merge(
        left=df_profile_merge,
        right=df_daily_demand_share,
        on=["zensus_id", "day_of_year"],
    )
    # multiply demand by daily demand share
    df_profile_merge.demand = df_profile_merge.demand.mul(
        df_profile_merge.daily_demand_share
    )
    df_profile_merge.drop("daily_demand_share", axis="columns", inplace=True)
    df_profile_merge = tools.reduce_memory_usage(df_profile_merge)

    # merge daily profiles by profile ID
    df_profile_merge = pd.merge(
        left=df_profile_merge,
        right=df_profiles[["idp", "hour"]],
        left_on="selected_idp_profiles",
        right_index=True,
    )
    # multiply demand by hourly demand share
    df_profile_merge.demand = df_profile_merge.demand.mul(
        df_profile_merge.idp.astype(float)
    )
    df_profile_merge.drop("idp", axis="columns", inplace=True)
    df_profile_merge = tools.reduce_memory_usage(df_profile_merge)

    # pivot to allow aggregation with CTS profiles
    df_profile_merge = df_profile_merge.pivot(
        index=["day_of_year", "hour"],
        columns="building_id",
        values="demand",
    )
    df_profile_merge = df_profile_merge.sort_index().reset_index(drop=True)

    building_ids_res_select = [_ for _ in df_profile_merge.columns if _ in building_ids]
    return df_profile_merge.loc[:, building_ids_res_select]


def get_district_heating_heat_demand_profiles(district_heating_ids, scenario, engine):
    """
    Gets heat demand profiles of district heating networks from oedb.

    Parameters
    ----------
    district_heating_ids : list(int)
        List of district heating area IDs to get heat demand profiles for.
    scenario : str
        Scenario for which to retrieve data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
            Database engine.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with heat demand profiles per district heating network for one year
        in an hourly resolution in MW. Index contains hour of the year (from 1 to 8760)
        and column names are district heating network ID as integer.

    """
    saio.register_schema("demand", engine)
    from saio.demand import egon_timeseries_district_heating

    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_timeseries_district_heating.area_id,
            egon_timeseries_district_heating.dist_aggregated_mw,
        ).filter(
            egon_timeseries_district_heating.area_id.in_(district_heating_ids),
            egon_timeseries_district_heating.scenario == scenario,
        )
        df = pd.read_sql(query.statement, engine, index_col=None)
    # unnest demand profile and make area_id column names
    df = df.explode("dist_aggregated_mw")
    df["hour_of_year"] = df.groupby("area_id").cumcount() + 1
    df = df.pivot(index="hour_of_year", columns="area_id", values="dist_aggregated_mw")

    return df.astype("float")


def get_cts_profiles_per_building(edisgo_obj, scenario, sector, engine):
    """
    Gets CTS heat demand profiles per CTS building for all CTS buildings in MV grid.

    This function is a helper function that should not be but is necessary, as in
    egon_data buildings are mapped to a grid based on the zensus cell they are in
    whereas in ding0 buildings are mapped to a grid based on the geolocation. As it can
    happen that buildings lie outside an MV grid but within a zensus cell that is
    assigned to that MV grid, they are mapped differently in egon_data and ding0.
    This function therefore checks, if there are CTS loads with other grid IDs and if
    so, gets profiles for other grid IDs (by calling
    :func:`~.io.timeseries_import.get_cts_profiles_per_grid` with different grid IDs)
    in order to obtain a demand profile for all CTS loads.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    sector : str
        Demand sector for which profile is calculated: "electricity" or "heat"
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with CTS demand profiles per building for one year in an
        hourly resolution in MW. Index contains hour of the year (from 0 to 8759) and
        column names are building ID as integer.

    """
    saio.register_schema("boundaries", engine)
    from saio.boundaries import egon_map_zensus_mvgd_buildings

    # get MV grid IDs of CTS loads
    cts_loads = edisgo_obj.topology.loads_df[
        (edisgo_obj.topology.loads_df.type == "conventional_load")
        & (edisgo_obj.topology.loads_df.sector == "cts")
    ]
    cts_building_ids = cts_loads.building_id.dropna().unique()
    with session_scope_egon_data(engine) as session:
        query = session.query(
            egon_map_zensus_mvgd_buildings.building_id,
            egon_map_zensus_mvgd_buildings.bus_id,
        ).filter(
            egon_map_zensus_mvgd_buildings.building_id.in_(cts_building_ids),
        )
        df = pd.read_sql(query.statement, engine, index_col="building_id")

    # iterate over grid IDs
    profiles_df = pd.DataFrame()
    for bus_id in df.bus_id.unique():
        profiles_grid_df = get_cts_profiles_per_grid(
            bus_id=bus_id, scenario=scenario, sector=sector, engine=engine
        )
        profiles_df = pd.concat([profiles_df, profiles_grid_df], axis=1)

    # filter CTS loads in grid
    return profiles_df.loc[:, cts_building_ids]


def get_cts_profiles_per_grid(
    bus_id,
    scenario,
    sector,
    engine,
):
    """
    Gets CTS heat or electricity demand profiles per building for all buildings in the
    given MV grid.

    Parameters
    ----------
    bus_id : int
        MV grid ID.
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    sector : str
        Demand sector for which profile is calculated: "electricity" or "heat"
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with CTS demand profiles per building for one year in an
        hourly resolution in MW. Index contains hour of the year (from 0 to 8759) and
        column names are building ID as integer.

    """

    def _get_demand_share():
        """
        Get CTS demand share per building.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Index contains building ID and column 'profile_share' the corresponding
            demand share.

        """
        if sector == "electricity":
            db_table = egon_cts_electricity_demand_building_share
        else:
            db_table = egon_cts_heat_demand_building_share

        with session_scope_egon_data(engine) as session:
            query = session.query(
                db_table.building_id,
                db_table.profile_share,
            ).filter(
                db_table.scenario == scenario,
                db_table.bus_id == bus_id,
            )
            df = pd.read_sql(query.statement, engine, index_col="building_id")
        return df

    def _get_substation_profile():
        """
        Get aggregated CTS demand profile used in eTraGo.

        In case of heat the profile only contains zensus cells with individual heating.
        In order to obtain a profile for the whole MV grid it needs to be scaled by the
        grid's total CTS demand from peta.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Index contains bus ID and columns contain time steps, numbered from 0 to
            8759.

        """
        if sector == "electricity":
            db_table = egon_etrago_electricity_cts
        else:
            db_table = egon_etrago_heat_cts

        with session_scope_egon_data(engine) as session:
            query = session.query(
                db_table.bus_id,
                db_table.p_set,
            ).filter(
                db_table.scn_name == scenario,
                db_table.bus_id == bus_id,
            )
            df = pd.read_sql(query.statement, engine, index_col=None)
        df = pd.DataFrame.from_dict(
            df.set_index("bus_id")["p_set"].to_dict(),
            orient="index",
        )

        if sector == "heat" and not df.empty:
            total_heat_demand = _get_total_heat_demand_grid()
            scaling_factor = total_heat_demand / df.loc[bus_id, :].sum()
            df.loc[bus_id, :] *= scaling_factor

        return df

    def _get_total_heat_demand_grid():
        """
        Returns total annual CTS heat demand for all CTS buildings in the MV grid,
        including the ones connected to a district heating system.

        Returns
        -------
        float
            Total CTS heat demand in MV grid.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_map_zensus_grid_districts.zensus_population_id,
                egon_peta_heat.demand,
            ).filter(
                egon_peta_heat.sector == "service",
                egon_peta_heat.scenario == scenario,
                egon_map_zensus_grid_districts.bus_id == int(bus_id),
                egon_map_zensus_grid_districts.zensus_population_id
                == egon_peta_heat.zensus_population_id,
            )

            df = pd.read_sql(query.statement, engine, index_col=None)
        return df.demand.sum()

    saio.register_schema("demand", engine)

    if sector == "electricity":
        from saio.demand import (
            egon_cts_electricity_demand_building_share,
            egon_etrago_electricity_cts,
        )

        df_cts_substation_profiles = _get_substation_profile()
        if df_cts_substation_profiles.empty:
            return
        df_demand_share = _get_demand_share()

    elif sector == "heat":
        from saio.demand import (
            egon_cts_heat_demand_building_share,
            egon_etrago_heat_cts,
            egon_peta_heat,
        )

        saio.register_schema("boundaries", engine)
        from saio.boundaries import egon_map_zensus_grid_districts

        df_cts_substation_profiles = _get_substation_profile()
        if df_cts_substation_profiles.empty:
            return
        df_demand_share = _get_demand_share()

    else:
        raise KeyError("Sector needs to be either 'electricity' or 'heat'")

    shares = df_demand_share["profile_share"]
    profile_ts = df_cts_substation_profiles.loc[bus_id]
    building_profiles = np.outer(profile_ts, shares)
    building_profiles = pd.DataFrame(
        building_profiles, index=profile_ts.index, columns=shares.index
    )

    # sanity checks
    if sector == "electricity":
        check_sum_profile = building_profiles.sum().sum()
        check_sum_db = df_cts_substation_profiles.sum().sum()
        if not np.isclose(check_sum_profile, check_sum_db, atol=1e-1):
            logger.warning("Total CTS electricity demand does not match.")
    if sector == "heat":
        check_sum_profile = building_profiles.sum().sum()
        check_sum_db = _get_total_heat_demand_grid()
        if not np.isclose(check_sum_profile, check_sum_db, atol=1e-1):
            logger.warning("Total CTS heat demand does not match.")
    return building_profiles


def get_residential_electricity_profiles_per_building(building_ids, scenario, engine):
    """
    Gets residential electricity demand profiles per building.

    Parameters
    ----------
    building_ids : list(int)
        List of building IDs to retrieve electricity demand profiles for.
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with residential electricity demand profiles per building for one year
        in an hourly resolution in MW. Index contains hour of the year (from 0 to 8759)
        and column names are building ID as integer.

    """

    def _get_scaling_factors_of_zensus_cells(zensus_ids):
        """
        Get profile scaling factors per zensus cell for specified scenario.

        Parameters
        ----------
        zensus_ids : list(int)
            List of zensus cell IDs to get scaling factors for.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with zensus cell ID in index and respective scaling factor in
            column factor.

        """
        with session_scope_egon_data(engine) as session:
            if scenario == "eGon2035":
                query = session.query(
                    egon_household_electricity_profile_in_census_cell.cell_id,
                    egon_household_electricity_profile_in_census_cell.factor_2035.label(
                        "factor"
                    ),
                ).filter(
                    egon_household_electricity_profile_in_census_cell.cell_id.in_(
                        zensus_ids
                    )
                )
            else:
                query = session.query(
                    egon_household_electricity_profile_in_census_cell.cell_id,
                    egon_household_electricity_profile_in_census_cell.factor_2050.label(
                        "factor"
                    ),
                ).filter(
                    egon_household_electricity_profile_in_census_cell.cell_id.in_(
                        zensus_ids
                    )
                )
        return pd.read_sql(query.statement, engine, index_col="cell_id")

    def _get_profile_ids_of_buildings(building_ids):
        """
        Get profile IDs per building.

        Parameters
        ----------
        building_ids : list(int)
            List of building IDs to retrieve profile IDs for.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with building ID in column building_id, zensus cell ID in column
            cell_id and corresponding profile IDs in column profile_id.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_household_electricity_profile_of_buildings.building_id,
                egon_household_electricity_profile_of_buildings.cell_id,
                egon_household_electricity_profile_of_buildings.profile_id,
            ).filter(
                egon_household_electricity_profile_of_buildings.building_id.in_(
                    building_ids
                )
            )
        return pd.read_sql(query.statement, engine, index_col=None)

    def _get_profiles(profile_ids):
        """
        Get hourly household electricity demand profiles for specified profile IDs.

        Parameters
        ----------
        profile_ids: list(str)
            (type)a00..(profile number) with number having exactly 4 digits

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
             Hourly household demand profiles with profile ID as column name and index
             containing hour of the year (from 0 to 8759).

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                iee_household_load_profiles.load_in_wh, iee_household_load_profiles.type
            ).filter(iee_household_load_profiles.type.in_(profile_ids))
        df = pd.read_sql(query.statement, engine, index_col="type")

        # convert array to dataframe
        df_converted = pd.DataFrame.from_records(df["load_in_wh"], index=df.index).T

        return df_converted

    saio.register_schema("demand", engine)
    from saio.demand import (
        egon_household_electricity_profile_in_census_cell,
        egon_household_electricity_profile_of_buildings,
        iee_household_load_profiles,
    )

    # get zensus cells of buildings
    zensus_ids_buildings = _get_zensus_cells_of_buildings(building_ids, engine)
    zensus_ids = zensus_ids_buildings.zensus_id.unique()

    # get profile scaling factors per zensus cell
    scaling_factors_zensus_cells = _get_scaling_factors_of_zensus_cells(zensus_ids)

    # get profile IDs per building and merge scaling factors
    profile_ids_buildings = _get_profile_ids_of_buildings(building_ids)
    profile_ids = profile_ids_buildings.profile_id.unique()
    profile_ids_buildings = profile_ids_buildings.join(
        scaling_factors_zensus_cells, on="cell_id"
    )
    if profile_ids_buildings.empty:
        logger.info("No residential electricity demand.")
        return pd.DataFrame()

    # get hourly profiles per profile ID
    profiles_df = _get_profiles(profile_ids)

    # calculate demand profile per building
    ts_df = pd.DataFrame()
    for building_id, df in profile_ids_buildings.groupby(by="building_id"):
        load_ts_building = (
            profiles_df.loc[:, df["profile_id"]].sum(axis=1)
            * df["factor"].iloc[0]
            / 1e6  # from Wh to MWh
        ).to_frame(name=building_id)
        ts_df = pd.concat([ts_df, load_ts_building], axis=1).dropna(axis=1)

    return ts_df


def get_industrial_electricity_profiles_per_site(site_ids, scenario, engine):
    """
    Gets industrial electricity demand profiles per site and OSM area.

    Parameters
    ----------
    site_ids : list(int)
        List of industrial site and OSM IDs to retrieve electricity demand profiles for.
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe with industrial electricity demand profiles per site and OSM area for
        one year in an hourly resolution in MW. Index contains hour of the year (from 0
        to 8759) and column names are site ID as integer.

    """

    def _get_load_curves_sites(site_ids):
        """
        Get industrial load profiles for sites for specified scenario.

        Parameters
        ----------
        site_ids : list(int)
            List of industrial site IDs to retrieve electricity demand profiles for.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with site ID in column site_id and electricity profile as list
            in column p_set.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_sites_ind_load_curves_individual.site_id,
                egon_sites_ind_load_curves_individual.p_set,
            ).filter(
                egon_sites_ind_load_curves_individual.scn_name == scenario,
                egon_sites_ind_load_curves_individual.site_id.in_(site_ids),
            )
        return pd.read_sql(query.statement, engine, index_col=None)

    def _get_load_curves_areas(site_ids):
        """
        Get industrial load profiles for OSM areas for specified scenario.

        Parameters
        ----------
        site_ids : list(int)
            List of industrial OSM IDs to retrieve electricity demand profiles for.

        Returns
        -------
        :pandas:`pandas.DataFrame<DataFrame>`
            Dataframe with OSM ID in column site_id and electricity profile as list
            in column p_set.

        """
        with session_scope_egon_data(engine) as session:
            query = session.query(
                egon_osm_ind_load_curves_individual.osm_id.label("site_id"),
                egon_osm_ind_load_curves_individual.p_set,
            ).filter(
                egon_osm_ind_load_curves_individual.scn_name == scenario,
                egon_osm_ind_load_curves_individual.osm_id.in_(site_ids),
            )
        return pd.read_sql(query.statement, engine, index_col=None)

    saio.register_schema("demand", engine)
    from saio.demand import (
        egon_osm_ind_load_curves_individual,
        egon_sites_ind_load_curves_individual,
    )

    # get profiles of sites and OSM areas
    profiles_sites = _get_load_curves_sites(site_ids)
    profiles_areas = _get_load_curves_areas(site_ids)

    # concat profiles
    profiles_df = pd.concat([profiles_sites, profiles_areas])
    # add time step column
    profiles_df["time_step"] = len(profiles_df) * [np.arange(0, 8760)]
    # un-nest p_set and pivot so that time_step becomes index and site_id the
    # name of the columns
    return (
        profiles_df.explode(["p_set", "time_step"])
        .pivot(index="time_step", columns="site_id", values="p_set")
        .astype("float")
    )
