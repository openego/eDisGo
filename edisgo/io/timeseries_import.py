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


def feedin_oedb_legacy(config_data, weather_cell_ids, timeindex):
    """
    Import feed-in time series data for wind and solar power plants from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    config_data : :class:`~.tools.config.Config`
        Configuration data from config files, relevant for information of
        which data base table to retrieve feed-in data from.
    weather_cell_ids : list(int)
        List of weather cell id's (integers) to obtain feed-in data for.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Feed-in data is currently only provided for weather year 2011. If
        timeindex contains a different year, the data is reindexed.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly time series for active power feed-in per
        generator type (wind or solar, in column level 0) and weather cell
        (in column level 1), normalized to a capacity of 1 MW.

    """

    def _retrieve_timeseries_from_oedb(session, timeindex):
        """Retrieve time series from oedb"""
        # ToDo: add option to retrieve subset of time series instead of whole
        #  year
        # ToDo: find the reference power class for mvgrid/w_id and insert
        #  instead of 4
        feedin_sqla = (
            session.query(orm_feedin.w_id, orm_feedin.source, orm_feedin.feedin)
            .filter(orm_feedin.w_id.in_(weather_cell_ids))
            .filter(orm_feedin.power_class.in_([0, 4]))
            .filter(orm_feedin_version)
            .filter(orm_feedin.weather_year.in_(timeindex.year.unique().values))
        )

        feedin = pd.read_sql_query(
            feedin_sqla.statement, session.bind, index_col=["source", "w_id"]
        )
        return feedin

    if config_data["data_source"]["oedb_data_source"] == "model_draft":
        orm_feedin_name = config_data["model_draft"]["res_feedin_data"]
        orm_feedin = model_draft.__getattribute__(orm_feedin_name)
        orm_feedin_version = 1 == 1
    else:
        orm_feedin_name = config_data["versioned"]["res_feedin_data"]
        orm_feedin = supply.__getattribute__(orm_feedin_name)
        orm_feedin_version = orm_feedin.version == config_data["versioned"]["version"]

    if timeindex is None:
        timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")

    with session_scope() as session:
        feedin = _retrieve_timeseries_from_oedb(session, timeindex)

    if feedin.empty:
        raise ValueError(
            "The year you inserted could not be imported from "
            "the oedb. So far only 2011 is provided. Please "
            "check website for updates."
        )

    feedin.sort_index(axis=0, inplace=True)

    recasted_feedin_dict = {}
    for type_w_id in feedin.index:
        recasted_feedin_dict[type_w_id] = feedin.loc[type_w_id, :].values[0]

    # Todo: change when possibility for other years is given
    conversion_timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
    feedin = pd.DataFrame(recasted_feedin_dict, index=conversion_timeindex)

    # rename 'wind_onshore' and 'wind_offshore' to 'wind'
    new_level = [
        _ if _ not in ["wind_onshore"] else "wind" for _ in feedin.columns.levels[0]
    ]
    feedin.columns = feedin.columns.set_levels(new_level, level=0)

    feedin.columns.rename("type", level=0, inplace=True)
    feedin.columns.rename("weather_cell_id", level=1, inplace=True)

    return feedin.loc[timeindex]


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

    return feedin_df.loc[timeindex, :]


def load_time_series_demandlib(config_data, timeindex):
    """
    Get normalized sectoral electricity load time series using the
    `demandlib <https://github.com/oemof/demandlib/>`_.

    Resulting electricity load profiles hold time series of hourly conventional
    electricity demand for the sectors residential, cts, agricultural
    and industrial. Time series are normalized to a consumption of 1 MWh per
    year.

    Parameters
    ----------
    config_data : :class:`~.tools.config.Config`
        Configuration data from config files, relevant for industrial load
        profiles.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Timesteps for which to generate load time series.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with conventional electricity load time series for sectors
        residential, cts, agricultural and industrial.
        Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns
        hold the sector type.

    """
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
            config_data["demandlib"]["day_start"].hour,
            config_data["demandlib"]["day_start"].minute,
            0,
        ),
        pm=datetime.time(
            config_data["demandlib"]["day_end"].hour,
            config_data["demandlib"]["day_end"].minute,
            0,
        ),
        profile_factors={
            "week": {
                "day": config_data["demandlib"]["week_day"],
                "night": config_data["demandlib"]["week_night"],
            },
            "weekend": {
                "day": config_data["demandlib"]["weekend_day"],
                "night": config_data["demandlib"]["weekend_night"],
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


def cop_oedb(engine, weather_cell_ids, year=None):
    """
    Get COP (coefficient of performance) time series data from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    weather_cell_ids : list(int) or list(float)
        List (or array) of weather cell IDs to obtain COP data for.
    year : int
        COP data is only provided for the weather year 2011. If a different year
        is provided through this parameter, the data is reindexed. A leap year can
        currently not be handled. In case a leap year is given, the time index is
        set for 2011!

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly COP time series in p.u. per weather cell. Index of the
        dataframe is a time index. Columns contain the weather cell ID as integer.

    """
    # set up time index to index COP data by
    if year is None:
        timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
    else:
        if pd.Timestamp(year, 1, 1).is_leap_year:
            year = 2011
            logger.warning(
                "A leap year was given to 'cop_oedb' function. This is currently not "
                "valid. The year data is indexed by is therefore set to the default "
                "value of 2011."
            )
        timeindex = pd.date_range(f"1/1/{year}", periods=8760, freq="H")

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
    cop = pd.DataFrame({w_id: ts.cop for w_id, ts in cop.iterrows()}, index=timeindex)

    return cop


def heat_demand_oedb(edisgo_obj, scenario, engine, year=None):
    """
    Get heat demand profiles for heat pumps from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Heat demand data is returned for all heat pumps in the grid.
    For more information on how individual heat demand profiles are obtained see
    functions :attr:`~.io.timeseries_import.get_residential_heat_profiles_per_building`
    and :attr:`~.io.timeseries_import.get_cts_profiles_per_building`.
    For more information on how district heating heat demand profiles are obtained see
    function :attr:`~.io.timeseries_import.get_district_heating_heat_demand_profiles`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    year : int or None
        Year to index heat demand data by. Per default this is set to 2035 in case
        of the 'eGon2035' and to 2045 in case of the 'eGon100RE' scenario.
        A leap year can currently not be handled. In case a leap year is given, the
        time index is set according to the chosen scenario.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly heat demand for one year in MW per heat pump. Index of the
        dataframe is a time index. Columns contain the heat pump name as in index of
        :attr:`~.network.topology.Topology.loads_df`.

    """
    # set up time index to index data by
    if year is None:
        year = tools.get_year_based_on_scenario(scenario)
        if year is None:
            raise ValueError(
                "Invalid input for parameter 'scenario'. Possible options are "
                "'eGon2035' and 'eGon100RE'."
            )
    else:
        if pd.Timestamp(year, 1, 1).is_leap_year:
            logger.warning(
                "A leap year was given to 'heat_demand_oedb' function. This is "
                "currently not valid. The year the data is indexed by is therefore set "
                "to the default value of 2011."
            )
            return heat_demand_oedb(edisgo_obj, scenario, engine, year=None)
    timeindex = pd.date_range(f"1/1/{year}", periods=8760, freq="H")

    hp_df = edisgo_obj.topology.loads_df[
        edisgo_obj.topology.loads_df.type == "heat_pump"
    ]

    # get individual heating profiles from oedb
    building_ids = hp_df.building_id.dropna().unique()
    if len(building_ids) > 0:
        residential_profiles_df = get_residential_heat_profiles_per_building(
            building_ids, scenario, engine
        )
        cts_profiles_df = get_cts_profiles_per_building(
            edisgo_obj.topology.id, scenario, "heat", engine
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
        rename_series = (
            hp_df.loc[:, ["building_id"]]
            .dropna()
            .reset_index()
            .set_index("building_id")
            .iloc[:, 0]
        )
        individual_heating_df.rename(columns=rename_series, inplace=True)
        # set index
        individual_heating_df.index = timeindex
    else:
        individual_heating_df = pd.DataFrame(index=timeindex)

    # get district heating profiles from oedb
    dh_ids = hp_df.district_heating_id.dropna().unique()
    if len(dh_ids) > 0:
        dh_profile_df = get_district_heating_heat_demand_profiles(
            dh_ids, scenario, engine
        )
        # set column names to be heat pump names instead of district heating IDs
        rename_series = (
            hp_df.loc[:, ["district_heating_id"]]
            .dropna()
            .reset_index()
            .set_index("district_heating_id")
            .iloc[:, 0]
        )
        dh_profile_df.rename(columns=rename_series, inplace=True)
        # set index
        dh_profile_df.index = timeindex
    else:
        dh_profile_df = pd.DataFrame(index=timeindex)

    return pd.concat([individual_heating_df, dh_profile_df], axis=1)


def electricity_demand_oedb(edisgo_obj, scenario, engine, year=None, load_names=None):
    """
    Get electricity demand profiles for all conventional loads from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Conventional loads comprise conventional electricity applications in the
    residential, CTS and industrial sector.
    For more information on how the demand profiles are obtained see functions
    :attr:`~.io.timeseries_import.get_residential_electricity_profiles_per_building`,
    :attr:`~.io.timeseries_import.get_cts_profiles_per_building` and
    :attr:`~.io.timeseries_import.get_industrial_electricity_profiles_per_site`.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve demand data. Possible options
        are 'eGon2035' and 'eGon100RE'.
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.
    year : int or None
        Year to index electricity demand data by. Per default this is set to 2035 in
        case of the 'eGon2035' and to 2045 in case of the 'eGon100RE' scenario.
        A leap year can currently not be handled. In case a leap year is given, the
        time index is set according to the chosen scenario.
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
    # set up time index to index data by
    if year is None:
        year = tools.get_year_based_on_scenario(scenario)
        if year is None:
            raise ValueError(
                "Invalid input for parameter 'scenario'. Possible options are "
                "'eGon2035' and 'eGon100RE'."
            )
    else:
        if pd.Timestamp(year, 1, 1).is_leap_year:
            logger.warning(
                "A leap year was given to 'electricity_demand_oedb' function. This is "
                "currently not valid. The year the data is indexed by is therefore set "
                "to the default value of 2011."
            )
            return electricity_demand_oedb(
                edisgo_obj, scenario, engine, year=None, load_names=load_names
            )
    timeindex = pd.date_range(f"1/1/{year}", periods=8760, freq="H")

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
        residential_profiles_df.index = timeindex
    else:
        residential_profiles_df = pd.DataFrame()

    # get CTS electricity profiles from oedb
    cts_loads = conventional_loads[conventional_loads.sector == "cts"]
    cts_building_ids = cts_loads.building_id.dropna().unique()
    if len(cts_building_ids) > 0:
        cts_profiles_df = get_cts_profiles_per_building(
            edisgo_obj.topology.id, scenario, "electricity", engine
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
        cts_profiles_df.index = timeindex
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
        ind_profiles_df.index = timeindex
    else:
        ind_profiles_df = pd.DataFrame()

    return pd.concat(
        [residential_profiles_df, cts_profiles_df, ind_profiles_df], axis=1
    )


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

    return df


def get_cts_profiles_per_building(
    bus_id,
    scenario,
    sector,
    engine,
):
    """
    Gets CTS heat demand profiles per building for all buildings in MV grid.

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
            query = session.query(db_table.building_id, db_table.profile_share,).filter(
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
            query = session.query(db_table.bus_id, db_table.p_set,).filter(
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
    for building_id, df in profile_ids_buildings.groupby(by=["building_id"]):
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
    return profiles_df.explode(["p_set", "time_step"]).pivot(
        index="time_step", columns="site_id", values="p_set"
    )
