import datetime
import os

import pandas as pd

from demandlib import bdew as bdew
from demandlib import particular_profiles as profiles
from workalendar.europe import Germany

from edisgo.tools import session_scope

if "READTHEDOCS" not in os.environ:
    from egoio.db_tables import model_draft, supply


def feedin_oedb(config_data, weather_cell_ids, timeindex):
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


def load_time_series_demandlib(config_data, timeindex):
    """
    Get normalized sectoral electricity load time series using the
    `demandlib <https://github.com/oemof/demandlib/>`_.

    Resulting electricity load profiles hold time series of hourly conventional
    electricity demand for the sectors residential, retail, agricultural
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
        residential, retail, agricultural and industrial.
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
            "g0": "retail",
            "h0": "residential",
            "l0": "agricultural",
            "i0": "industrial",
        },
        inplace=True,
    )

    return elec_demand.loc[timeindex]


def cop_oedb(config_data, weather_cell_ids=None, timeindex=None):
    """
    Get COP (coefficient of performance) time series data from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    config_data : :class:`~.tools.config.Config`
        Configuration data from config files, relevant for information of
        which data base table to retrieve COP data from.
    weather_cell_ids : list(int)
        List of weather cell id's (integers) to obtain COP data for.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        COP data is only provided for the weather year 2011. If
        timeindex contains a different year, the data is reindexed.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly COP time series in p.u. per weather cell.

    """
    raise NotImplementedError

    # if timeindex is None:
    #     timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
    #
    # if weather_cell_ids is None:
    #     # get weather cells in grid district
    #     pass
    #
    # import saio
    # saio.register_schema("supply", engine)
    # from saio.supply import egon_era5_renewable_feedin
    #
    # # get cop from database
    # with db.session_scope() as session:
    #     query = session.query(
    #         egon_era5_renewable_feedin.w_id,
    #         egon_era5_renewable_feedin.feedin.label("cop"),
    #     ).filter(
    #         egon_era5_renewable_feedin.carrier == "heat_pump_cop"
    #     ).filter(
    #         egon_era5_renewable_feedin.w_id.in_(weather_cell_ids)
    #     )
    #
    #     cop = pd.read_sql(
    #         query.statement, query.session.bind, index_col="w_id"
    #     )
    #
    # # convert dataframe to have weather cell ID as column name and time index
    # cop = pd.DataFrame(
    #     {w_id: ts.cop for w_id, ts in cop.iterrows()},
    #     index=timeindex
    # )
    #
    # return cop


def heat_demand_oedb(config_data, building_ids, timeindex=None):
    """
    Get heat demand time series data from the
    `OpenEnergy DataBase <https://openenergy-platform.org/dataedit/schemas>`_.

    Parameters
    ----------
    config_data : :class:`~.tools.config.Config`
        Configuration data from config files, relevant for information of
        which data base table to retrieve data from.
    building_ids : list(int)
        List of building IDs to obtain heat demand for.
    timeindex : :pandas:`pandas.DatetimeIndex<DatetimeIndex>`
        Heat demand data is only provided for the weather year 2011. If
        timeindex contains a different year, the data is reindexed.

    Returns
    -------
    :pandas:`pandas.DataFrame<DataFrame>`
        DataFrame with hourly heat demand time series in MW per building ID.

    """

    def _get_CTS_demand():
        """
        Gets CTS heat demand time series for each building in building_ids.

        First, the share of the total CTS demand in the NUTS 3 region for
        each building and the total CTS heat demand time series in the NUTS 3 region
        are retrieved. To obtain the heat demand time series per building the building's
        share is multiplied with the total time series.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            CTS heat demand time series per building with building IDs in columns
            and time steps as index.

        """
        raise NotImplementedError
        # # get share per building
        # # get total demand time series
        # # multiply
        # pd.read_sql(
        #     query.statement, session.bind, index_col="id"
        # )
        #
        # return

    def _get_household_demand():
        """
        Gets household heat demand time series for each building in building_ids.

        Returns
        --------
        :pandas:`pandas.DataFrame<DataFrame>`
            Household heat demand time series per building with building IDs in columns
            and time steps as index.

        """
        raise NotImplementedError
        # # get representative days per building
        # # get time series for representative days
        # # set up time series per building
        # pd.read_sql(
        #     query.statement, session.bind, index_col="id"
        # )
        #
        # return

    raise NotImplementedError
    # ToDo Also include large heat pumps for district heating that don't have
    #  a building ID

    # if timeindex is None:
    #     timeindex = pd.date_range("1/1/2011", periods=8760, freq="H")
    #
    # with session_scope() as session:
    #     # get heat demand from database
    #     heat_demand_CTS = _get_CTS_demand()
    #     heat_demand_households = _get_household_demand()
    #
    # heat_demand = heat_demand_CTS + heat_demand_households
    # heat_demand.sort_index(axis=0, inplace=True)
    #
    # return heat_demand
